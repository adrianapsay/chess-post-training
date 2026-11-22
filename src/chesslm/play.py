"""Full games between a model and an opponent (Stockfish, or a player that moves at random).

All games are played at the same time: whenever it is the model's turn in several games,
those positions are answered in one batch, which is much faster than one game at a time.

Rules (settings in configs/eval.yaml, under games):
- Each game starts from the position after the first few moves of a held-out game, so the
  games differ from each other. Every opening is played twice, once with each color.
- The model sees the same prompt as everywhere else, and its move is read the same way.
- If its answer is illegal or unparseable, a random legal move is played instead and the
  mistake is counted. This lets every game finish, so a game result measures move quality
  and the mistake count measures legality.
- A game still running after max_moves (full moves) counts as a draw.
"""

import random

import chess
import chess.engine
import chess.pgn

from chesslm.engine import open_engine
from chesslm.prompts import parse_move


class Game:
    """One game in progress: the board, the model's color, and what went wrong along the way."""

    def __init__(self, opening_uci, model_color, game_index):
        self.board = chess.Board()
        self.san_history = []
        for uci in opening_uci.split():
            self.push(chess.Move.from_uci(uci))
        self.model_color = model_color
        self.index = game_index
        self.opening_plies = len(self.board.move_stack)
        self.model_moves = self.illegal = self.unparseable = 0

    def push(self, move):
        self.san_history.append(self.board.san(move))
        self.board.push(move)

    def prompt_row(self):
        """The position in the form build_messages expects (the same as a training row)."""
        return {"fen": self.board.fen(),
                "history_uci": " ".join(m.uci() for m in self.board.move_stack),
                "history_san": " ".join(self.san_history)}


def opening_lines(positions, n_openings, opening_plies, seed):
    """n different openings: the first `opening_plies` moves (UCI) of held-out games, seeded.

    Popular openings appear in many games, so duplicates are removed before sampling.
    """
    games = positions.drop_duplicates("game_id")
    games = games[games.history_uci.str.split().str.len() >= opening_plies]
    lines = games.history_uci.map(lambda h: " ".join(h.split()[:opening_plies])).drop_duplicates()
    return list(lines.sample(n=n_openings, random_state=seed))


class StockfishOpponent:
    """Stockfish at a fixed shallow depth. Deterministic, so every run plays the same games."""

    def __init__(self, stockfish_cfg, depth):
        self.engine = open_engine(stockfish_cfg["path"])
        self.limit = chess.engine.Limit(depth=depth)

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class RandomOpponent:
    """Plays a uniformly random legal move (seeded). The weakest possible opponent."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))

    def close(self):
        pass


def model_result(game, end_reason):
    """win / draw / loss from the model's point of view."""
    if end_reason == "checkmate":
        return "loss" if game.board.turn == game.model_color else "win"  # the side to move is mated
    return "draw"


def end_reason(game, max_moves):
    """Why the game is over, or None if it goes on."""
    outcome = game.board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.termination.name.lower()  # checkmate, stalemate, insufficient_material, ...
    if game.board.fullmove_number > max_moves:
        return "move limit"
    return None


def play_games(reply_fn, opponent, openings, prompt_cfg, max_moves, seed):
    """Play every opening twice (model as White, then as Black). Returns one dict per game.

    reply_fn(rows) returns the model's reply to each position (see Game.prompt_row).
    """
    rng = random.Random(seed)  # for the fallback move after an illegal answer
    games = [Game(line, color, 2 * i + (color == chess.BLACK))
             for i, line in enumerate(openings) for color in (chess.WHITE, chess.BLACK)]
    finished = {}
    while len(finished) < len(games):
        active = [g for g in games if g.index not in finished]

        # The model moves in every game where it is its turn, all in one batch.
        waiting = [g for g in active if g.board.turn == g.model_color]
        replies = reply_fn([g.prompt_row() for g in waiting]) if waiting else []
        for game, reply in zip(waiting, replies):
            move, outcome = parse_move(reply, game.board)
            game.model_moves += 1
            if move is None:
                game.illegal += outcome == "illegal"
                game.unparseable += outcome == "unparseable"
                move = rng.choice(list(game.board.legal_moves))
            game.push(move)

        # Then the opponent moves wherever it is its turn and the game is not over.
        for game in active:
            if end_reason(game, max_moves) is None and game.board.turn != game.model_color:
                game.push(opponent.move(game.board))

        for game in active:
            reason = end_reason(game, max_moves)
            if reason is not None:
                finished[game.index] = reason

    return [game_record(g, finished[g.index]) for g in games]


def game_record(game, reason):
    pgn = chess.pgn.Game.from_board(game.board)
    pgn.headers["White"] = "model" if game.model_color == chess.WHITE else "opponent"
    pgn.headers["Black"] = "opponent" if game.model_color == chess.WHITE else "model"
    result = model_result(game, reason)
    model_won = "1-0" if game.model_color == chess.WHITE else "0-1"
    model_lost = "0-1" if game.model_color == chess.WHITE else "1-0"
    pgn.headers["Result"] = {"win": model_won, "loss": model_lost, "draw": "1/2-1/2"}[result]
    pgn.headers["Termination"] = reason
    return {
        "game": game.index,
        "model_color": "white" if game.model_color == chess.WHITE else "black",
        "result": result,
        "end_reason": reason,
        "plies": len(game.board.move_stack) - game.opening_plies,
        "model_moves": game.model_moves,
        "illegal": game.illegal,
        "unparseable": game.unparseable,
        "pgn": str(pgn),
    }


def summarize_games(records):
    """Totals over a set of games. Score = (wins + half the draws) / games, the usual chess score."""
    n = len(records)
    count = lambda result: sum(r["result"] == result for r in records)
    moves = sum(r["model_moves"] for r in records)
    mistakes = sum(r["illegal"] + r["unparseable"] for r in records)
    return {
        "games": n,
        "wins": count("win"),
        "draws": count("draw"),
        "losses": count("loss"),
        "score": (count("win") + 0.5 * count("draw")) / n,
        "model_moves": moves,
        "invalid_move_rate": mistakes / moves if moves else None,  # illegal or unparseable, replaced at random
        "average_game_length": sum(r["plies"] for r in records) / n,
    }
