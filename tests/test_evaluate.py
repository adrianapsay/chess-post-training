import chess
import pandas as pd

from chesslm.evaluate import grade, is_correct, summarize

# Back-rank position: both Ra8# and Rb8# are checkmate.
BACK_RANK = "6k1/5ppp/8/8/8/8/8/RR4K1 w - - 0 1"


class FakeScorer:
    """Stands in for Stockfish, so these tests run without the engine."""

    def centipawn_loss(self, fen, move):
        return 0


def puzzle(answer_uci, mate_in_1):
    return {"puzzle_id": "x", "band": "<1200", "fen": BACK_RANK, "answer_uci": answer_uci, "mate_in_1": mate_in_1}


def test_any_mate_counts_in_mate_in_1():
    board = chess.Board(BACK_RANK)
    other_mate = chess.Move.from_uci("b1b8")
    assert is_correct(board, other_mate, puzzle("a1a8", mate_in_1=True))
    assert not is_correct(board, other_mate, puzzle("a1a8", mate_in_1=False))
    assert board.fen() == BACK_RANK  # checking for mate must not change the board


def test_grade_reply():
    row = grade(puzzle("a1a8", mate_in_1=True), "I play Ra8#", FakeScorer())
    assert row["move"] == "a1a8" and row["outcome"] == "legal" and row["correct"]

    row = grade(puzzle("a1a8", mate_in_1=True), "Qd8", FakeScorer())
    assert row["outcome"] == "illegal" and not row["correct"] and row["cp_loss"] is None


def test_summarize():
    graded = pd.DataFrame({
        "band": ["<1200", "<1200", "2400+", "2400+"],
        "outcome": ["legal", "legal", "illegal", "unparseable"],
        "correct": [True, False, False, False],
        "cp_loss": [0, 300, None, None],
        "seconds": [0.5] * 4,
        "n_tokens": [4] * 4,
    })
    s = summarize(graded)
    assert s["puzzle_accuracy"] == 0.25
    assert s["accuracy_by_band"] == {"2400+": 0.0, "<1200": 0.5}
    assert s["legal_rate"] == 0.5 and s["illegal_rate"] == 0.25 and s["unparseable_rate"] == 0.25
    assert s["acpl"] == 150  # legal moves only


class FakePositionScorer:
    """Stockfish stand-in for game positions: the best move is Ra8, and any other move loses 300."""

    def best_move(self, fen):
        return chess.Move.from_uci("a1a8"), 0

    def centipawn_loss(self, fen, move):
        return 0 if move.uci() == "a1a8" else 300


def test_grade_and_summarize_game_positions():
    from chesslm.evaluate import grade_position, summarize_positions

    position = {"game_id": "g", "ply": 30, "fen": BACK_RANK, "move_uci": "b1b8"}
    rows = [grade_position(position, reply, FakePositionScorer(), blunder_cp=200)
            for reply in ["a1a8", "b1b8", "g1h1", "Qd8", "hmm"]]
    graded = pd.DataFrame(rows)
    assert list(graded.outcome) == ["legal", "legal", "legal", "illegal", "unparseable"]
    assert list(graded.matches_stockfish) == [True, False, False, False, False]
    assert list(graded.matches_player) == [False, True, False, False, False]
    summary = summarize_positions(graded)
    assert summary["legal_rate"] == 0.6
    assert summary["acpl"] == 200  # (0 + 300 + 300) / 3
    assert summary["blunder_rate_of_legal"] == 2 / 3
    assert summary["blunder_rate"] == 0.4
