import chess
import pandas as pd

from chesslm.play import RandomOpponent, opening_lines, play_games, summarize_games

PROMPT = {"notation": "uci", "history_moves": 0, "show_legal_moves": False}


def first_legal_move(rows):
    """A stand-in model: always plays the first legal move (sorted), answering in UCI."""
    return [sorted(m.uci() for m in chess.Board(r["fen"]).legal_moves)[0] for r in rows]


def test_every_game_finishes_and_each_opening_is_played_with_both_colors():
    records = play_games(first_legal_move, RandomOpponent(seed=0), ["e2e4 e7e5", "d2d4 d7d5"], PROMPT,
                         max_moves=40, seed=0)
    assert len(records) == 4
    assert [r["model_color"] for r in records] == ["white", "black", "white", "black"]
    assert all(r["result"] in {"win", "draw", "loss"} for r in records)
    assert all(r["illegal"] == r["unparseable"] == 0 for r in records)
    assert all(r["pgn"].startswith("[Event") for r in records)


def test_invalid_answers_are_replaced_and_counted():
    records = play_games(lambda rows: ["Ke9"] * len(rows), RandomOpponent(seed=0), ["e2e4 e7e5"], PROMPT,
                         max_moves=5, seed=0)
    for r in records:
        assert r["unparseable"] == r["model_moves"] > 0  # "Ke9" is not a move at all
    summary = summarize_games(records)
    assert summary["invalid_move_rate"] == 1.0
    assert summary["wins"] + summary["draws"] + summary["losses"] == 2


def test_checkmate_is_a_win_for_the_side_that_mates():
    # After 1.f3 e5 2.g4 the model (Black) mates with Qh4#.
    records = play_games(lambda rows: ["d8h4"] * len(rows), RandomOpponent(seed=0), ["f2f3 e7e5 g2g4"], PROMPT,
                         max_moves=40, seed=0)
    black_game = records[1]
    assert black_game["result"] == "win" and black_game["end_reason"] == "checkmate"


def test_opening_lines_are_seeded_and_cut_to_length():
    positions = pd.DataFrame({"game_id": ["a", "a", "b", "c"],
                              "history_uci": ["e2e4 e7e5 g1f3 b8c6", "e2e4 e7e5 g1f3 b8c6 f1b5",
                                              "d2d4 d7d5 c2c4 e7e6", "c2c4"]})
    lines = opening_lines(positions, n_openings=2, opening_plies=3, seed=1)
    assert lines == opening_lines(positions, n_openings=2, opening_plies=3, seed=1)
    assert sorted(lines) == ["d2d4 d7d5 c2c4", "e2e4 e7e5 g1f3"]  # game c is too short
