import os
import shutil

import chess
import pytest

from chesslm.engine import MoveScorer

pytestmark = pytest.mark.skipif(
    not (os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")), reason="Stockfish not installed"
)
CFG = {"path": None, "depth": 10, "cp_loss_cap": 1000}

# After 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6: Qxf7# wins at once; Qh3 throws the win away.
SCHOLARS_MATE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_best_move_has_zero_loss():
    with MoveScorer(CFG) as scorer:
        best, _ = scorer.best_move(SCHOLARS_MATE)
        assert best.uci() == "h5f7"
        assert scorer.centipawn_loss(SCHOLARS_MATE, best) == 0


def test_missing_a_mate_costs_the_capped_maximum():
    with MoveScorer(CFG) as scorer:
        loss = scorer.centipawn_loss(SCHOLARS_MATE, chess.Move.from_uci("h5h3"))
        assert loss == CFG["cp_loss_cap"]


def test_results_are_cached():
    with MoveScorer(CFG) as scorer:
        move = chess.Move.from_uci("h5h3")
        scorer.centipawn_loss(SCHOLARS_MATE, move)
        assert (SCHOLARS_MATE, "h5h3") in scorer.loss_cache


def test_map_with_engines_keeps_order():
    from chesslm.engine import map_with_engines

    fens = [SCHOLARS_MATE, chess.STARTING_FEN, SCHOLARS_MATE]
    best = map_with_engines(lambda fen, scorer: scorer.best_move(fen)[0].uci(), fens, CFG, workers=2)
    assert best[0] == best[2] == "h5f7"
    assert chess.Move.from_uci(best[1]) in chess.Board().legal_moves


def test_stockfish_labels_are_legal_moves():
    import pandas as pd

    from chesslm.data import add_stockfish_moves

    positions = pd.DataFrame({"fen": [SCHOLARS_MATE, chess.STARTING_FEN]})
    labelled = add_stockfish_moves(positions, CFG, workers=2)
    assert list(labelled.best_uci)[0] == "h5f7" and list(labelled.best_san)[0] == "Qxf7#"
    assert chess.Board().parse_san(labelled.best_san[1]).uci() == labelled.best_uci[1]
