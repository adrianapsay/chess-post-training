"""DPO preference pairs: for each position, a better move ("chosen") and a worse one ("rejected").

The chosen move is always Stockfish's best move. The rejected move comes from one of two
sources (configs/dpo.yaml, rejected_source):
- random (dpo_v1): a random legal move that loses at least `rejected_min_cp_loss` centipawns
- sft_mistakes (dpo_v2, dpo_v3): the SFT model's own answer, when it is illegal or loses
  at least `rejected_min_cp_loss` centipawns
Prompts are the same as in SFT and evaluation.
"""

import random

import chess
import pandas as pd
from datasets import Dataset

from chesslm.engine import map_with_engines
from chesslm.prompts import build_messages, move_to_text, parse_move


def rejected_move(fen, scorer, min_cp_loss, rng):
    """A random legal move that loses at least min_cp_loss centipawns, or None if there is none."""
    moves = list(chess.Board(fen).legal_moves)
    rng.shuffle(moves)
    for move in moves:
        loss = scorer.centipawn_loss(fen, move)
        if loss >= min_cp_loss:
            return move, loss
    return None, None


def build_pairs(positions, prompt_cfg, dpo_cfg, scorer):
    """Build one preference pair per position. Positions with no clearly worse move are skipped."""
    rng = random.Random(dpo_cfg["seed"])
    notation = prompt_cfg["notation"]
    rows = []
    for row in positions.to_dict("records"):
        board = chess.Board(row["fen"])
        best, _ = scorer.best_move(row["fen"])
        worse, loss = rejected_move(row["fen"], scorer, dpo_cfg["rejected_min_cp_loss"], rng)
        if worse is None:
            continue
        rows.append({
            "prompt": build_messages(row, prompt_cfg),
            "chosen": [{"role": "assistant", "content": move_to_text(board, best, notation)}],
            "rejected": [{"role": "assistant", "content": move_to_text(board, worse, notation)}],
            "fen": row["fen"],
            "rejected_cp_loss": loss,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pairs from the model's own mistakes

def judge_reply(position, reply, notation, min_cp_loss, scorer):
    """Turn one model reply into a preference pair, if the reply is a mistake. Otherwise None.

    A mistake is an illegal or unparseable reply (kept exactly as written), or a legal move
    that loses at least min_cp_loss centipawns. The chosen answer is Stockfish's best move.
    """
    board = chess.Board(position["fen"])
    best, _ = scorer.best_move(position["fen"])
    move, outcome = parse_move(reply, board)
    if move is None:
        kind, loss, rejected = outcome, None, reply.strip()
    else:
        loss = scorer.centipawn_loss(position["fen"], move)
        if loss < min_cp_loss:
            return None  # a good answer: nothing to learn from it
        kind, rejected = "blunder", move_to_text(board, move, notation)
    if not rejected:
        return None  # an empty reply cannot be a rejected answer
    return {"chosen_text": move_to_text(board, best, notation), "rejected_text": rejected,
            "rejected_kind": kind, "rejected_cp_loss": loss}


def build_mistake_pairs(positions, reply_fn, prompt_cfg, dpo_cfg, stockfish_cfg, batch_positions=2000):
    """Preference pairs whose rejected answer is a mistake the model itself makes.

    reply_fn(rows) returns the model's (greedy) reply to each position. Positions are
    answered in batches, in order, until dpo_cfg["target_pairs"] mistakes are found.
    Returns (pairs, number of positions answered).
    """
    notation, target = prompt_cfg["notation"], dpo_cfg["target_pairs"]
    rows, answered = [], 0
    records = positions.to_dict("records")
    for start in range(0, len(records), batch_positions):
        batch = records[start : start + batch_positions]
        replies = reply_fn(batch)
        answered += len(batch)
        judged = map_with_engines(
            lambda item, scorer: judge_reply(item[0], item[1], notation, dpo_cfg["rejected_min_cp_loss"], scorer),
            zip(batch, replies), stockfish_cfg, dpo_cfg["stockfish_workers"])
        for position, reply, pair in zip(batch, replies, judged):
            if pair is None:
                continue
            rows.append({
                "prompt": build_messages(position, prompt_cfg),
                "chosen": [{"role": "assistant", "content": pair["chosen_text"]}],
                "rejected": [{"role": "assistant", "content": pair["rejected_text"]}],
                "fen": position["fen"],
                "model_reply": reply,
                "rejected_kind": pair["rejected_kind"],
                "rejected_cp_loss": pair["rejected_cp_loss"],
            })
        print(f"{answered:,} positions answered -> {len(rows):,} mistakes so far")
        if len(rows) >= target:
            break
    return pd.DataFrame(rows[:target]), answered


def to_dpo_dataset(pairs):
    """The three columns TRL's DPOTrainer expects: prompt, chosen, rejected."""
    return Dataset.from_pandas(pairs[["prompt", "chosen", "rejected"]], preserve_index=False)
