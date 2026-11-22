"""Shared steps for the experiment notebooks (08 onwards): grading, saving, and readable tables.

Notebooks 01-07 keep their own copies of these steps and are left unchanged, so their
saved results stay exactly as they were. Every experiment writes its figures and tables
to its own folder, results/figures/<run name>/.
"""

import json
import shutil
import tempfile
from pathlib import Path

import chess
import pandas as pd

import chesslm.evaluate
from chesslm.config import REPO_ROOT
from chesslm.engine import MoveScorer, map_with_engines
from chesslm.evaluate import evaluate, evaluate_positions, generate_replies, save_run, summarize, summarize_positions
from chesslm.play import RandomOpponent, StockfishOpponent, play_games, summarize_games
from chesslm.prompts import build_prompt

REPO_RESULTS = REPO_ROOT / "results"
RESULTS = REPO_RESULTS  # where outputs go; a quick check moves this to a temporary folder


def use_quick_check_folder():
    """Send every output of this session to a temporary folder, so a quick check never changes results/.

    The summary tables are copied there first, so comparisons with earlier runs still work.
    """
    global RESULTS
    RESULTS = Path(tempfile.gettempdir()) / "chess_quick_check" / "results"
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    for name in ["summary.csv", "positions_summary.csv", "games_summary.csv"]:
        if (REPO_RESULTS / name).exists():
            shutil.copy(REPO_RESULTS / name, RESULTS / name)
    chesslm.evaluate.RESULTS_DIR = RESULTS
    print(f"Quick check: outputs go to {RESULTS}, not to results/. Set QUICK_CHECK = False for the real run.")

# Column names a reader can follow without the code, for every printed or saved table.
PUZZLE_COLUMNS = {
    "puzzle_accuracy": "Puzzle accuracy (%)", "accuracy_<1200": "<1200", "accuracy_1200-1800": "1200-1800",
    "accuracy_1800-2400": "1800-2400", "accuracy_2400+": "2400+", "legal_rate": "Legal (%)",
    "illegal_rate": "Illegal (%)", "unparseable_rate": "Unparseable (%)", "acpl": "ACPL (cp)",
}


def figures_dir(run_name):
    """results/figures/<run_name>/, created if needed. Every figure and table of one experiment goes here."""
    path = RESULTS / "figures" / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def grade_puzzles(model, tokenizer, split, run_name, configs, adapter, limit=None):
    """Grade a model on the dev or test puzzles, save results/<run_name>.json and a summary.csv row.

    configs is a dict with the "model", "prompt" and "eval" settings. Prints one line of results.
    """
    puzzles = pd.read_csv(REPO_ROOT / "data" / "eval" / f"puzzles_{split}.csv")
    if limit:
        puzzles = puzzles.head(limit)
    eval_cfg = configs["eval"]
    with MoveScorer(eval_cfg["stockfish"]) as scorer:
        graded = evaluate(model, tokenizer, puzzles, configs["prompt"], eval_cfg["generation"], scorer)
    summary = summarize(graded)
    config = {"model": configs["model"], "adapter": adapter, "prompt": configs["prompt"], "eval": eval_cfg,
              "split": split}
    save_run(run_name, graded, summary, config)
    acpl = summary["acpl"] if summary["acpl"] is not None else float("nan")
    print(f"{run_name}: {len(puzzles)} puzzles | accuracy {summary['puzzle_accuracy']:.1%} | "
          f"legal {summary['legal_rate']:.1%} | ACPL {acpl:.0f}")
    return graded, summary


def save_training(run_name, config, compute, log, start_adapter=None, extra=None):
    """Save results/<run_name>_training.json: the exact settings used, the compute, and the training log."""
    record = {"config": config, "start_adapter": str(start_adapter) if start_adapter else None,
              "compute": compute, **(extra or {}), "log": log.to_dict("records")}
    with open(RESULTS / f"{run_name}_training.json", "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(json.dumps(compute, indent=2))


def puzzle_table(runs, folder=None, name="puzzle_results"):
    """Test-set puzzle results for the given runs, with readable column names (percentages and centipawns).

    If a folder is given, the table is also saved there as <name>.md and <name>.csv.
    """
    summary = pd.read_csv(RESULTS / "summary.csv").set_index("run")
    table = summary.loc[runs, list(PUZZLE_COLUMNS)].rename(columns=PUZZLE_COLUMNS)
    percent = [c for c in table.columns if c != "ACPL (cp)"]
    table[percent] = (table[percent] * 100).round(1)
    table["ACPL (cp)"] = table["ACPL (cp)"].round(0)
    table.index.name = "Model"
    if folder:
        save_table(table, folder, name, "1,000 test puzzles (250 per rating band). Rating-band columns are "
                                        "puzzle accuracy (%) in that band. ACPL is over legal answers only.")
    return table


def save_table(table, folder, name, note):
    """Save a table as Markdown (for reading on GitHub, with a one-line note) and as CSV (for reuse)."""
    table.to_csv(folder / f"{name}.csv")
    header = "| " + " | ".join([table.index.name or ""] + list(map(str, table.columns))) + " |"
    lines = [header, "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {idx} | " + " | ".join("–" if pd.isna(v) else str(v) for v in row) + " |"
              for idx, row in table.iterrows()]
    (folder / f"{name}.md").write_text("\n".join(lines) + f"\n\n{note}\n")


def load_graded(run_name):
    """The graded rows (one per puzzle) saved for a run, as a table."""
    path = RESULTS / f"{run_name}.json"
    if not path.exists():  # during a quick check, earlier runs are still in results/
        path = REPO_RESULTS / f"{run_name}.json"
    with open(path) as f:
        return pd.DataFrame(json.load(f)["rows"])


def load_training_log(run_name):
    """The training log saved for a run (results/<run>_training.json), as a table."""
    path = RESULTS / f"{run_name}_training.json"
    if not path.exists():  # during a quick check, earlier runs are still in results/
        path = REPO_RESULTS / f"{run_name}_training.json"
    with open(path) as f:
        return pd.DataFrame(json.load(f)["log"])


def move_kinds(graded, fens):
    """Share of a model's legal answers that are pawn moves, captures, or checks.

    fens maps each row of `graded` (by position in the table) to its position. Used to see
    whether a model plays forcing moves (captures, checks) or quiet ones (the grpo_v1 problem).
    """
    rows = []
    for fen, row in zip(fens, graded.to_dict("records")):
        if row["outcome"] != "legal":
            continue
        board, move = chess.Board(fen), chess.Move.from_uci(row["move"])
        rows.append({"pawn move": board.piece_type_at(move.from_square) == chess.PAWN,
                     "capture": board.is_capture(move), "check": board.gives_check(move)})
    return pd.DataFrame(rows).mean() * 100


# ---------------------------------------------------------------- held-out game positions and full games

def reply_function(model, tokenizer, prompt_cfg, gen_cfg):
    """reply(rows) -> the model's greedy reply to each position, with the usual prompt and settings."""
    def reply(rows):
        prompts = [build_prompt(tokenizer, row, prompt_cfg) for row in rows]
        return generate_replies(model, tokenizer, prompts, gen_cfg)[0]
    return reply


def update_summary(name, key_columns, row):
    """Add one row to results/<name>.csv, replacing an earlier row with the same key (e.g. a re-run)."""
    path = RESULTS / f"{name}.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        same = (table[key_columns] == pd.Series({k: row[k] for k in key_columns})).all(axis=1)
        table = table[~same]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.round(4).to_csv(path, index=False)


def grade_game_positions(model, tokenizer, run_name, configs, positions):
    """Grade a model on the held-out game positions; save results/<run>_positions.json and a summary row."""
    eval_cfg = configs["eval"]
    with MoveScorer(eval_cfg["stockfish"]) as scorer:
        graded = evaluate_positions(model, tokenizer, positions, configs["prompt"], eval_cfg["generation"], scorer,
                                    eval_cfg["game_positions"]["blunder_cp"])
    summary = summarize_positions(graded)
    with open(RESULTS / f"{run_name}_positions.json", "w") as f:
        json.dump({"run": run_name, "config": configs, "summary": summary, "rows": graded.to_dict("records")},
                  f, indent=2, default=str)
    update_summary("positions_summary", ["run"], {"run": run_name, **summary})
    print(f"{run_name}: {len(graded)} game positions | legal {summary['legal_rate']:.1%} | "
          f"blunders {summary['blunder_rate']:.1%} of all answers | same as Stockfish {summary['stockfish_match_rate']:.1%}")
    return graded, summary


def random_move_positions(positions, eval_cfg, workers, seed):
    """The same position metrics for a player that picks a random legal move: a reference point."""
    import random
    rng = random.Random(seed)
    moves = [rng.choice(list(chess.Board(fen).legal_moves)).uci() for fen in positions.fen]
    blunder_cp = eval_cfg["game_positions"]["blunder_cp"]

    def score(item, scorer):
        fen, uci = item
        best, _ = scorer.best_move(fen)
        return best.uci() == uci, scorer.centipawn_loss(fen, chess.Move.from_uci(uci))

    scored = map_with_engines(score, zip(positions.fen, moves), eval_cfg["stockfish"], workers)
    losses = pd.Series([loss for _, loss in scored])
    return {"n_positions": len(positions), "legal_rate": 1.0, "illegal_rate": 0.0, "unparseable_rate": 0.0,
            "acpl": losses.mean(), "blunder_rate_of_legal": (losses >= blunder_cp).mean(),
            "blunder_rate": (losses >= blunder_cp).mean(),
            "stockfish_match_rate": sum(match for match, _ in scored) / len(scored),
            "player_match_rate": sum(m == p for m, p in zip(moves, positions.move_uci)) / len(moves)}


def play_against_opponents(model, tokenizer, run_name, configs, openings):
    """Play every opening twice against each opponent in the config. Saves results/<run>_games.json."""
    games_cfg, records = configs["eval"]["games"], {}
    reply = reply_function(model, tokenizer, configs["prompt"], configs["eval"]["generation"])
    for name in games_cfg["opponents"]:
        opponent = (StockfishOpponent(configs["eval"]["stockfish"], games_cfg["stockfish_depth"])
                    if name == "stockfish" else RandomOpponent(games_cfg["seed"]))
        try:
            records[name] = play_games(reply, opponent, openings, configs["prompt"], games_cfg["max_moves"],
                                       games_cfg["seed"])
        finally:
            opponent.close()
        summary = summarize_games(records[name])
        update_summary("games_summary", ["run", "opponent"], {"run": run_name, "opponent": name, **summary})
        print(f"{run_name} vs {name}: {summary['wins']} wins, {summary['draws']} draws, {summary['losses']} losses | "
              f"invalid moves {summary['invalid_move_rate']:.1%} of the model's moves")
    with open(RESULTS / f"{run_name}_games.json", "w") as f:
        json.dump({"run": run_name, "config": configs, "games": records}, f, indent=2, default=str)
    return records


def compare_with(new_run, old_run):
    """Paired comparison of two models on the same puzzles: how many each solves that the other doesn't.

    The p-value (sign test) is the chance of a split at least this uneven if neither model were
    better. Below 0.05 is the usual line for "a real difference, not noise".
    """
    from chesslm.analysis import sign_test

    both = load_graded(new_run).merge(load_graded(old_run), on="puzzle_id", suffixes=("_new", "_old"))
    gained = int((both.correct_new & ~both.correct_old).sum())
    lost = int((~both.correct_new & both.correct_old).sum())
    return {"model": new_run, "compared with": old_run, "puzzles": len(both),
            "solved only by this model": gained, "solved only by the other": lost,
            "p-value (sign test)": float(f"{sign_test(gained, lost):.2g}")}
