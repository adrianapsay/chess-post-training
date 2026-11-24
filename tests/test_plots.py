import matplotlib

matplotlib.use("Agg")

import pandas as pd

from chesslm import plots


def _label_rotations(fig):
    return {label.get_rotation() for ax in fig.axes if ax.get_visible() for label in ax.get_xticklabels()}


def test_crowded_model_names_are_written_vertically(tmp_path):
    names = [f"a_long_model_name_{i}" for i in range(10)]
    table = pd.DataFrame({"Legal (%)": range(10)}, index=names)
    fig = plots.metric_panels(table, [("Legal (%)", "legal (%)", "{:.0f}")], tmp_path / "crowded.png")
    assert _label_rotations(fig) == {90.0}


def test_few_short_names_stay_horizontal(tmp_path):
    table = pd.DataFrame({"Legal (%)": [40, 70]}, index=["base", "sft_v1"])
    fig = plots.metric_panels(table, [("Legal (%)", "legal (%)", "{:.0f}")], tmp_path / "roomy.png")
    assert _label_rotations(fig) == {0.0}


def test_new_summary_figures_draw(tmp_path):
    effects = [{"label": "more data", "puzzles": (1.6, -0.4, 3.6), "cp": (38, 12, 64)},
               {"label": "not measured", "puzzles": (0.5, -1.0, 2.0), "cp": (None, None, None)}]
    plots.paired_effects(effects, [("puzzles", "accuracy", "{:+.1f}"), ("cp", "cp saved", "{:+.0f}")],
                         tmp_path / "effects.png")
    table = pd.DataFrame({"acc": [7.8, 9.4, 13.2], "acc low": [6.3, 7.7, 11.2], "acc high": [9.6, 11.4, 15.4]},
                         index=["a", "b", "c"])
    plots.two_by_two(table, {"no list": ["a", "b"], "list": ["c", "missing"]}, ["20k", "100k"], [("acc", "accuracy")],
                     tmp_path / "two_by_two.png", references={"acc": ("random", 5.6)})
    points = pd.DataFrame({"cp": [40, -10], "cp low": [20, -30], "cp high": [60, 10], "sf": [-2, 4], "sf low": [-3, 2],
                           "sf high": [-1, 6], "group": ["RL", "SFT"]}, index=["grpo", "sft"])
    plots.tradeoff_map(points, tmp_path / "tradeoff.png")
    bars = pd.DataFrame({"positions": [35, 0], "games": [61, None]}, index=["sft_v1", "sft_legal_moves"])
    plots.grouped_bars(bars, "invalid (%)", tmp_path / "bars.png")
    assert all((tmp_path / name).exists() for name in ["effects.png", "two_by_two.png", "tradeoff.png", "bars.png"])


def test_puzzle_before_after_gif(tmp_path):
    puzzles = pd.DataFrame({"puzzle_id": ["a", "b"], "fen": ["6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"] * 2,
                            "answer_uci": ["a1a8"] * 2, "rating": [900, 1500]})
    before = pd.DataFrame({"puzzle_id": ["a", "b"], "reply": ["Ke9", "h2h3"], "move": [None, "h2h3"],
                           "outcome": ["illegal", "legal"], "correct": [False, False]}).set_index("puzzle_id")
    after = pd.DataFrame({"puzzle_id": ["a", "b"], "reply": ["a1a8", "a1a8"], "move": ["a1a8", "a1a8"],
                          "outcome": ["legal", "legal"], "correct": [True, True]}).set_index("puzzle_id")
    path = plots.puzzle_before_after(puzzles, before, after, tmp_path / "before_after.gif",
                                     labels=("before: base", "after: best"), seconds_per_puzzle=0.5)
    from PIL import Image
    assert Image.open(path).n_frames == 2


def test_paper_figures_are_drawn_at_their_printed_size(tmp_path):
    effects = [{"label": "player's moves", "group": "more data", "color": "#2a78d6", "star": True,
                "puzzles": (1.6, 0.4, 3.6)},
               {"label": "legal list", "group": "more data", "puzzles": (4.3, 2.1, 6.6)},
               {"label": "β 0.2", "group": "RL", "puzzles": (-2.8, -4.4, -1.3)}]
    with plots.paper_style():
        fig = plots.paired_effects(effects, [("puzzles", "accuracy", "{:+.1f}")], tmp_path / "effects.pdf",
                                   figsize=(6.75, 2))
        headers = {t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_fontweight() == "bold"}
        table = pd.DataFrame({"acc": [2.8, 7.8, 9.4], "acc low": [1.9, 6.3, 7.7], "acc high": [4.0, 9.6, 11.4]},
                             index=["base", "a", "b"])
        three = plots.two_by_two(table, {"player": ["base", "a", "b"]}, ["0", "20k", "100k"], [("acc", "accuracy")],
                                 tmp_path / "sft.pdf", figsize=(3.25, 2), values=False)
        points = pd.DataFrame({"cp": [40, -10], "cp low": [20, -30], "cp high": [60, 10], "sf": [-2, 4],
                               "sf low": [-3, 2], "sf high": [-1, 6], "group": ["RL", "SFT"]}, index=["grpo", "sft"])
        tradeoff = plots.tradeoff_map(points, tmp_path / "tradeoff.pdf", figsize=(3.25, 2.9), labels={"grpo"},
                                      colors={"RL": "#8a5cc7", "SFT": "#2a78d6"})
        plots.dumbbells([{"label": "β 0", "puzzles": (7.8, 3.3), "pawns": (1.7, 40.6)}],
                        [("puzzles", "puzzle accuracy (%)"), ("pawns", "pawn moves (%)")], tmp_path / "rl.pdf",
                        figsize=(6.75, 1.2))
        plots.model_tree({"base": (0.1, 1, "", "none"), "sft_v1": (1.2, 1, "SFT, 20k", "SFT")}, [("base", "sft_v1")],
                         tmp_path / "tree.pdf", figsize=(3.25, 2), colors={"none": "#7a7872", "SFT": "#2a78d6"},
                         legend=["none", "SFT"])
        log = pd.DataFrame({"step": [10, 20], "rewards/chosen": [0.0, -0.1], "rewards/rejected": [0.0, -0.3]})
        plots.dpo_probability_ratios({"dpo_v3": log}, 0.1, tmp_path / "dpo.pdf", figsize=(3.25, 2))
    assert tuple(fig.get_size_inches()) == (6.75, 2) and tuple(three.get_size_inches()) == (3.25, 2)
    assert headers == {"more data", "RL"}
    assert [t.get_text() for t in tradeoff.axes[0].texts if t.get_text() in ("grpo", "sft")] == ["grpo"]
    assert plots._TEXT_SCALE == 1.0 and plots.SURFACE == "#fcfcfb"  # the paper style ends with the block
    assert all((tmp_path / f"{name}.pdf").exists() for name in ["effects", "sft", "tradeoff", "rl", "tree", "dpo"])


def test_latex_table_escapes_text_and_bolds_the_best_value(tmp_path):
    table = pd.DataFrame({"What changed": ["β = 0.2", "100k, 50% more"], "Puzzles (%)": ["5.0", "17.5"],
                          "ACPL": ["335", "344*"]}, index=["grpo_v5", "sft_100k_legal_moves"])
    table.index.name = "Model"
    text = plots.latex_table(table, tmp_path / "table.tex", "Every model.", "tab:models",
                             groups=[("RL", ["grpo_v5"]), ("SFT", ["sft_100k_legal_moves"])],
                             bold={"Puzzles (%)": "max", "ACPL": "min"}, column_groups=[("", 2), ("Results", 2)])
    assert r"sft\_100k\_legal\_moves" in text and r"$\beta$ = 0.2" in text and r"50\% more" in text
    assert r"\textbf{17.5}" in text and r"\textbf{335}" in text and r"\textbf{5.0}" not in text
    assert r"\emph{RL}" in text and r"\begin{tabular}{llrr}" in text
    assert r"\multicolumn{2}{c}{Results}" in text and r"\cmidrule(lr){3-4}" in text
    assert (tmp_path / "table.tex").read_text() == text
