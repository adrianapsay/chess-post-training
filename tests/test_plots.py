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
