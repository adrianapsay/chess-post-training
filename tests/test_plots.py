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
