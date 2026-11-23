"""Figures for comparing runs. One shared style so every figure in the report looks the same."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colors follow the thing shown, in a fixed order: first run blue, second orange, third green, fourth yellow.
# Figures with more than four runs (the final comparison) continue with purple, teal, red, grey.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc7", "#1f9fb4", "#c8423b", "#7a7872"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BANDS = ["<1200", "1200-1800", "1800-2400", "2400+"]


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def _turn_crowded_labels(fig, gap=4):
    """Write the x-axis labels vertically on every panel where they would otherwise overlap (e.g. many model names).

    Returns True if any panel was changed. gap: the smallest space (in pixels) allowed between two labels.
    """
    fig.canvas.draw()  # places the labels, so their sizes can be measured
    renderer = fig.canvas.get_renderer()
    turned = False
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        boxes = sorted((label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_text()),
                       key=lambda box: box.x0)
        if any(left.x1 + gap > right.x0 for left, right in zip(boxes, boxes[1:])):
            ax.tick_params(axis="x", labelrotation=90)
            turned = True
    return turned


def _finish(fig, path, title):
    """Add an optional title above the whole figure, then save it. Titles help readers who see a figure on its own."""
    if title:
        fig.suptitle(title, color=TEXT, fontsize=11)
    fig.tight_layout()
    if _turn_crowded_labels(fig):
        fig.tight_layout()  # make room for the vertical labels
    fig.savefig(path, dpi=200)
    return fig


def _label_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8, color=TEXT, xytext=(0, 2), textcoords="offset points")


def histogram(values, xlabel, path, edges=None, title=None):
    """Distribution of one quantity (e.g. ratings). Optional dashed lines mark boundaries such as rating bands."""
    fig, ax = plt.subplots(figsize=(7, 3), facecolor=SURFACE)
    ax.hist(values, bins=40, color=COLORS[0], edgecolor=SURFACE, linewidth=1)
    for edge in edges or []:
        ax.axvline(edge, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel(xlabel, color=MUTED)
    _style(ax, "count")
    return _finish(fig, path, title)


def compare_metrics(summary, runs, path, title=None):
    """Three panels (accuracy, legal-move rate, ACPL), one bar per run. Each panel has its own axis."""
    panels = [("puzzle_accuracy", "puzzle accuracy (%)", 100, "{:.1f}"),
              ("legal_rate", "legal-move rate (%)", 100, "{:.1f}"),
              ("acpl", "ACPL (centipawns, lower is better)", 1, "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), facecolor=SURFACE)
    for ax, (column, label, scale, fmt) in zip(axes, panels):
        values = [summary.loc[summary.run == run, column].iloc[0] * scale for run in runs]
        bars = ax.bar(runs, values, color=COLORS[: len(runs)], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, fmt)
        _style(ax, label)
        ax.set_ylim(bottom=0)  # bars always start at zero
    return _finish(fig, path, title)


def accuracy_by_band(summary, runs, path, title=None):
    """Puzzle accuracy per rating band, one bar per run in each band."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        row = summary[summary.run == run].iloc[0]
        values = [row[f"accuracy_{band}"] * 100 for band in BANDS]
        positions = [b + (i - (len(runs) - 1) / 2) * width for b in range(len(BANDS))]
        bars = ax.bar(positions, values, width, label=run, color=COLORS[i], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, "{:.1f}")
    ax.set_xticks(range(len(BANDS)), BANDS)
    ax.set_xlabel("puzzle rating band", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, "puzzle accuracy (%)")
    ax.set_ylim(bottom=0)
    return _finish(fig, path, title)


def outcome_breakdown(summary, runs, path, title=None):
    """For each run, the share of answers that were legal, illegal, or unparseable (stacked to 100%)."""
    outcomes = [("legal_rate", "legal"), ("illegal_rate", "illegal"), ("unparseable_rate", "unparseable")]
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=SURFACE)
    left = [0.0] * len(runs)
    for color, (column, label) in zip(COLORS, outcomes):
        values = [summary.loc[summary.run == run, column].iloc[0] * 100 for run in runs]
        ax.barh(runs, values, left=left, label=label, color=color, edgecolor=SURFACE, linewidth=2)
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 6:  # label segments wide enough to hold text
                ax.text(start + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TEXT)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of answers (%)", color=MUTED)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _finish(fig, path, title)


def loss_curve(log, path, title=None):
    """Training and validation loss over training steps."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    train = log.dropna(subset=["loss"])
    ax.plot(train.step, train.loss, color=COLORS[0], linewidth=2, label="training loss")
    if "eval_loss" in log:  # short runs may end before the first validation check
        val = log.dropna(subset=["eval_loss"])
        ax.plot(val.step, val.eval_loss, color=COLORS[1], linewidth=2, marker="o", markersize=8, label="validation loss")
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, "loss")
    return _finish(fig, path, title)


def reward_curve(log, path, title=None):
    """Average reward of the sampled answers over RL training steps (one series, so no legend)."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    rows = log.dropna(subset=["reward"])
    ax.plot(rows.step, rows.reward, color=COLORS[0], linewidth=2)
    ax.set_xlabel("training step", color=MUTED)
    _style(ax, "average reward of sampled answers")
    return _finish(fig, path, title)


# ---------------------------------------------------------------- interpreting results (notebook 07)

# One hue from light to dark, for ordered groups such as "how much a move loses".
BLUES = ["#d3e3f7", "#8fb7ea", "#4a8bdc", "#1d5aa8"]


def accuracy_with_intervals(table, chance, path, title=None):
    """Puzzle accuracy per run with its 95% interval, and a dashed line at the random-move accuracy."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    runs = list(table.index)
    values = table.accuracy * 100
    errors = [values - table.low * 100, table.high * 100 - values]
    bars = ax.bar(runs, values, color=COLORS[: len(runs)], edgecolor=SURFACE, linewidth=2,
                  yerr=errors, capsize=6, error_kw={"ecolor": TEXT, "elinewidth": 1.2})
    for bar, high in zip(bars, table.high * 100):
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, high),
                    ha="center", va="bottom", fontsize=8, color=TEXT, xytext=(0, 2), textcoords="offset points")
    ax.axhline(chance * 100, color=MUTED, linewidth=1.2, linestyle="--", label=f"random legal move ({chance:.1%})")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))  # above the bars
    _style(ax, "puzzle accuracy (%)")
    ax.set_ylim(bottom=0)
    return _finish(fig, path, title)


def accuracy_decomposition(table, chance, path, title=None):
    """Accuracy = (share of legal answers) x (accuracy when the answer is legal), one panel for each factor."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), facecolor=SURFACE)
    runs = list(table.index)
    panels = [("legal_rate", "legal answers (%)"), ("accuracy_when_legal", "accuracy when legal (%)")]
    for ax, (column, label) in zip(axes, panels):
        bars = ax.bar(runs, table[column] * 100, color=COLORS[: len(runs)], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, "{:.1f}")
        _style(ax, label)
        ax.set_ylim(bottom=0)
    axes[1].axhline(chance * 100, color=MUTED, linewidth=1.2, linestyle="--",
                    label=f"random legal move ({chance:.1%})")
    axes[1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))  # above the bars
    return _finish(fig, path, title)


def stacked_shares(shares, colors, xlabel, path, title=None):
    """One horizontal bar per row of `shares` (a table of percentages summing to 100), split by column."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(shares) + 1.2), facecolor=SURFACE)
    rows = list(shares.index)[::-1]  # first row on top
    left = [0.0] * len(rows)
    for color, column in zip(colors, shares.columns):
        values = [shares.loc[row, column] for row in rows]
        ax.barh(rows, values, left=left, label=column, color=color, edgecolor=SURFACE, linewidth=2)
        dark = color in (BLUES[2], BLUES[3])
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 5:  # label segments wide enough to hold text
                ax.text(start + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else TEXT)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.legend(frameon=False, ncol=len(shares.columns), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    _style(ax, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _finish(fig, path, title)


def token_probability_curve(log, path, title=None):
    """SFT loss turned into a probability: exp(-loss) is the average (geometric mean) probability per answer token."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    train = log.dropna(subset=["loss"])
    val = log.dropna(subset=["eval_loss"])
    ax.plot(train.step, np.exp(-train.loss) * 100, color=COLORS[0], linewidth=2, label="training examples")
    ax.plot(val.step, np.exp(-val.eval_loss) * 100, color=COLORS[1], linewidth=2, marker="o", markersize=8,
            label="validation examples")
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, "probability per answer token (%)")
    ax.set_ylim(0, 100)
    return _finish(fig, path, title)


def dpo_diagnostics(log, beta, path, title=None, rejected_label="rejected (loses 200+ cp)"):
    """Left: how often the model prefers the chosen move (50% = no preference).
    Right: the probability of the chosen and rejected answers, relative to the SFT model (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    train = log.dropna(subset=["rewards/accuracies"])
    val = log.dropna(subset=["eval_rewards/accuracies"])

    ax = axes[0]
    ax.plot(train.step, train["rewards/accuracies"] * 100, color=COLORS[0], linewidth=2, label="training pairs")
    ax.plot(val.step, val["eval_rewards/accuracies"] * 100, color=COLORS[1], linewidth=2, marker="o", markersize=8,
            label="held-out pairs")
    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "prefers the chosen move (%)")

    ax = axes[1]
    for column, label, color in [("rewards/chosen", "chosen (Stockfish's best)", COLORS[2]),
                                 ("rewards/rejected", rejected_label, COLORS[3])]:
        ax.plot(train.step, np.exp(train[column] / beta), color=color, linewidth=2, label=label)
    ax.axhline(1, color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}×")
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False, loc="lower left")
    _style(ax, "probability vs. the SFT model (×)")
    return _finish(fig, path, title)


# ---------------------------------------------------------------- experiments from notebook 08 on

def metric_panels(table, panels, path, title=None):
    """One panel per metric, one bar per run (the rows of `table`). panels = [(column, axis label, number format)]."""
    runs = list(table.index)
    if len(runs) <= 8:  # one row of panels
        nrows, ncols, width = 1, len(panels), 3.4
    else:  # many models: two panels per row, each wide enough for its bars
        nrows, ncols, width = -(-len(panels) // 2), min(2, len(panels)), 0.45 * len(runs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, 3.4 * nrows), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (column, label, fmt) in zip(axes.flat, panels):
        bars = ax.bar(runs, table[column], color=COLORS[: len(runs)], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, bars, fmt)
        _style(ax, label)
        ax.set_ylim(bottom=0)
    return _finish(fig, path, title)


def game_results(table, path, title=None):
    """Wins, draws and losses for each model (rows of `table`, with columns wins/draws/losses), stacked to 100%."""
    shares = table[["wins", "draws", "losses"]].div(table[["wins", "draws", "losses"]].sum(axis=1), axis=0) * 100
    shares.columns = ["win", "draw", "loss"]
    return stacked_shares(shares, [COLORS[2], "#c9c7c1", COLORS[6]], "share of games (%)", path, title=title)


def training_curve(log, column, ylabel, path, title=None):
    """Any one quantity logged during training (e.g. "kl") over training steps."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    rows = log.dropna(subset=[column])
    ax.plot(rows.step, rows[column], color=COLORS[0], linewidth=2)
    ax.set_xlabel("training step", color=MUTED)
    _style(ax, ylabel)
    return _finish(fig, path, title)


def training_curves(logs, column, ylabel, path, title=None):
    """One logged quantity (e.g. "kl") over training steps for several runs. logs = {run name: training log}."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    for color, (name, log) in zip(COLORS, logs.items()):
        rows = log.dropna(subset=[column])
        ax.plot(rows.step, rows[column], color=color, linewidth=2, label=name)
    ax.set_xlabel("training step", color=MUTED)
    ax.legend(frameon=False)
    _style(ax, ylabel)
    return _finish(fig, path, title)


def scaling_curve(table, panels, references, path, title=None):
    """Metrics against the number of training examples, one panel per metric.

    table: one row per model copy, indexed by training examples seen. panels = [(column, axis label)].
    references = {label: {column: value}}, drawn as dashed horizontal lines (e.g. sft_v1, a random move).
    """
    return scaling_curves({"this run": table}, panels, references, path, title)


def scaling_curves(tables, panels, references, path, title=None):
    """Like scaling_curve, with one curve per run on the same axes. tables = {curve label: table}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), facecolor=SURFACE, squeeze=False)
    for ax, (column, label) in zip(axes[0], panels):
        for color, (name, table) in zip(COLORS, tables.items()):
            ax.plot(table.index / 1000, table[column], color=color, linewidth=2, marker="o", markersize=7, label=name)
        for value_color, (name, values) in zip(COLORS[len(tables):], references.items()):
            if column in values:
                ax.axhline(values[column], color=value_color, linestyle="--", linewidth=1.5, label=name)
        ax.set_xlabel("training examples seen (thousands)", color=MUTED)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        _style(ax, label)
    return _finish(fig, path, title)


def paired_effects(effects, panels, path, title=None, note=None):
    """The effect of each change as a dot with its 95% interval, one row per change and one panel per measure.

    effects: one dict per change, with "label" and, for each panel key, an (average, low, high) tuple
    (or Nones if not measured). panels = [(key, axis label, number format)]. A filled dot means the
    interval excludes 0. Right is better in every panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels) + 2.2, 0.55 * len(effects) + 1.6),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    ys = np.arange(len(effects))[::-1]
    for ax, (key, label, fmt) in zip(axes[0], panels):
        _style(ax, "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=1)
        for y, effect in zip(ys, effects):
            mean, low, high = effect.get(key, (None, None, None))
            if mean is None:
                ax.annotate("not measured yet", (0, y), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=7, color=MUTED)
                continue
            clear = low > 0 or high < 0
            ax.plot([low, high], [y, y], color=COLORS[0], linewidth=2, solid_capstyle="round")
            ax.plot(mean, y, "o", markersize=7, color=COLORS[0] if clear else SURFACE, markeredgecolor=COLORS[0],
                    markeredgewidth=2)
            ax.annotate(fmt.format(mean), (high, y), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=7, color=TEXT)
        ax.set_xlabel(label + "\nbetter →", color=MUTED, fontsize=9)
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + (right - left) * 0.15)
    axes[0][0].set_yticks(ys, [effect["label"] for effect in effects], fontsize=8)
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=MUTED)
    return _finish(fig, path, title)


def two_by_two(table, rows, columns, metrics, path, title=None, references=None):
    """A 2 x 2 design as lines: one line per row of the design, its two columns on the x-axis.

    table: one row per model, with each metric's value and its interval as "<metric>", "<metric> low",
    "<metric> high" (in %). rows = {line label: [model in column 1, model in column 2]}; columns = the two
    x-axis labels; metrics = [(metric, axis label)]. references = {metric: (label, value)} for a dashed line.
    Models missing from the table are left out.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.9), facecolor=SURFACE, squeeze=False)
    for ax, (metric, ylabel) in zip(axes[0], metrics):
        _style(ax, ylabel)
        for i, (label, models) in enumerate(rows.items()):
            points = [(x + (i - (len(rows) - 1) / 2) * 0.06, table.loc[m]) for x, m in enumerate(models)
                      if m in table.index and pd.notna(table.loc[m, metric])]
            if not points:
                continue
            xs = [x for x, _ in points]
            values = [row[metric] for _, row in points]
            errors = [[row[metric] - row[f"{metric} low"] for _, row in points],
                      [row[f"{metric} high"] - row[metric] for _, row in points]]
            ax.errorbar(xs, values, yerr=errors, color=COLORS[i], marker="o", markersize=8, linewidth=2, capsize=3,
                        label=label)
            for x, v in zip(xs, values):
                ax.annotate(f"{v:.1f}", (x, v), xytext=(9 if i else -9, 0), textcoords="offset points",
                            ha="left" if i else "right", va="center", fontsize=8, color=TEXT,
                            bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
        if references and metric in references:
            ref_label, value = references[metric]
            ax.axhline(value, color=MUTED, linestyle="--", linewidth=1, label=ref_label)
        ax.set_xticks(range(len(columns)), columns)
        ax.set_xlim(-0.45, len(columns) - 0.55)
        ax.set_ylim(bottom=0)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    return _finish(fig, path, title)


def tradeoff_map(points, path, title=None, reference="sft_v1", offsets=None, groups=None):
    """Each model as a dot: centipawns saved (x) against the change in Stockfish match (y), both vs. one reference.

    points: one row per model with columns cp, cp low, cp high, sf, sf low, sf high and group (for the color
    and the legend). offsets = {model: (dx, dy)} moves a label away from its dot (in points).
    groups = the legend order of the groups (default: as they appear).
    """
    offsets = offsets or {}
    groups = groups or list(dict.fromkeys(points.group))
    color = {group: COLORS[i % len(COLORS)] for i, group in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(9, 6.8), facecolor=SURFACE)
    _style(ax, f"same move as Stockfish vs. {reference} (percentage points)\nfinds the best move more often ↑")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    for model, row in points.iterrows():
        c = color[row.group]
        ax.errorbar(row.cp, row.sf, xerr=[[row.cp - row["cp low"]], [row["cp high"] - row.cp]],
                    yerr=[[row.sf - row["sf low"]], [row["sf high"] - row.sf]], color=c, alpha=0.35, linewidth=1,
                    capsize=0, zorder=1)
        ax.plot(row.cp, row.sf, "o", color=c, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        dx, dy = offsets.get(model, (6, 4))
        ax.annotate(model, (row.cp, row.sf), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=TEXT,
                    ha="left" if dx >= 0 else "right", zorder=4)
    ax.plot(0, 0, "o", color=SURFACE, markersize=9, markeredgecolor=TEXT, markeredgewidth=1.5, zorder=3)
    ax.annotate(f"{reference} (reference)", (0, 0), xytext=(-8, -12), textcoords="offset points", fontsize=8,
                ha="right", color=TEXT)
    ax.set_xlabel(f"centipawns saved per legal move vs. {reference} (positions both answered legally)\n"
                  "loses fewer centipawns →", color=MUTED)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color[g], markersize=8) for g in groups]
    ax.legend(handles, groups, frameon=False, fontsize=8, loc="lower left")
    return _finish(fig, path, title)


def grouped_bars(table, xlabel, path, title=None, colors=None, fmt="{:.0f}"):
    """Horizontal bars: one group per row of `table` (e.g. a model), one bar per column (e.g. a setting)."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(table) * len(table.columns) / 2 + 1.6), facecolor=SURFACE)
    _style(ax, "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    height = 0.8 / len(table.columns)
    for j, column in enumerate(table.columns):
        ys = np.arange(len(table)) + (j - (len(table.columns) - 1) / 2) * height  # first column on top
        ax.barh(ys, table[column], height=height * 0.92, color=colors[j % len(colors)], label=column,
                edgecolor=SURFACE, linewidth=1)
        for y, value in zip(ys, table[column]):
            if pd.notna(value):
                ax.annotate(fmt.format(value), (value, y), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=7, color=TEXT)
    ax.set_yticks(range(len(table)), table.index)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _finish(fig, path, title)


# ---------------------------------------------------------------- animated before/after on puzzles

BOARD_LIGHT, BOARD_DARK = "#ecdab9", "#ae8a68"
PIECE_GLYPHS = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}  # filled glyphs, colored per side


def _draw_board(ax, board, move=None, arrow_color=None):
    """Draw a position from the side to move's point of view, with an optional arrow for a move."""
    import chess
    from matplotlib import patheffects

    flip = board.turn == chess.BLACK
    ax.clear()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    def xy(square):  # the centre of a square on the drawn board
        file, rank = chess.square_file(square), chess.square_rank(square)
        return (7 - file + 0.5, 7 - rank + 0.5) if flip else (file + 0.5, rank + 0.5)

    for square in chess.SQUARES:
        x, y = xy(square)
        light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=BOARD_LIGHT if light else BOARD_DARK, linewidth=0))
        piece = board.piece_at(square)
        if piece:
            white = piece.color == chess.WHITE
            ax.text(x, y - 0.04, PIECE_GLYPHS[piece.symbol().lower()], ha="center", va="center", fontsize=21,
                    color="#ffffff" if white else "#111111",
                    path_effects=[patheffects.withStroke(linewidth=1.6, foreground="#111111" if white else "#ffffff")])
    if move is not None:
        (x0, y0), (x1, y1) = xy(move.from_square), xy(move.to_square)
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7",
                                                            color=arrow_color, linewidth=4, alpha=0.85))


def puzzle_before_after(puzzles, before, after, path, labels=("before", "after"), seconds_per_puzzle=1.8, note=None):
    """An animated GIF of the same puzzles answered by two models, side by side, with running counts on top.

    puzzles: rows with fen, answer_uci and rating (one frame per row, in order). before / after: each model's
    graded answers (columns puzzle_id, reply, move, outcome, correct), indexed by puzzle_id. A green arrow is a
    correct answer, a grey arrow a legal but wrong one; an illegal or unparseable answer is written in red.
    note: a line printed at the bottom of every frame, e.g. how the puzzles were chosen. A GIF is often shared
    without its caption, so any selection of puzzles should be stated here.
    """
    import chess
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.12, wspace=0.12)
    headers = [fig.text(x, 0.955, "", ha="center", fontsize=12, fontweight="bold", color=TEXT) for x in (0.27, 0.73)]
    counts = [fig.text(x, 0.915, "", ha="center", fontsize=9, color=MUTED) for x in (0.27, 0.73)]
    puzzle_line = fig.text(0.5, 0.855, "", ha="center", fontsize=9, color=MUTED)
    answers = [fig.text(x, 0.065, "", ha="center", fontsize=10) for x in (0.27, 0.73)]
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8.5, color=MUTED, style="italic")
    rows = list(puzzles.itertuples())

    def frame(i):
        row = rows[i]
        board = chess.Board(row.fen)
        solution = board.san(chess.Move.from_uci(row.answer_uci))
        side = "White" if board.turn == chess.WHITE else "Black"
        puzzle_line.set_text(f"puzzle {i + 1} of {len(rows)}  ·  rating {row.rating}  ·  {side} to move  ·  "
                             f"solution: {solution}")
        for k, (graded, label, ax) in enumerate(zip((before, after), labels, axes)):
            answer = graded.loc[row.puzzle_id]
            move, color = None, None
            if answer.outcome == "legal":
                move = chess.Move.from_uci(answer.move)
                color = COLORS[2] if answer.correct else MUTED
                text = f"{board.san(move)}  " + ("✓ correct" if answer.correct else "✗ wrong")
                text_color = COLORS[2] if answer.correct else TEXT
            else:
                reply = str(answer.reply).strip().split("\n")[0][:18] or "(nothing)"
                text, text_color = f"{reply}  ✗ {answer.outcome}", COLORS[6]
            so_far = graded.loc[[r.puzzle_id for r in rows[: i + 1]]]  # counted afresh: frames can be drawn twice
            _draw_board(ax, board, move, color)
            headers[k].set_text(label)
            counts[k].set_text(f"solved {int(so_far.correct.sum())} of {i + 1}  ·  "
                               f"illegal or unreadable: {int((so_far.outcome != 'legal').sum())}")
            answers[k].set_text(text)
            answers[k].set_color(text_color)
        return []

    animation = FuncAnimation(fig, frame, frames=len(rows), blit=False, repeat=False)
    animation.save(path, writer=PillowWriter(fps=1 / seconds_per_puzzle), dpi=100)
    plt.close(fig)
    return path
