#!/usr/bin/env python3
"""Paper figures v3 — grouped bar charts with paired tints for feedback.

Same three metrics as scripts/11b_paper_figures_modern.py. Each figure is
a single axes (not a 1x2 panel): within every max_steps group, six bars
appear in this order:

    grep (no fb) | grep (+fb) | BM25 (no fb) | BM25 (+fb) | emb (no fb) | emb (+fb)

Each retriever has a paired color — a light tint for the "no feedback"
baseline and the full color for the "+ feedback" treatment. The shared
palette is the same one used by the line figures, so the paper reads as
one coherent visual system.

Writes to <run-dir>/analysis/plots/paper_bars/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETRIEVERS = ["grep", "bm25", "embedding"]

# Shared paper palette. Base = "+ feedback" (treatment), light = baseline.
RETRIEVER_COLOR = {
    "grep": "#BC5A4E",
    "bm25": "#3A6A87",
    "embedding": "#5D8A6A",
}
RETRIEVER_COLOR_LIGHT = {
    "grep": "#E6BFB8",
    "bm25": "#B7C7D3",
    "embedding": "#C5D3C5",
}
RETRIEVER_LABEL = {"grep": "grep", "bm25": "BM25", "embedding": "embedding"}

INK = "#1F2328"
MUTED = "#4B5563"
GRID = "#DDE0E4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/results/submit87_fast_merged"),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": "#B8BEC6",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "regular",
            "axes.titlecolor": INK,
            "axes.titlepad": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": (0, (1, 2)),
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.2,
            "legend.handleheight": 1.0,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.labelspacing": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.pad": 4,
            "ytick.major.pad": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = (
        df.groupby(["retriever", "max_steps", "process_feedback"])[metric]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    g["ci"] = 1.96 * g["sem"]
    return g


# Bar layout within one max_steps group (total width ≈ 0.94):
BAR_W = 0.12
WITHIN_PAIR = 0.13    # center-to-center, within a retriever's pair
BETWEEN_PAIR = 0.30   # center-to-center, between pair centers

# Pair centers relative to group center. Three pairs, symmetric.
PAIR_CENTERS = np.array([-BETWEEN_PAIR, 0.0, BETWEEN_PAIR])


def _bar_offsets() -> dict[tuple[str, bool], float]:
    """Return the x-offset (from group center) for each (retriever, feedback) bar."""
    offsets: dict[tuple[str, bool], float] = {}
    for retr, pair_center in zip(RETRIEVERS, PAIR_CENTERS):
        offsets[(retr, False)] = pair_center - WITHIN_PAIR / 2
        offsets[(retr, True)] = pair_center + WITHIN_PAIR / 2
    return offsets


def _draw(
    ax: plt.Axes,
    agg: pd.DataFrame,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    xs_all = sorted(agg["max_steps"].unique())
    base_x = np.arange(len(xs_all))
    offsets = _bar_offsets()

    for retr in RETRIEVERS:
        for feedback in (False, True):
            cell = (
                agg[(agg["retriever"] == retr) & (agg["process_feedback"] == feedback)]
                .set_index("max_steps")
                .reindex(xs_all)
            )
            means = cell["mean"].to_numpy()
            cis = cell["ci"].to_numpy()
            xs = base_x + offsets[(retr, feedback)]
            c = RETRIEVER_COLOR[retr] if feedback else RETRIEVER_COLOR_LIGHT[retr]
            # Subtle stroke in the base hue around light bars so they don't
            # vanish against a white background in print.
            edgecolor = "white" if feedback else RETRIEVER_COLOR[retr]
            linewidth = 0.8 if feedback else 0.6
            ax.bar(
                xs, means, width=BAR_W,
                color=c, edgecolor=edgecolor, linewidth=linewidth,
                zorder=2,
            )
            ax.errorbar(
                xs, means, yerr=cis,
                fmt="none",
                ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9,
                zorder=3,
            )

    ax.set_xticks(base_x)
    ax.set_xticklabels([str(x) for x in xs_all])
    ax.set_xlim(-0.55, len(xs_all) - 1 + 0.55)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def _legend(ax: plt.Axes) -> None:
    """Legend above the axes: two rows (no fb / + fb) × three cols (retrievers).

    matplotlib fills legend cells column-major, so items are interleaved
    per retriever to produce a visually row-major grid.
    """
    handles = []
    labels = []
    for retr in RETRIEVERS:
        handles.append(mpatches.Patch(
            facecolor=RETRIEVER_COLOR_LIGHT[retr],
            edgecolor=RETRIEVER_COLOR[retr], linewidth=0.6,
        ))
        labels.append(f"{RETRIEVER_LABEL[retr]} (no feedback)")
        handles.append(mpatches.Patch(
            facecolor=RETRIEVER_COLOR[retr],
            edgecolor="white", linewidth=0.8,
        ))
        labels.append(f"{RETRIEVER_LABEL[retr]} (+ feedback)")
    leg = ax.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False,
    )
    for t in leg.get_texts():
        t.set_color(INK)


def _render(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    out_stub: Path,
) -> None:
    agg = aggregate(df, metric)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.4))
    _draw(ax, agg, ylabel=ylabel, ylim=ylim)
    _legend(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stub.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.run_dir / "per_example_metrics.csv"
    df = pd.read_csv(csv_path)
    df = df[df["quality_reweight"] == False].copy()  # noqa: E712
    df["success"] = df["success"].astype(float)

    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "paper_bars")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    _render(
        df, "judge_score",
        ylabel="LLM-as-judge score (1–5)",
        ylim=(3.0, 4.9),
        out_stub=out_dir / "figure_1_judge_score",
    )
    _render(
        df, "hallucination_rate",
        ylabel="Hallucination rate",
        ylim=(0.0, 0.22),
        out_stub=out_dir / "figure_2_hallucination",
    )
    _render(
        df, "success",
        ylabel="Success rate",
        ylim=(0.45, 0.95),
        out_stub=out_dir / "figure_3_success",
    )


if __name__ == "__main__":
    main()
