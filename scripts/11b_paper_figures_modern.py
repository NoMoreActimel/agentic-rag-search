#!/usr/bin/env python3
"""Paper figures, v2 — conference-paper style.

Writes to <run-dir>/analysis/plots/paper_modern/:
  figure_1_judge_score.(png|pdf) — LLM-as-judge by retriever × iterations.
  figure_2_hallucination.(png|pdf) — hallucination rate, zoomed.
  figure_3_success.(png|pdf)      — binary success rate (cleaner alt. to judge).

Each figure is a 1x2 panel (feedback off | on) designed to live inside a
single-column or two-column LaTeX figure environment with its caption
supplied by the document, so the figure itself avoids blog-style headlines
and footer captions. Uses the quality_reweight=off slice (N=87/cell).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETRIEVERS = ["grep", "bm25", "embedding"]

# Shared paper palette — editorial print tones, distinguishable in grayscale.
# grep = warm terracotta; bm25 = prussian navy; embedding = eucalyptus sage.
RETRIEVER_COLOR = {
    "grep": "#BC5A4E",
    "bm25": "#3A6A87",
    "embedding": "#5D8A6A",
}
RETRIEVER_MARKER = {"grep": "o", "bm25": "s", "embedding": "D"}
RETRIEVER_LABEL = {"grep": "grep", "bm25": "BM25", "embedding": "embedding"}

INK = "#1F2328"
MUTED = "#4B5563"
GRID = "#D8DCE1"


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
            # Serif to match Times-based body text used by ACM/IEEE/NeurIPS.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": "#B8BEC6",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.labelsize": 10,
            "axes.labelweight": "regular",
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
            "grid.linestyle": (0, (1, 2)),  # fine dotted
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.6,
            "legend.labelspacing": 0.35,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.pad": 4,
            "ytick.major.pad": 3,
            "pdf.fonttype": 42,  # keep text editable in vector PDFs
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


def _panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    feedback: bool,
    ylabel: str,
    ylim: tuple[float, float],
    title: str,
    show_legend: bool = False,
    legend_loc: str = "best",
) -> None:
    sub = agg[agg["process_feedback"] == feedback]
    xs_all = sorted(sub["max_steps"].unique())
    x_to_idx = {x: i for i, x in enumerate(xs_all)}
    for retr in RETRIEVERS:
        cell = sub[sub["retriever"] == retr].sort_values("max_steps")
        xs = np.array([x_to_idx[s] for s in cell["max_steps"]], dtype=float)
        ys = cell["mean"].to_numpy()
        lo = ys - cell["ci"].to_numpy()
        hi = ys + cell["ci"].to_numpy()
        c = RETRIEVER_COLOR[retr]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(
            xs, ys,
            color=c, linewidth=1.6, solid_capstyle="round",
            marker=RETRIEVER_MARKER[retr], markersize=5.5,
            markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.0,
            label=RETRIEVER_LABEL[retr], zorder=3,
        )
    ax.set_xticks(list(x_to_idx.values()))
    ax.set_xticklabels([str(x) for x in xs_all])
    ax.set_xlim(-0.35, len(xs_all) - 1 + 0.35)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Max retrieval steps")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center")
    ax.tick_params(length=0)
    if show_legend:
        leg = ax.legend(loc=legend_loc)
        for t in leg.get_texts():
            t.set_color(INK)


def _two_panel(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    out_stub: Path,
    legend_panel: int = 0,
    legend_loc: str = "lower right",
    add_zero_line: bool = False,
) -> None:
    agg = aggregate(df, metric)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    _panel(
        axes[0], agg, feedback=False,
        ylabel=ylabel, ylim=ylim,
        title="(a) no process feedback",
        show_legend=(legend_panel == 0),
        legend_loc=legend_loc,
    )
    _panel(
        axes[1], agg, feedback=True,
        ylabel="", ylim=ylim,
        title="(b) with process feedback",
        show_legend=(legend_panel == 1),
        legend_loc=legend_loc,
    )
    if add_zero_line:
        for ax in axes:
            ax.axhline(0, color=MUTED, linewidth=0.5, linestyle=(0, (1, 3)), alpha=0.5, zorder=0)
    fig.tight_layout(w_pad=1.0)
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

    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "paper_modern")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    _two_panel(
        df, "judge_score",
        ylabel="LLM-as-judge score (1–5)",
        ylim=(3.3, 4.85),
        out_stub=out_dir / "figure_1_judge_score",
        legend_panel=0, legend_loc="lower right",
    )
    _two_panel(
        df, "hallucination_rate",
        ylabel="Hallucination rate",
        ylim=(-0.008, 0.22),
        out_stub=out_dir / "figure_2_hallucination",
        legend_panel=1, legend_loc="upper right",
        add_zero_line=True,
    )
    _two_panel(
        df, "success",
        ylabel="Success rate",
        ylim=(0.48, 0.94),
        out_stub=out_dir / "figure_3_success",
        legend_panel=0, legend_loc="lower right",
    )


if __name__ == "__main__":
    main()
