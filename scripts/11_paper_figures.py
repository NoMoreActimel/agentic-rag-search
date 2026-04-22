#!/usr/bin/env python3
"""Paper-ready figures from the merged 87-QA run.

Two figures, each a 1x2 panel (feedback off | on):
  figure_1_judge_score.(png|pdf) — LLM-as-judge by retriever × iterations.
  figure_2_hallucination.(png|pdf) — hallucination rate, zoomed in.

Uses the quality_reweight=off slice (N=87/cell) for clarity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETRIEVERS = ["grep", "bm25", "embedding"]

# Okabe–Ito palette: colorblind-safe, print-friendly.
RETRIEVER_COLOR = {
    "grep": "#D55E00",      # vermillion
    "bm25": "#0072B2",      # blue
    "embedding": "#009E73",  # bluish green
}
RETRIEVER_MARKER = {"grep": "o", "bm25": "s", "embedding": "D"}
RETRIEVER_LABEL = {"grep": "grep", "bm25": "BM25", "embedding": "embedding"}


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
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean + 95% CI (t-approx via SEM*1.96) for every cell."""
    g = (
        df.groupby(["retriever", "max_steps", "process_feedback"])[metric]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    g["ci"] = 1.96 * g["sem"]
    return g


def plot_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    metric: str,
    feedback: bool,
    ylabel: str,
    ylim: tuple[float, float],
    title: str,
    jitter: float = 0.06,
) -> None:
    sub = agg[agg["process_feedback"] == feedback]
    xs_all = sorted(sub["max_steps"].unique())
    x_to_idx = {x: i for i, x in enumerate(xs_all)}
    for i, retr in enumerate(RETRIEVERS):
        cell = sub[sub["retriever"] == retr].sort_values("max_steps")
        offset = (i - 1) * jitter  # center the three lines
        xs = np.array([x_to_idx[s] for s in cell["max_steps"]]) + offset
        ys = cell["mean"].to_numpy()
        errs = cell["ci"].to_numpy()
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            color=RETRIEVER_COLOR[retr],
            marker=RETRIEVER_MARKER[retr],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.8,
            capsize=3,
            elinewidth=1.0,
            label=RETRIEVER_LABEL[retr],
        )
    ax.set_xticks(list(x_to_idx.values()))
    ax.set_xticklabels([str(x) for x in xs_all])
    ax.set_xlim(-0.35, len(xs_all) - 1 + 0.35)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def figure_judge(df: pd.DataFrame, out_stub: Path) -> None:
    agg = aggregate(df, "judge_score")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    y_lim = (3.3, 4.8)
    plot_panel(
        axes[0], agg, "judge_score", feedback=False,
        ylabel="LLM-as-judge score (1–5)", ylim=y_lim,
        title="No process feedback",
    )
    plot_panel(
        axes[1], agg, "judge_score", feedback=True,
        ylabel="", ylim=y_lim,
        title="With process feedback",
    )
    axes[0].legend(loc="lower right", title=None, ncol=1)
    fig.suptitle(
        "Judge score vs. iterations by retriever (N=87 QAs per point, 95% CI)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stub.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def figure_hallucination(df: pd.DataFrame, out_stub: Path) -> None:
    agg = aggregate(df, "hallucination_rate")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    # Zoom to highlight the drop; keep enough headroom for grep@2 whisker.
    y_lim = (-0.005, 0.22)
    plot_panel(
        axes[0], agg, "hallucination_rate", feedback=False,
        ylabel="Hallucination rate (lower is better)", ylim=y_lim,
        title="No process feedback",
    )
    plot_panel(
        axes[1], agg, "hallucination_rate", feedback=True,
        ylabel="", ylim=y_lim,
        title="With process feedback",
    )
    for ax in axes:
        ax.axhline(0, color="0.3", linewidth=0.6, linestyle=":")
    # Legend on the right panel where the top is empty, so it doesn't cover data.
    axes[1].legend(loc="upper right", title=None, ncol=1)
    fig.suptitle(
        "Hallucination rate vs. iterations by retriever (N=87 QAs per point, 95% CI)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_stub.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.run_dir / "per_example_metrics.csv"
    df = pd.read_csv(csv_path)
    df = df[df["quality_reweight"] == False].copy()  # noqa: E712

    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    figure_judge(df, out_dir / "figure_1_judge_score")
    figure_hallucination(df, out_dir / "figure_2_hallucination")


if __name__ == "__main__":
    main()
