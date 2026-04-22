#!/usr/bin/env python3
"""Plot iterations (max_steps) vs four metrics, split by process_feedback,
for each retriever. Produces one 4x3 grid figure AND 12 individual per-cell figures.

Metrics:
  - judge_score           (LLM-as-judge 1-5, higher is better)
  - success               (binary correctness, higher is better)
  - reference_episode_recall (objective retrieval recall, higher is better)
  - hallucination_rate    (fraction unsupported, LOWER is better)

Each line shows the mean over QAs with a 95% CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RETRIEVERS = ["grep", "bm25", "embedding"]
# (column, y-axis label, y-limits, lower_is_better)
METRICS = [
    ("judge_score", "LLM-as-judge (1-5)", (1.0, 5.0), False),
    ("success", "Success rate", (0.0, 1.0), False),
    ("reference_episode_recall", "Reference episode recall", (0.0, 1.0), False),
    ("hallucination_rate", "Hallucination rate (lower is better)", (0.0, 1.0), True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/results/20260420_221641_full_main36_vertex_v2"),
    )
    parser.add_argument(
        "--quality",
        choices=["off", "on", "both"],
        default="off",
        help="Which quality_reweight slice to use (default: off -> N=20/cell).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <run-dir>/analysis/plots/iter_vs_judge_em).",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


def filter_quality(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "off":
        return df[df["quality_reweight"] == False].copy()  # noqa: E712
    if mode == "on":
        return df[df["quality_reweight"] == True].copy()  # noqa: E712
    return df.copy()


def plot_single(
    ax,
    data: pd.DataFrame,
    y: str,
    ylim: tuple[float, float],
    title: str,
) -> None:
    # Cast bool -> str so the legend is readable.
    data = data.copy()
    data["Process feedback"] = data["process_feedback"].map({True: "on", False: "off"})
    sns.pointplot(
        data=data,
        x="max_steps",
        y=y,
        hue="Process feedback",
        hue_order=["off", "on"],
        errorbar=("ci", 95),
        dodge=0.12,
        markers=["o", "s"],
        linestyles=["-", "--"],
        ax=ax,
    )
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.set_ylim(*ylim)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    csv_path = run_dir / "per_example_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df_all = pd.read_csv(csv_path)
    df = filter_quality(df_all, args.quality)

    out_dir = args.out_dir or (run_dir / "analysis" / "plots" / "iter_vs_judge_em")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Per-cell N (20 QAs × quality_reweight slice × 1 judge × 1 retriever × 1 max_steps)
    sample = df.groupby(["retriever", "max_steps", "process_feedback"]).size()
    print(f"Rows used: {len(df)} | per-cell N: min={sample.min()} max={sample.max()}")
    print(f"Quality slice: {args.quality}")

    # --- Combined 4x3 grid ----------------------------------------------------
    nrows = len(METRICS)
    ncols = len(RETRIEVERS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.4 * nrows), sharex=True)
    for row, (metric, ylabel, ylim, lower_better) in enumerate(METRICS):
        for col, retr in enumerate(RETRIEVERS):
            ax = axes[row, col]
            sub = df[df["retriever"] == retr]
            plot_single(ax, sub, y=metric, ylim=ylim, title=f"{retr} — {metric}")
            ax.set_ylabel(ylabel if col == 0 else "")
            if row < nrows - 1:
                ax.set_xlabel("")
            # Keep the legend only on the top-right panel.
            if not (row == 0 and col == ncols - 1):
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.suptitle(
        f"Iterations vs metrics by retriever (quality_reweight={args.quality})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = out_dir / f"iter_vs_metrics_grid_q{args.quality}.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grid_path}")

    # --- Individual plots (one per retriever × metric) ------------------------
    for metric, ylabel, ylim, _ in METRICS:
        for retr in RETRIEVERS:
            sub = df[df["retriever"] == retr]
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            plot_single(ax, sub, y=metric, ylim=ylim, title=f"{retr} — {metric}")
            ax.set_ylabel(ylabel)
            fname = f"iter_vs_{metric}__{retr}__q{args.quality}.png"
            fig.savefig(out_dir / fname, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_dir / fname}")

    # --- Numeric table for verification --------------------------------------
    agg_spec = {
        f"{m}_{stat}": (m, stat)
        for m, _, _, _ in METRICS
        for stat in ("mean", "sem")
    }
    agg_spec["n"] = ("judge_score", "size")
    agg = (
        df.groupby(["retriever", "max_steps", "process_feedback"])
        .agg(**agg_spec)
        .round(3)
        .reset_index()
    )
    table_path = out_dir / f"iter_vs_metrics_table_q{args.quality}.csv"
    agg.to_csv(table_path, index=False)
    print(f"Saved {table_path}")
    print("\nNumeric table:\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
