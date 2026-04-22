#!/usr/bin/env python3
"""Direct quality_reweight on/off comparison for the merged 87-QA run.

Produces three artifacts (the existing qoff/qon/qboth plots stay untouched):
  1. Two side-by-side grids (one per process_feedback value): rows=metrics,
     cols=retrievers, hue=quality_reweight, x=max_steps.
  2. Paired delta grid: for each (qa_id, retriever, max_steps, process_feedback)
     we compute delta = metric(q=on) - metric(q=off) and plot the mean with
     a 95% CI. Controls for question difficulty — the correct comparison.
  3. CSV table with per-cell paired-delta mean, SEM, N, and a two-sided
     paired-t p-value against H0: delta==0.

Output goes to <run-dir>/analysis/plots/quality_onoff/ (new subdir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RETRIEVERS = ["grep", "bm25", "embedding"]
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
        default=Path("data/results/submit87_fast_merged"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


def _label_feedback(v: bool) -> str:
    return "on" if v else "off"


def plot_grid_for_feedback(df: pd.DataFrame, feedback: bool, out_path: Path) -> None:
    sub = df[df["process_feedback"] == feedback].copy()
    sub["Quality reweight"] = sub["quality_reweight"].map({True: "on", False: "off"})

    nrows = len(METRICS)
    ncols = len(RETRIEVERS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.4 * nrows), sharex=True)
    for row, (metric, ylabel, ylim, _) in enumerate(METRICS):
        for col, retr in enumerate(RETRIEVERS):
            ax = axes[row, col]
            cell = sub[sub["retriever"] == retr]
            sns.pointplot(
                data=cell,
                x="max_steps",
                y=metric,
                hue="Quality reweight",
                hue_order=["off", "on"],
                errorbar=("ci", 95),
                dodge=0.12,
                markers=["o", "s"],
                linestyles=["-", "--"],
                ax=ax,
            )
            ax.set_title(f"{retr} — {metric}")
            ax.set_xlabel("Max retrieval steps" if row == nrows - 1 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_ylim(*ylim)
            if not (row == 0 and col == ncols - 1):
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.suptitle(
        f"Quality reweight on vs off — process_feedback={_label_feedback(feedback)} (N=87/cell)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame of per-QA deltas (qon - qoff) for every
    (qa_id, retriever, max_steps, process_feedback) and metric.
    """
    key = ["qa_id", "retriever", "max_steps", "process_feedback"]
    metric_cols = [m for m, _, _, _ in METRICS]
    df = df.copy()
    for m in metric_cols:
        df[m] = df[m].astype(float)
    wide = (
        df.pivot_table(
            index=key,
            columns="quality_reweight",
            values=metric_cols,
            aggfunc="first",
        )
        .dropna()
    )
    records = []
    for m in metric_cols:
        delta = wide[(m, True)] - wide[(m, False)]
        tmp = delta.reset_index().rename(columns={0: "delta"})
        tmp.columns = list(tmp.columns[:-1]) + ["delta"]
        tmp["metric"] = m
        records.append(tmp)
    return pd.concat(records, ignore_index=True)


def plot_delta_grid(deltas: pd.DataFrame, out_path: Path) -> None:
    dd = deltas.copy()
    dd["Process feedback"] = dd["process_feedback"].map({True: "on", False: "off"})

    nrows = len(METRICS)
    ncols = len(RETRIEVERS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.4 * nrows), sharex=True)
    for row, (metric, ylabel, _ylim, lower_better) in enumerate(METRICS):
        for col, retr in enumerate(RETRIEVERS):
            ax = axes[row, col]
            cell = dd[(dd["metric"] == metric) & (dd["retriever"] == retr)]
            sns.pointplot(
                data=cell,
                x="max_steps",
                y="delta",
                hue="Process feedback",
                hue_order=["off", "on"],
                errorbar=("ci", 95),
                dodge=0.12,
                markers=["o", "s"],
                linestyles=["-", "--"],
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            direction = " (negative is better)" if lower_better else " (positive is better)"
            ax.set_title(f"{retr} — Δ {metric}{direction}")
            ax.set_xlabel("Max retrieval steps" if row == nrows - 1 else "")
            ax.set_ylabel(f"Δ {ylabel} (on − off)" if col == 0 else "")
            if not (row == 0 and col == ncols - 1):
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
    fig.suptitle(
        "Paired Δ (quality=on minus quality=off), 95% CI — N=87 QAs per point",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _paired_t_pvalue(d: np.ndarray) -> tuple[float, float]:
    """Two-sided paired-t test against H0: mean==0 using a normal approximation
    for the p-value (N=87 is well into large-sample territory)."""
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return (0.0, 1.0) if mean == 0.0 else (float("inf"), 0.0)
    t = mean / (sd / math.sqrt(n))
    # normal approximation to t-distribution (N=87 is fine)
    pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return t, pval


def delta_stats_table(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, retriever, max_steps, feedback), grp in deltas.groupby(
        ["metric", "retriever", "max_steps", "process_feedback"]
    ):
        d = grp["delta"].to_numpy()
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        sem = sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        tstat, pval = _paired_t_pvalue(d)
        rows.append(
            {
                "metric": metric,
                "retriever": retriever,
                "max_steps": int(max_steps),
                "process_feedback": bool(feedback),
                "n": int(len(d)),
                "mean_delta": round(mean, 4),
                "sem": round(sem, 4),
                "t": round(tstat, 3),
                "p_value": round(pval, 4),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "retriever", "process_feedback", "max_steps"]
    )


def main() -> None:
    args = parse_args()
    csv_path = args.run_dir / "per_example_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "quality_onoff")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Per-feedback grids
    plot_grid_for_feedback(df, feedback=False, out_path=out_dir / "quality_grid_feedbackoff.png")
    plot_grid_for_feedback(df, feedback=True, out_path=out_dir / "quality_grid_feedbackon.png")

    # Paired delta plot + stats
    deltas = paired_delta(df)
    plot_delta_grid(deltas, out_path=out_dir / "quality_delta_grid.png")
    tbl = delta_stats_table(deltas)
    tbl_path = out_dir / "quality_delta_stats.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"Saved {tbl_path}")
    print("\nPaired-delta stats (on − off):")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
