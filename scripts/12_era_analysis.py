#!/usr/bin/env python3
"""Era-stratified analysis for the merged 87-QA run.

The 87 QA pairs are split into cohorts relative to Gemini 2.5 Flash's
2025-01-31 cutoff:

  - old   : all reference episodes pre-cutoff (model saw them in training)
  - new   : all reference episodes post-cutoff (unseen by model)
  - mixed : references span both eras

If agentic RAG relied heavily on the model's parametric memory, `old` would
outperform `new`. This script quantifies the actual gap on this benchmark.

Outputs (under <run-dir>/analysis/plots/era/):
  - era_means.csv         : per-era means + bootstrap 95% CIs
  - era_by_retriever.csv  : retriever x era means
  - era_by_qa_type.csv    : qa_type x era means
  - era_stat_tests.csv    : Mann-Whitney old vs new for every metric
  - era_overview.(png|pdf): bar chart of headline metrics by era with CIs
  - era_by_retriever.(png|pdf): retriever x era heatmap/bars
  - era_by_qa_type.(png|pdf)  : qa_type x era stratified bars
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import mannwhitneyu

ERAS = ["old", "mixed", "new"]
RETRIEVERS = ["grep", "bm25", "embedding"]

# Okabe-Ito-ish, matching other figures in the repo.
ERA_COLOR = {
    "old": "#999999",    # neutral grey — model has seen it
    "mixed": "#0072B2",  # blue — partial
    "new": "#D55E00",    # vermillion — unseen
}

METRICS = [
    ("success", "Success rate", (0.0, 1.0), False),
    ("judge_score", "LLM judge (1-5)", (1.0, 5.0), False),
    ("reference_episode_recall", "Reference episode recall", (0.0, 1.0), False),
    ("retrieval_precision", "Retrieval precision", (0.0, 1.0), False),
    ("hallucination_rate", "Hallucination rate", (0.0, 0.2), True),
    ("lexical_f1", "Lexical F1", (0.0, 0.6), False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, default=Path("data/results/submit87_fast_merged")
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
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
            "axes.axisbelow": True,
            "grid.color": "0.88",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def bootstrap_ci(vals: np.ndarray, rng, n_boot: int = 5000) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_qa_means(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Collapse 36 conditions per question down to one number per question."""
    return (
        df.groupby(["qa_id", "era", "qa_type"])[metrics].mean().reset_index()
    )


def era_summary(per_qa: pd.DataFrame, metrics: list[str], rng) -> pd.DataFrame:
    rows = []
    for era in ERAS:
        sub = per_qa[per_qa["era"] == era]
        row: dict = {"era": era, "n_qa": len(sub)}
        for m in metrics:
            vals = sub[m].values
            mean = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals, rng)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def plot_overview(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for ax, (metric, label, ylim, lower_better) in zip(axes.flat, METRICS):
        means = [summary.loc[summary.era == e, f"{metric}_mean"].iloc[0] for e in ERAS]
        los = [summary.loc[summary.era == e, f"{metric}_ci_lo"].iloc[0] for e in ERAS]
        his = [summary.loc[summary.era == e, f"{metric}_ci_hi"].iloc[0] for e in ERAS]
        yerr = np.array(
            [
                [m - lo for m, lo in zip(means, los)],
                [hi - m for m, hi in zip(means, his)],
            ]
        )
        bars = ax.bar(
            ERAS,
            means,
            yerr=yerr,
            color=[ERA_COLOR[e] for e in ERAS],
            capsize=4,
            edgecolor="black",
            linewidth=0.7,
        )
        title = label + (" (lower=better)" if lower_better else "")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.01,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.suptitle(
        "Era-stratified metrics (87 QA, 36 conditions, per-QA means, 95% bootstrap CI)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_by_retriever(df: pd.DataFrame, out_path: Path) -> None:
    metrics_small = [
        ("success", "Success rate", (0.0, 1.0)),
        ("judge_score", "LLM judge (1-5)", (1.0, 5.0)),
        ("reference_episode_recall", "Reference recall", (0.0, 1.0)),
        ("hallucination_rate", "Hallucination rate", (0.0, 0.15)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    x = np.arange(len(RETRIEVERS))
    width = 0.26
    for ax, (metric, label, ylim) in zip(axes, metrics_small):
        for i, era in enumerate(ERAS):
            vals = [
                df[(df.retriever == r) & (df.era == era)][metric].mean()
                for r in RETRIEVERS
            ]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=era,
                color=ERA_COLOR[era],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(RETRIEVERS)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(title="era", loc="lower right")
    fig.suptitle("Retriever x era — means over all grid cells", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_by_qa_type(df: pd.DataFrame, out_path: Path) -> None:
    qa_types = sorted(df["qa_type"].unique())
    metrics_small = [
        ("success", "Success rate", (0.0, 1.0)),
        ("judge_score", "LLM judge (1-5)", (1.0, 5.0)),
        ("reference_episode_recall", "Reference recall", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(qa_types))
    width = 0.26
    for ax, (metric, label, ylim) in zip(axes, metrics_small):
        for i, era in enumerate(ERAS):
            vals = []
            for t in qa_types:
                sub = df[(df.qa_type == t) & (df.era == era)]
                vals.append(sub[metric].mean() if len(sub) else np.nan)
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=era,
                color=ERA_COLOR[era],
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(qa_types, rotation=20)
        ax.set_ylim(*ylim)
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(title="era", loc="lower right")
    fig.suptitle(
        "QA-type x era — does the era gap track question type?",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def stat_tests(per_qa: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        old_vals = per_qa[per_qa.era == "old"][m].values
        new_vals = per_qa[per_qa.era == "new"][m].values
        mix_vals = per_qa[per_qa.era == "mixed"][m].values
        u_on, p_on = mannwhitneyu(old_vals, new_vals, alternative="two-sided")
        u_om, p_om = mannwhitneyu(old_vals, mix_vals, alternative="two-sided")
        rows.append(
            {
                "metric": m,
                "old_mean": float(np.mean(old_vals)),
                "mixed_mean": float(np.mean(mix_vals)),
                "new_mean": float(np.mean(new_vals)),
                "delta_new_minus_old": float(np.mean(new_vals) - np.mean(old_vals)),
                "mwu_old_vs_new_U": float(u_on),
                "mwu_old_vs_new_p": float(p_on),
                "mwu_old_vs_mixed_p": float(p_om),
            }
        )
    return pd.DataFrame(rows)


def stratified_macro_means(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Macro-average across qa_type to decouple era effect from type mix."""
    per_type = (
        df.groupby(["era", "qa_type"])[metrics].mean().reset_index()
    )
    return per_type.groupby("era")[metrics].mean().reset_index()


def main() -> None:
    args = parse_args()
    setup_style()
    rng = default_rng(args.seed)

    csv_path = args.run_dir / "per_example_metrics.csv"
    df = pd.read_csv(csv_path)
    metric_names = [m for m, *_ in METRICS]

    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "era")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_qa = per_qa_means(df, metric_names)
    summary = era_summary(per_qa, metric_names, rng)
    summary.to_csv(out_dir / "era_means.csv", index=False)

    by_ret = (
        df.groupby(["retriever", "era"])[metric_names].mean().round(4).reset_index()
    )
    by_ret.to_csv(out_dir / "era_by_retriever.csv", index=False)

    by_type = (
        df.groupby(["qa_type", "era"])[metric_names].mean().round(4).reset_index()
    )
    by_type.to_csv(out_dir / "era_by_qa_type.csv", index=False)

    strat = stratified_macro_means(df, metric_names)
    strat.to_csv(out_dir / "era_means_macro_by_qa_type.csv", index=False)

    tests = stat_tests(per_qa, metric_names)
    tests.to_csv(out_dir / "era_stat_tests.csv", index=False)

    plot_overview(summary, out_dir / "era_overview")
    plot_by_retriever(df, out_dir / "era_by_retriever")
    plot_by_qa_type(df, out_dir / "era_by_qa_type")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=== Per-era summary (per-QA means, 95% bootstrap CI) ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Old vs New statistical tests (per-QA, Mann-Whitney U) ===")
    print(tests.round(4).to_string(index=False))
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
