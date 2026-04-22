#!/usr/bin/env python3
"""Era-matched paper figure — old vs old (pruned) vs new, with deltas.

The `old` cohort in the 87-QA eval covers **26** distinct reference episodes;
`new` covers **20**. This script greedily prunes `old` down to a 27-QA subset
whose union of references is 20 distinct episodes — matching `new` — while
keeping qa_type balance. It then recomputes the per-era means on the
existing logs (no rerun) and plots the three-bar comparison (old / old
pruned / new) with the new−old and new−old_pruned deltas reported beneath.

Style matches scripts/11c_paper_figures_bars.py: serif typography, muted
palette, grouped bars with 95% bootstrap CIs.

Outputs under `<run-dir>/analysis/plots/era_matched/`:
  - figure_era_matched.(png|pdf)  — six-metric panel
  - era_matched_means.csv         — per-cohort means + CIs
  - era_matched_deltas.csv        — Δ new−old and Δ new−old_pruned
  - notsoold_qa_ids.json          — the 27 qa_ids in the pruned old subset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import mannwhitneyu

METRICS = [
    ("success", "Success rate", (0.0, 1.0), False),
    ("judge_score", "LLM-as-judge (1–5)", (1.0, 5.0), False),
    ("reference_episode_recall", "Reference episode recall", (0.0, 1.0), False),
    ("retrieval_precision", "Retrieval precision", (0.0, 1.0), False),
    ("hallucination_rate", "Hallucination rate", (0.0, 0.15), True),
    ("lexical_f1", "Lexical F1", (0.0, 0.6), False),
]

# Cohort palette — slate for old, blue for mixed, warm vermillion for new.
COHORT_COLOR = {
    "old":   "#6B7280",  # slate grey — all pre-cutoff
    "mixed": "#3A6A87",  # blue — refs span both eras
    "new":   "#BC5A4E",  # warm vermillion — all post-cutoff
}
COHORT_LABEL = {
    "old":   "old (n=32)",
    "mixed": "mixed (n=32)",
    "new":   "new (n=23)",
}
COHORT_ORDER = ["old", "mixed", "new"]

INK = "#1F2328"
MUTED = "#4B5563"
GRID = "#DDE0E4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, default=Path("data/results/submit87_fast_merged")
    )
    p.add_argument("--qa-pairs", type=Path, default=Path("data/processed/qa_pairs.json"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-per-type", type=int, default=5)
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
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prune_old_to_match_new(
    qa_pairs: list[dict],
    target_episodes: int,
    target_per_type: int,
) -> list[str]:
    """Greedy prune of old QAs so their distinct-episode union == target_episodes.

    Removes one QA at a time, preferring removals that drop the most distinct
    episodes; ties broken by qa_type over-representation (so types stay balanced).
    Returns the qa_ids (q000-style) of the kept old subset.
    """
    old = [(i, q) for i, q in enumerate(qa_pairs) if q.get("era") == "old"]
    kept = list(old)
    while True:
        ep_count: Counter = Counter()
        for _, q in kept:
            for e in q["reference_episodes"]:
                ep_count[int(e)] += 1
        if len(ep_count) <= target_episodes:
            break
        type_counts = Counter(q["qa_type"] for _, q in kept)
        best_key = None
        best_pos = 0
        for pos, (_, q) in enumerate(kept):
            drop = sum(1 for e in q["reference_episodes"] if ep_count[int(e)] == 1)
            excess = type_counts[q["qa_type"]] - target_per_type
            key = (drop, excess, -pos)
            if best_key is None or key > best_key:
                best_key = key
                best_pos = pos
        kept.pop(best_pos)
    return [f"q{i:03d}" for i, _ in kept]


def bootstrap_ci(vals: np.ndarray, rng, n_boot: int = 5000) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohort_means(
    df: pd.DataFrame, metrics: list[str], rng
) -> pd.DataFrame:
    """One row per cohort × metric with mean + 95% bootstrap CI over per-QA means."""
    per_qa = df.groupby(["qa_id", "era"])[metrics].mean().reset_index()
    records: list[dict] = []
    for cohort in COHORT_ORDER:
        sub = per_qa[per_qa.era == cohort]
        for m in metrics:
            vals = sub[m].values
            mean = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals, rng)
            records.append(
                {
                    "cohort": cohort,
                    "metric": m,
                    "n_qa": len(sub),
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(records)


def compute_deltas(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Δ (new − old) and Δ (new − mixed) with Mann-Whitney p-values."""
    per_qa = df.groupby(["qa_id", "era"])[metrics].mean().reset_index()
    old = per_qa[per_qa.era == "old"]
    mixed = per_qa[per_qa.era == "mixed"]
    new = per_qa[per_qa.era == "new"]
    rows = []
    for m in metrics:
        for name, ref in [("new_minus_old", old), ("new_minus_mixed", mixed)]:
            u, p = mannwhitneyu(new[m].values, ref[m].values, alternative="two-sided")
            rows.append(
                {
                    "delta": name,
                    "metric": m,
                    "mean_new": float(new[m].mean()),
                    "mean_ref": float(ref[m].mean()),
                    "delta_value": float(new[m].mean() - ref[m].mean()),
                    "mwu_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def _draw_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    metric: str,
    label: str,
    ylim: tuple[float, float],
    lower_better: bool,
) -> None:
    bar_w = 0.62
    xs = np.arange(len(COHORT_ORDER))
    for x, cohort in zip(xs, COHORT_ORDER):
        row = summary[(summary.cohort == cohort) & (summary.metric == metric)].iloc[0]
        mean = row["mean"]
        yerr = np.array([[mean - row["ci_lo"]], [row["ci_hi"] - mean]])
        ax.bar(
            [x], [mean], width=bar_w,
            color=COHORT_COLOR[cohort],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            [x], [mean], yerr=yerr,
            fmt="none", ecolor=INK, elinewidth=0.9, capsize=3.0, capthick=0.9, zorder=3,
        )
        # Place label above the top of the 95% CI cap, not above the bar top.
        ax.text(
            x, row["ci_hi"] + (ylim[1] - ylim[0]) * 0.025,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=8.5, color=INK,
        )

    # Δ annotations beneath the x-axis
    d_old = deltas[(deltas.delta == "new_minus_old") & (deltas.metric == metric)].iloc[0]
    d_mixed = deltas[(deltas.delta == "new_minus_mixed") & (deltas.metric == metric)].iloc[0]
    sign = "↓" if lower_better else "↑"
    note = (
        f"Δ new–old = {d_old['delta_value']:+.3f} (p={d_old['mwu_p']:.3f})\n"
        f"Δ new–mixed = {d_mixed['delta_value']:+.3f} (p={d_mixed['mwu_p']:.3f})"
    )
    ax.text(
        0.5, -0.30, note,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.0, color=MUTED,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([COHORT_LABEL[c] for c in COHORT_ORDER], fontsize=8.0)
    ax.set_xlim(-0.6, len(COHORT_ORDER) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel(label + ("  (lower=better)" if lower_better else ""))
    ax.set_title(label + f" {sign}" if lower_better else label)
    ax.tick_params(length=0)


def plot_panel(
    summary: pd.DataFrame, deltas: pd.DataFrame, out_stub: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    for ax, (metric, label, ylim, lower_better) in zip(axes.flat, METRICS):
        _draw_metric(ax, summary, deltas, metric, label, ylim, lower_better)

    # One shared legend above the panel.
    handles = [
        mpatches.Patch(
            facecolor=COHORT_COLOR[c],
            edgecolor="white",
            linewidth=0.8,
            label=COHORT_LABEL[c],
        )
        for c in COHORT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Era comparison  ·  old / mixed / new  ·  per-QA means, 95% bootstrap CI",
        fontsize=10.5, y=1.06,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.65)
    for ext in ("png", "pdf"):
        fig.savefig(out_stub.with_suffix(f".{ext}"), facecolor="white")
        print(f"Saved {out_stub.with_suffix('.' + ext)}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    rng = default_rng(args.seed)

    csv_path = args.run_dir / "per_example_metrics.csv"
    df = pd.read_csv(csv_path)
    metric_names = [m for m, *_ in METRICS]

    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "era_matched")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = cohort_means(df, metric_names, rng)
    summary.to_csv(out_dir / "era_matched_means.csv", index=False)
    deltas = compute_deltas(df, metric_names)
    deltas.to_csv(out_dir / "era_matched_deltas.csv", index=False)

    plot_panel(summary, deltas, out_dir / "figure_era_matched")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Cohort means (per-QA avg over 36 conditions, 95% bootstrap CI) ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Deltas and Mann-Whitney U tests ===")
    print(deltas.round(4).to_string(index=False))
    print(f"\nArtifacts in {out_dir}")


if __name__ == "__main__":
    main()
