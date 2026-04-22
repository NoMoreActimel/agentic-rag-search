#!/usr/bin/env python3
"""Deep analysis of the 36-condition submit87 run.

Adds plots and tables that the existing paper_bars / quality_onoff / era /
iter_vs_judge_em folders don't cover:

  1. Factor main-effects + 2-way interaction table.
  2. Cost–quality and latency–quality Pareto plots (36 conditions).
  3. QA-type × retriever heatmaps (judge, success, hallucination, ref. recall).
  4. Retrieval-precision → success calibration (does precision actually predict
     correctness?).
  5. Trajectory-length realization (do agents use the steps budget?).
  6. Per-question consistency (agreement across conditions, hardest questions).
  7. Top/bottom condition ranking tables.

Writes everything to <run-dir>/analysis/plots/deep_analysis/ .
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETRIEVERS = ["grep", "bm25", "embedding"]
RETRIEVER_COLOR = {"grep": "#BC5A4E", "bm25": "#3A6A87", "embedding": "#5D8A6A"}
INK = "#1F2328"
GRID = "#DDE0E4"

METRICS = ["success", "judge_score", "hallucination_rate",
           "retrieval_precision", "reference_episode_recall", "lexical_f1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path,
                   default=Path("data/results/submit87_fast_merged"))
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.edgecolor": "#B8BEC6",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.grid.axis": "y",
        "axes.grid": True,
    })


# --------------------------------------------------------------------------- #
# 1. Main effects + interactions
# --------------------------------------------------------------------------- #

def main_effects(df: pd.DataFrame, out_dir: Path) -> None:
    factors = {
        "retriever": RETRIEVERS,
        "max_steps": [2, 3, 4],
        "process_feedback": [False, True],
        "quality_reweight": [False, True],
    }
    rows = []
    for metric in METRICS:
        grand = df[metric].mean()
        for fac, levels in factors.items():
            for lvl in levels:
                sub = df[df[fac] == lvl]
                mean = sub[metric].mean()
                rows.append({
                    "metric": metric,
                    "factor": fac,
                    "level": str(lvl),
                    "mean": mean,
                    "delta_vs_grand": mean - grand,
                    "n": len(sub),
                })
    me = pd.DataFrame(rows)
    me.to_csv(out_dir / "main_effects.csv", index=False)

    # Plot: main effects as horizontal bars, one panel per metric.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, metric in zip(axes.flat, METRICS):
        sub = me[me.metric == metric].copy()
        sub["label"] = sub.factor + " = " + sub.level
        order = [
            "retriever = grep", "retriever = bm25", "retriever = embedding",
            "max_steps = 2", "max_steps = 3", "max_steps = 4",
            "process_feedback = False", "process_feedback = True",
            "quality_reweight = False", "quality_reweight = True",
        ]
        sub = sub.set_index("label").loc[order].reset_index()
        colors = ["#999" if d < 0 else "#3A6A87" for d in sub["delta_vs_grand"]]
        ax.barh(sub["label"], sub["delta_vs_grand"], color=colors)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.invert_yaxis()
        ax.set_title(f"{metric}  (grand mean = {df[metric].mean():.3f})")
        ax.set_xlabel("Δ vs grand mean")
    fig.suptitle("Marginal main effects — Δ from grand mean (36 conditions × 87 QA)")
    fig.tight_layout()
    fig.savefig(out_dir / "main_effects.png")
    plt.close(fig)

    # 2-way interaction table: only the interesting metric pairs.
    inter_rows = []
    pairs = list(itertools.combinations(factors.keys(), 2))
    for a, b in pairs:
        for la in factors[a]:
            for lb in factors[b]:
                sub = df[(df[a] == la) & (df[b] == lb)]
                if sub.empty:
                    continue
                entry = {"factor_a": a, "level_a": str(la),
                         "factor_b": b, "level_b": str(lb),
                         "n": len(sub)}
                for m in ["success", "judge_score",
                          "hallucination_rate", "cost_usd"]:
                    entry[m] = sub[m].mean()
                inter_rows.append(entry)
    pd.DataFrame(inter_rows).to_csv(
        out_dir / "two_way_interactions.csv", index=False)


# --------------------------------------------------------------------------- #
# 2. Cost / latency quality frontier
# --------------------------------------------------------------------------- #

def cost_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (df.groupby(["retriever", "max_steps", "process_feedback",
                       "quality_reweight"])
             .agg(success=("success", "mean"),
                  judge_score=("judge_score", "mean"),
                  hallucination=("hallucination_rate", "mean"),
                  cost=("cost_usd", "mean"),
                  elapsed=("elapsed_seconds", "mean"),
                  traj_len=("trajectory_length", "mean"))
             .reset_index())

    def pareto(xs, ys, minimize_x=True, maximize_y=True):
        idx = np.argsort(xs if minimize_x else -np.asarray(xs))
        pf = []
        best = -np.inf
        for i in idx:
            y = ys[i]
            if y > best:
                pf.append(i)
                best = y
        return sorted(pf, key=lambda i: xs[i])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, x_col, x_label, log in [
        (axes[0], "cost", "Mean cost per question (USD)", True),
        (axes[1], "elapsed", "Mean latency per question (s)", False),
    ]:
        for r in RETRIEVERS:
            sub = agg[agg.retriever == r]
            ax.scatter(sub[x_col], sub.judge_score,
                       s=55 + sub.max_steps * 40,
                       marker="o" if sub.process_feedback.iloc[0] is None else "o",
                       color=RETRIEVER_COLOR[r], alpha=0.9,
                       edgecolor="white", linewidth=0.8, label=r)
        # mark feedback vs not with edge style
        for _, row in agg.iterrows():
            ax.scatter(row[x_col], row.judge_score,
                       s=55 + row.max_steps * 40,
                       facecolor="none",
                       edgecolor="black" if row.process_feedback else "none",
                       linewidth=1.0, zorder=3)
        # Pareto (minimize x, maximize judge)
        xs = agg[x_col].values
        ys = agg.judge_score.values
        pf = pareto(xs, ys, minimize_x=True, maximize_y=True)
        ax.plot(xs[pf], ys[pf], "--", color="black", linewidth=1, alpha=0.6)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean LLM-judge score (1–5)")
        ax.set_title(x_label.split(" per ")[0] + " frontier")
    # legend
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          color=RETRIEVER_COLOR[r], label=r,
                          markersize=9, markeredgecolor="white")
               for r in RETRIEVERS]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                              color="white", markeredgecolor="black",
                              markersize=10, label="+ feedback"))
    handles.append(plt.Line2D([0], [0], linestyle="--", color="black",
                              alpha=0.6, label="Pareto frontier"))
    axes[0].legend(handles=handles, loc="lower right", frameon=False)
    fig.suptitle("Quality vs. cost & latency — marker size ∝ max_steps")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_latency_frontier.png")
    plt.close(fig)

    agg["cost_per_judge_point"] = agg.cost / agg.judge_score
    agg.sort_values("judge_score", ascending=False).to_csv(
        out_dir / "condition_agg.csv", index=False)


# --------------------------------------------------------------------------- #
# 3. QA-type × retriever heatmaps
# --------------------------------------------------------------------------- #

def qa_type_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("judge_score", "LLM-judge score", False),
               ("success", "Success rate", False),
               ("hallucination_rate", "Hallucination rate", True),
               ("reference_episode_recall", "Ref. episode recall", False)]

    qa_types = sorted(df.qa_type.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    for ax, (m, label, lower_better) in zip(axes.flat, metrics):
        piv = (df.groupby(["qa_type", "retriever"])[m]
                 .mean().unstack("retriever")[RETRIEVERS])
        im = ax.imshow(piv.values, aspect="auto",
                       cmap="RdYlGn_r" if lower_better else "RdYlGn",
                       vmin=piv.values.min(), vmax=piv.values.max())
        ax.set_xticks(range(len(RETRIEVERS)))
        ax.set_xticklabels(RETRIEVERS)
        ax.set_yticks(range(len(qa_types)))
        ax.set_yticklabels(qa_types)
        ax.set_title(label)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("QA-type × retriever (averaged over steps / feedback / quality)")
    fig.tight_layout()
    fig.savefig(out_dir / "qa_type_by_retriever.png")
    plt.close(fig)

    # Same breakdown but also split by feedback (important for agentic claim).
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, qt in zip(axes.flat, qa_types):
        sub = df[df.qa_type == qt]
        g = (sub.groupby(["retriever", "process_feedback"])
                .judge_score.mean().unstack("process_feedback"))
        g = g.loc[RETRIEVERS]
        x = np.arange(len(RETRIEVERS))
        w = 0.38
        ax.bar(x - w / 2, g[False], w, label="no feedback",
               color=[RETRIEVER_COLOR[r] for r in RETRIEVERS], alpha=0.55)
        ax.bar(x + w / 2, g[True], w, label="+ feedback",
               color=[RETRIEVER_COLOR[r] for r in RETRIEVERS])
        ax.set_xticks(x)
        ax.set_xticklabels(RETRIEVERS)
        ax.set_title(f"{qt}  (n={len(sub)} rows)")
        ax.set_ylim(3.0, 4.75)
        if ax is axes[0]:
            ax.set_ylabel("Judge score")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle("Judge score by QA-type × retriever × feedback")
    fig.tight_layout()
    fig.savefig(out_dir / "qa_type_judge_by_feedback.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Retrieval precision → success calibration
# --------------------------------------------------------------------------- #

def precision_calibration(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    df["prec_bin"] = pd.cut(df.retrieval_precision,
                            bins=[-0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 1.01],
                            labels=["0–.1", ".1–.2", ".2–.3", ".3–.4",
                                    ".4–.5", ".5–.6", ".6–.7", ".7–.8",
                                    ".8–.9", ".9–1"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for ax, r in zip(axes, RETRIEVERS):
        sub = df[df.retriever == r]
        g = sub.groupby("prec_bin", observed=False).agg(
            success=("success", "mean"),
            judge=("judge_score", "mean"),
            n=("success", "size")).dropna()
        ax.bar(range(len(g)), g["success"], color=RETRIEVER_COLOR[r], alpha=0.85)
        for i, n in enumerate(g["n"]):
            ax.text(i, g["success"].iloc[i] + 0.02, f"n={int(n)}",
                    ha="center", fontsize=8, color="#333")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{r}  —  Pearson r = "
                     f"{sub.retrieval_precision.corr(sub.success):.3f}")
        ax.set_xlabel("retrieval_precision bucket")
        if ax is axes[0]:
            ax.set_ylabel("Mean success")
    fig.suptitle("Is retrieval precision predictive of success?")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_calibration.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Trajectory-length realization
# --------------------------------------------------------------------------- #

def traj_realization(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (df.groupby(["retriever", "max_steps", "process_feedback"])
             .trajectory_length.mean().reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in RETRIEVERS:
        for fb, ls in [(False, "--"), (True, "-")]:
            sub = agg[(agg.retriever == r) & (agg.process_feedback == fb)]
            ax.plot(sub.max_steps, sub.trajectory_length,
                    ls, color=RETRIEVER_COLOR[r],
                    marker="o", markersize=7,
                    label=f"{r} ({'+fb' if fb else 'no fb'})")
    ax.plot([2, 4], [2, 4], ":", color="black", alpha=0.5,
            label="y = max_steps (upper bound)")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("max_steps (budget)")
    ax.set_ylabel("Mean trajectory length (actual)")
    ax.set_title("Agents don't always use their step budget")
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_realization.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Per-question consistency
# --------------------------------------------------------------------------- #

def per_question(df: pd.DataFrame, out_dir: Path) -> None:
    # Fraction of conditions that got success==1 for each QA.
    per_q = df.groupby("qa_id").agg(
        success_rate=("success", "mean"),
        judge_mean=("judge_score", "mean"),
        judge_std=("judge_score", "std"),
        qa_type=("qa_type", "first"),
        era=("era", "first")).reset_index()
    per_q = per_q.sort_values("success_rate")
    per_q.to_csv(out_dir / "per_question.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.hist(per_q.success_rate, bins=20, color="#3A6A87",
            edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Fraction of 36 conditions that got this QA right")
    ax.set_ylabel("# questions")
    ax.set_title(f"Per-question difficulty (87 QAs)  "
                 f"— {int((per_q.success_rate == 0).sum())} impossible, "
                 f"{int((per_q.success_rate == 1).sum())} universally solved")

    ax = axes[1]
    for qt, color in zip(sorted(per_q.qa_type.unique()),
                         ["#BC5A4E", "#3A6A87", "#5D8A6A", "#B48A3A"]):
        sub = per_q[per_q.qa_type == qt]
        ax.scatter(sub.success_rate, sub.judge_mean,
                   color=color, alpha=0.7, label=qt, s=38,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Per-question success rate")
    ax.set_ylabel("Per-question mean judge score")
    ax.set_title("Per-question success vs judge — by QA type")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "per_question_difficulty.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Top/bottom condition table
# --------------------------------------------------------------------------- #

def top_bottom(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (df.groupby(["condition_id", "retriever", "max_steps",
                       "process_feedback", "quality_reweight"])
             .agg(success=("success", "mean"),
                  judge=("judge_score", "mean"),
                  ref_recall=("reference_episode_recall", "mean"),
                  halluc=("hallucination_rate", "mean"),
                  cost=("cost_usd", "mean"),
                  latency=("elapsed_seconds", "mean"))
             .reset_index()
             .sort_values("judge", ascending=False))
    agg.to_csv(out_dir / "condition_ranking.csv", index=False)

    # Print-ready top/bottom by multiple criteria.
    with open(out_dir / "top_bottom.txt", "w") as f:
        for metric, asc in [("judge", False), ("success", False),
                            ("halluc", True), ("cost", True)]:
            s = agg.sort_values(metric, ascending=asc)
            f.write(f"=== Top 5 by {metric} ===\n")
            f.write(s.head(5).to_string(index=False) + "\n\n")
            f.write(f"=== Bottom 5 by {metric} ===\n")
            f.write(s.tail(5).to_string(index=False) + "\n\n")


# --------------------------------------------------------------------------- #
# 8. Summary figure: one-slide overview
# --------------------------------------------------------------------------- #

def overview(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    # Panel A: retriever main effect (judge + success + halluc + refrecall)
    ax = axes[0, 0]
    g = df.groupby("retriever").agg(
        judge=("judge_score", "mean"),
        success=("success", "mean"),
        halluc=("hallucination_rate", "mean"),
        recall=("reference_episode_recall", "mean")).loc[RETRIEVERS]
    x = np.arange(len(RETRIEVERS))
    w = 0.2
    ax.bar(x - 1.5 * w, g.judge / 5.0, w, color="#3A6A87", label="judge / 5")
    ax.bar(x - 0.5 * w, g.success, w, color="#5D8A6A", label="success")
    ax.bar(x + 0.5 * w, g.recall, w, color="#B48A3A", label="ref. recall")
    ax.bar(x + 1.5 * w, g.halluc, w, color="#BC5A4E", label="halluc (↓)")
    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVERS)
    ax.set_ylim(0, 1)
    ax.set_title("Retriever main effects (all cells pooled)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    # Panel B: feedback gain per retriever
    ax = axes[0, 1]
    g = (df.groupby(["retriever", "process_feedback"])
           .judge_score.mean().unstack("process_feedback"))
    g = g.loc[RETRIEVERS]
    gain = (g[True] - g[False]).values
    ax.bar(RETRIEVERS, gain,
           color=[RETRIEVER_COLOR[r] for r in RETRIEVERS])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Δ judge_score  (+fb minus no fb)")
    ax.set_title("Process-feedback effect — retriever-specific")
    for i, v in enumerate(gain):
        ax.text(i, v + (0.01 if v > 0 else -0.02),
                f"{v:+.3f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=9)

    # Panel C: quality reweight is net-neutral
    ax = axes[1, 0]
    g = (df.groupby(["retriever", "quality_reweight"])
           .judge_score.mean().unstack("quality_reweight"))
    g = g.loc[RETRIEVERS]
    gain = (g[True] - g[False]).values
    ax.bar(RETRIEVERS, gain,
           color=[RETRIEVER_COLOR[r] for r in RETRIEVERS])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Δ judge_score  (quality on minus off)")
    ax.set_title("Chunk-quality reweight — retriever-specific")
    for i, v in enumerate(gain):
        ax.text(i, v + (0.01 if v > 0 else -0.02),
                f"{v:+.3f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=9)

    # Panel D: step budget gain
    ax = axes[1, 1]
    g = (df.groupby(["retriever", "max_steps"]).judge_score.mean()
           .unstack("max_steps")).loc[RETRIEVERS]
    for r in RETRIEVERS:
        ax.plot([2, 3, 4], g.loc[r].values,
                marker="o", color=RETRIEVER_COLOR[r], label=r, linewidth=2)
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("max_steps")
    ax.set_ylabel("Mean judge_score")
    ax.set_title("Step-budget effect — retriever-specific")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Deep-analysis overview — 36 conditions × 87 QA = 3132 rows")
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    setup_style()
    out_dir = args.out_dir or (args.run_dir / "analysis" / "plots" / "deep_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.run_dir / "per_example_metrics.csv")
    print(f"[load] {len(df)} rows, {df.condition_id.nunique()} conditions")

    main_effects(df, out_dir)
    print("  ✓ main effects")
    cost_frontier(df, out_dir)
    print("  ✓ cost/latency frontier")
    qa_type_heatmaps(df, out_dir)
    print("  ✓ QA-type × retriever")
    precision_calibration(df, out_dir)
    print("  ✓ precision calibration")
    traj_realization(df, out_dir)
    print("  ✓ trajectory realization")
    per_question(df, out_dir)
    print("  ✓ per-question difficulty")
    top_bottom(df, out_dir)
    print("  ✓ top/bottom tables")
    overview(df, out_dir)
    print("  ✓ overview")

    print(f"\nArtifacts in: {out_dir}")


if __name__ == "__main__":
    main()
