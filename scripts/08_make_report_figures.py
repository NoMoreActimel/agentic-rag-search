#!/usr/bin/env python3
"""Generate report-focused figures with uncertainty bars for ACL progress report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report figures from run outputs")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    per_example = pd.read_csv(run_dir / "per_example_metrics.csv")
    summary = pd.read_csv(run_dir / "summary_by_condition.csv")

    out_dir = run_dir / "analysis" / "plots" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # 1) Success by retriever with 95% CI over examples
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.barplot(
        data=per_example,
        x="retriever",
        y="success",
        errorbar=("ci", 95),
        capsize=0.15,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Success by Retriever (95% CI)")
    ax.set_xlabel("Retriever backend")
    ax.set_ylabel("Success rate")
    fig.savefig(out_dir / "fig_success_by_retriever_ci.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Process feedback delta by retriever and step
    none = summary[summary["process_feedback"] == False].copy()  # noqa: E712
    proc = summary[summary["process_feedback"] == True].copy()  # noqa: E712
    merged = none.merge(proc, on=["retriever", "max_steps"], suffixes=("_none", "_process"))
    merged["delta_success"] = merged["success_process"] - merged["success_none"]
    heat = merged.pivot(index="retriever", columns="max_steps", values="delta_success")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "Success delta (process - none)"},
        ax=ax,
    )
    ax.set_title("Process Feedback Effect on Success")
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel("Retriever")
    fig.savefig(out_dir / "fig_process_feedback_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Steps tradeoff with CI
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.pointplot(
        data=per_example,
        x="max_steps",
        y="success",
        hue="process_feedback",
        errorbar=("ci", 95),
        dodge=0.2,
        markers=["o", "s"],
        linestyles=["-", "--"],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Step Budget vs Success (95% CI)")
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel("Success rate")
    ax.legend(title="Process feedback", labels=["None", "Process"])
    fig.savefig(out_dir / "fig_steps_vs_success_ci.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Question-type difficulty with CI
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(
        data=per_example,
        x="qa_type",
        y="success",
        errorbar=("ci", 95),
        capsize=0.15,
        order=sorted(per_example["qa_type"].unique().tolist()),
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Success by Question Type (95% CI)")
    ax.set_xlabel("Question type")
    ax.set_ylabel("Success rate")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out_dir / "fig_success_by_qatype_ci.png", bbox_inches="tight")
    plt.close(fig)

    # 5) Cost-success frontier
    cond = summary.copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.scatterplot(
        data=cond,
        x="cost_usd",
        y="success",
        hue="retriever",
        style="process_feedback",
        s=90,
        ax=ax,
    )
    ax.set_title("Condition Frontier: Cost vs Success")
    ax.set_xlabel("Mean cost per example (USD)")
    ax.set_ylabel("Success rate")
    fig.savefig(out_dir / "fig_cost_success_frontier.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved report figures to {out_dir}")


if __name__ == "__main__":
    main()
