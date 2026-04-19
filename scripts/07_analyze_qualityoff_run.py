#!/usr/bin/env python3
"""Generate rigorous analysis tables and plots for a completed run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze quality-off experiment run artifacts")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory in data/results/<run_name>",
    )
    return parser.parse_args()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def validate_artifacts(run_dir: Path) -> dict:
    required = [
        "manifest.json",
        "summary_overview.json",
        "summary_by_condition.csv",
        "per_example_metrics.csv",
        "runs.jsonl",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_overview = json.loads((run_dir / "summary_overview.json").read_text(encoding="utf-8"))
    summary_by_condition = pd.read_csv(run_dir / "summary_by_condition.csv")
    per_example = pd.read_csv(run_dir / "per_example_metrics.csv")

    expected_rows = int(summary_overview["rows"])
    expected_conditions = int(summary_overview["conditions"])
    expected_q_per_cond = int(summary_overview["qa_count_per_condition"])

    checks = {
        "run_dir": str(run_dir),
        "manifest_qa_count": int(manifest.get("qa_count", -1)),
        "rows_expected": expected_rows,
        "rows_actual": int(len(per_example)),
        "conditions_expected": expected_conditions,
        "conditions_actual": int(per_example["condition_id"].nunique()),
        "qa_count_per_condition_expected": expected_q_per_cond,
        "qa_count_per_condition_actual_min": int(per_example.groupby("condition_id").size().min()),
        "qa_count_per_condition_actual_max": int(per_example.groupby("condition_id").size().max()),
        "api_error_like_rows": int(
            per_example["reason"]
            .fillna("")
            .str.contains("RESOURCE_EXHAUSTED|429|spending cap", case=False, regex=True)
            .sum()
        ),
        "error_answer_rows": int(
            (per_example["predicted_answer"] == "Unable to generate answer due to an error.").sum()
        ),
        "all_quality_off": bool((summary_by_condition["quality_reweight"] == False).all()),  # noqa: E712
    }
    return {
        "manifest": manifest,
        "summary_overview": summary_overview,
        "summary_by_condition": summary_by_condition,
        "per_example": per_example,
        "checks": checks,
    }


def derive_tables(summary_by_condition: pd.DataFrame, per_example: pd.DataFrame) -> dict[str, pd.DataFrame]:
    retriever = (
        per_example.groupby("retriever", as_index=False)[
            ["success", "judge_score", "retrieval_precision", "hallucination_rate", "lexical_f1", "cost_usd", "elapsed_seconds"]
        ]
        .mean(numeric_only=True)
        .sort_values("success", ascending=False)
    )

    steps = (
        per_example.groupby("max_steps", as_index=False)[
            ["success", "judge_score", "retrieval_precision", "hallucination_rate", "lexical_f1", "cost_usd", "elapsed_seconds"]
        ]
        .mean(numeric_only=True)
        .sort_values("max_steps")
    )

    process = (
        per_example.groupby("process_feedback", as_index=False)[
            ["success", "judge_score", "retrieval_precision", "hallucination_rate", "lexical_f1", "cost_usd", "elapsed_seconds"]
        ]
        .mean(numeric_only=True)
        .sort_values("process_feedback")
    )
    process["process_feedback"] = process["process_feedback"].map({False: "none", True: "process"})

    by_q = (
        per_example.groupby(["qa_id", "qa_type"], as_index=False)[
            ["success", "judge_score", "retrieval_precision", "hallucination_rate", "lexical_f1", "cost_usd", "elapsed_seconds"]
        ]
        .mean(numeric_only=True)
        .sort_values(["qa_type", "qa_id"])
    )

    cond_eff = summary_by_condition.copy()
    cond_eff["success_per_dollar"] = cond_eff["success"] / cond_eff["cost_usd"].clip(lower=1e-9)
    cond_eff["success_per_second"] = cond_eff["success"] / cond_eff["elapsed_seconds"].clip(lower=1e-9)
    cond_eff = cond_eff.sort_values(["success", "judge_score", "hallucination_rate"], ascending=[False, False, True])

    none = summary_by_condition[summary_by_condition["process_feedback"] == False].copy()  # noqa: E712
    proc = summary_by_condition[summary_by_condition["process_feedback"] == True].copy()  # noqa: E712
    keys = ["retriever", "max_steps"]
    merged = none.merge(
        proc,
        on=keys,
        suffixes=("_none", "_process"),
    )
    delta_cols = ["success", "judge_score", "retrieval_precision", "hallucination_rate", "lexical_f1", "cost_usd", "elapsed_seconds"]
    for col in delta_cols:
        merged[f"delta_{col}_process_minus_none"] = merged[f"{col}_process"] - merged[f"{col}_none"]
    process_delta = merged[
        keys + [f"delta_{c}_process_minus_none" for c in delta_cols]
    ].sort_values(keys)

    q_heat = (
        per_example.pivot_table(
            index="qa_id",
            columns="condition_id",
            values="success",
            aggfunc="mean",
        )
        .reset_index()
    )

    return {
        "aggregate_by_retriever": retriever,
        "aggregate_by_step": steps,
        "aggregate_by_process_feedback": process,
        "per_question_diagnostics": by_q,
        "condition_efficiency_ranking": cond_eff,
        "process_vs_none_deltas": process_delta,
        "question_condition_success_matrix": q_heat,
    }


def make_plots(run_dir: Path, summary_by_condition: pd.DataFrame, per_example: pd.DataFrame) -> list[str]:
    plot_style()
    out_dir = run_dir / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    cond = summary_by_condition.copy().sort_values("success", ascending=False)
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=cond, x="condition_id", y="success", hue="retriever", ax=ax)
    ax.set_title("Success Rate by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Success")
    ax.tick_params(axis="x", rotation=80)
    ax.legend(title="Retriever")
    path = out_dir / "condition_success_rate.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=cond, x="condition_id", y="hallucination_rate", hue="retriever", ax=ax)
    ax.set_title("Hallucination Rate by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Hallucination Rate")
    ax.tick_params(axis="x", rotation=80)
    ax.legend(title="Retriever")
    path = out_dir / "condition_hallucination_rate.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    agg_r = per_example.groupby("retriever", as_index=False)[["success", "judge_score", "retrieval_precision", "hallucination_rate"]].mean()
    melted = agg_r.melt(id_vars=["retriever"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="retriever", ax=ax)
    ax.set_title("Retriever-Level Metric Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    path = out_dir / "retriever_metric_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    pair = summary_by_condition.pivot_table(
        index=["retriever", "max_steps"],
        columns="process_feedback",
        values="success",
        aggfunc="mean",
    ).reset_index()
    pair["delta"] = pair.get(True, 0) - pair.get(False, 0)
    heat = pair.pivot(index="retriever", columns="max_steps", values="delta")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Process Feedback Delta (Success, Process - None)")
    path = out_dir / "process_feedback_delta_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    agg_steps = per_example.groupby("max_steps", as_index=False)[["success", "hallucination_rate", "cost_usd", "elapsed_seconds"]].mean()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    sns.lineplot(data=agg_steps, x="max_steps", y="success", marker="o", label="Success", ax=ax1)
    sns.lineplot(data=agg_steps, x="max_steps", y="hallucination_rate", marker="o", label="Hallucination", ax=ax1)
    sns.lineplot(data=agg_steps, x="max_steps", y="cost_usd", marker="o", color="purple", label="Cost USD", ax=ax2)
    ax1.set_title("Step-Count Tradeoff: Quality vs Cost")
    ax1.set_ylabel("Success / Hallucination")
    ax2.set_ylabel("Cost USD")
    path = out_dir / "step_tradeoff_quality_cost.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    q_heat = per_example.pivot_table(
        index="qa_id",
        columns="condition_id",
        values="success",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(q_heat, annot=False, cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-Question Success Across Conditions")
    ax.set_xlabel("Condition")
    ax.set_ylabel("QA ID")
    path = out_dir / "question_condition_success_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    eff = summary_by_condition.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=eff,
        x="cost_usd",
        y="success",
        hue="retriever",
        style="process_feedback",
        s=90,
        ax=ax,
    )
    ax.set_title("Cost vs Success by Condition")
    ax.set_xlabel("Cost USD")
    ax.set_ylabel("Success")
    path = out_dir / "cost_vs_success_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(str(path))

    return created


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    data = validate_artifacts(run_dir)
    summary_by_condition = data["summary_by_condition"]
    per_example = data["per_example"]

    table_dir = run_dir / "analysis" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    checks_df = pd.DataFrame([data["checks"]])
    save_csv(checks_df, table_dir / "run_integrity_checks.csv")

    tables = derive_tables(summary_by_condition=summary_by_condition, per_example=per_example)
    for name, df in tables.items():
        save_csv(df, table_dir / f"{name}.csv")

    plots = make_plots(run_dir=run_dir, summary_by_condition=summary_by_condition, per_example=per_example)

    summary = {
        "run_dir": str(run_dir),
        "overview": data["summary_overview"],
        "checks": data["checks"],
        "generated_tables": sorted([str(p) for p in table_dir.glob("*.csv")]),
        "generated_plots": plots,
    }
    out_summary = run_dir / "analysis" / "analysis_manifest.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote analysis manifest: {out_summary}")


if __name__ == "__main__":
    main()
