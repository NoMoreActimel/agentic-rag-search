#!/usr/bin/env python3
"""Merge shard result directories into one combined run folder.

Reads each shard's runs.jsonl, deduplicates on (condition_id, qa_id) with last-write-wins,
then rebuilds per_example_metrics.csv and summary CSVs.

Does not merge gemini_usage_final.json (per-shard totals remain authoritative for billing).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_jsonl_records(paths: list[Path]) -> list[dict]:
    """Last record wins per (condition_id, qa_id)."""
    last: dict[tuple[str, str], dict] = {}
    for p in paths:
        runs = p / "runs.jsonl"
        if not runs.exists():
            print(f"WARNING: missing runs.jsonl in {p}, skipping")
            continue
        with open(runs, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec.get("condition_id", ""), rec.get("qa_id", ""))
                if key[0] and key[1]:
                    last[key] = rec
    keys_sorted = sorted(last.keys(), key=lambda k: (k[0], k[1]))
    return [last[k] for k in keys_sorted]


def _aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in [
            "success",
            "judge_score",
            "retrieval_precision",
            "hallucination_rate",
            "lexical_f1",
            "exact_match",
            "reference_episode_recall",
            "trajectory_length",
            "elapsed_seconds",
            "cost_usd",
        ]
        if c in df.columns
    ]
    group_cols = [
        c
        for c in [
            "suite",
            "condition_id",
            "retriever",
            "max_steps",
            "process_feedback",
            "quality_reweight",
            "judge_mode",
            "qa_count",
        ]
        if c in df.columns
    ]
    return (
        df.groupby(group_cols, as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values([c for c in ["suite", "condition_id"] if c in df.columns])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard output directories.")
    parser.add_argument(
        "merged_dir",
        type=Path,
        help="Output directory to create/populate with merged artifacts.",
    )
    parser.add_argument(
        "shard_dirs",
        nargs="+",
        type=Path,
        help="One or more shard result directories (each containing runs.jsonl).",
    )
    args = parser.parse_args()

    merged = args.merged_dir.resolve()
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "trajectories").mkdir(exist_ok=True)

    shard_dirs = [p.resolve() for p in args.shard_dirs]
    rows = _load_jsonl_records(shard_dirs)
    if not rows:
        raise SystemExit("No rows found to merge.")

    out_jsonl = merged / "runs.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(merged / "per_example_metrics.csv", index=False)

    summary = _aggregate_results(df)
    summary.to_csv(merged / "summary_by_condition.csv", index=False)

    overview = {
        "rows": len(df),
        "conditions": int(df["condition_id"].nunique()),
        "qa_count_per_condition": int(df["qa_count"].max()) if "qa_count" in df.columns else 0,
        "overall_success_rate": float(df["success"].astype(float).mean()),
        "overall_judge_score": float(df["judge_score"].mean()),
        "overall_retrieval_precision": float(df["retrieval_precision"].mean()),
        "overall_hallucination_rate": float(df["hallucination_rate"].mean()),
        "overall_cost_usd": float(df["cost_usd"].sum()),
    }
    if "status" in df.columns:
        overview["rows_status_error"] = int((df["status"] == "error").sum())
    with open(merged / "summary_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=True)

    manifest = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "shard_dirs": [str(p) for p in shard_dirs],
        "rows": len(rows),
    }
    with open(merged / "manifest_merge.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    # Copy conditions JSON from first shard if present
    for name in ("conditions_main.json", "conditions_oracle_mini.json"):
        src = shard_dirs[0] / name
        if src.exists():
            shutil.copy2(src, merged / name)

    print(f"Merged {len(rows)} rows into {merged}")


if __name__ == "__main__":
    main()
