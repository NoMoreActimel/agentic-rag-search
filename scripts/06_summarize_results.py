#!/usr/bin/env python3
"""Generate a short markdown narrative from experiment outputs."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to data/results/<run_dir>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.results_dir / "summary_by_condition.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("summary_by_condition.csv is empty.")

    best_success = df.sort_values("success", ascending=False).iloc[0]
    best_judge = df.sort_values("judge_score", ascending=False).iloc[0]
    lowest_hallucination = df.sort_values("hallucination_rate", ascending=True).iloc[0]

    lines = [
        "# Experiment Summary",
        "",
        "## Top Conditions",
        "",
        (
            f"- Best success rate: `{best_success['condition_id']}` "
            f"with success={best_success['success']:.3f}, "
            f"judge_score={best_success['judge_score']:.3f}, "
            f"retrieval_precision={best_success['retrieval_precision']:.3f}."
        ),
        (
            f"- Best judge score: `{best_judge['condition_id']}` "
            f"with judge_score={best_judge['judge_score']:.3f}, "
            f"success={best_judge['success']:.3f}."
        ),
        (
            f"- Lowest hallucination: `{lowest_hallucination['condition_id']}` "
            f"with hallucination_rate={lowest_hallucination['hallucination_rate']:.3f}."
        ),
        "",
        "## Caveats",
        "",
        "- Metrics rely on LLM judge outputs and should be interpreted with evaluator variance in mind.",
        "- Smoke-mode results are for pipeline validation, not final claims.",
        "- Report final conclusions from full-grid runs on the full QA set.",
        "",
    ]

    output_md = args.results_dir / "report_summary.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
