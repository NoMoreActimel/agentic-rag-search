#!/usr/bin/env python3
"""Standalone README figure: panel (b) of the judge-score figure.

Renders only the "with process feedback" panel of Figure 2 (judge score vs.
max retrieval steps) as a single self-contained plot, so it can sit side by
side with Figure 1 (feedback_effect) in the README. Reuses the exact plotting
code from scripts/11_paper_figures.py for visual consistency with the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from importlib import import_module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
pf = import_module("11_paper_figures")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("data/results/submit87_fast_merged"))
    args = p.parse_args()

    df = pd.read_csv(args.run_dir / "per_example_metrics.csv")
    df = df[df["quality_reweight"] == False].copy()  # noqa: E712

    out_dir = args.run_dir / "analysis" / "plots" / "paper_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf.setup_style()

    agg = pf.aggregate(df, "judge_score")
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    pf.plot_panel(
        ax, agg, "judge_score", feedback=True,
        ylabel="LLM-as-judge score (1–5)", ylim=(3.3, 4.8),
        title="With process feedback (N=87 QAs/point, 95% CI)",
    )
    ax.legend(loc="lower right", title=None, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"figure_2b_judge_score_feedback.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
