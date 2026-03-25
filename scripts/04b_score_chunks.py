#!/usr/bin/env python3
"""Step 4b (optional): Precompute chunk quality scores using LLM verifier.

This is the "Static Ranking" method from the proposal.
Run this BEFORE 05_run_experiments.py if you want to test the quality_scores=True experiments.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs
from src.tools.chunk_quality import score_all_chunks, load_quality_scores, compute_chunk_quality


async def main():
    ensure_dirs()

    print("=" * 60)
    print("Scoring chunk quality with LLM verifier")
    print("=" * 60)

    scores = await score_all_chunks()

    # Summary statistics
    qualities = [compute_chunk_quality(s) for s in scores.values()]
    ads = sum(1 for s in scores.values() if s.get("is_ad_or_filler", False))

    print(f"\nSummary:")
    print(f"  Total chunks scored: {len(scores)}")
    print(f"  Ads/filler detected: {ads} ({ads/len(scores)*100:.1f}%)")
    print(f"  Avg quality score: {sum(qualities)/len(qualities):.3f}")
    print(f"  Min quality: {min(qualities):.3f}")
    print(f"  Max quality: {max(qualities):.3f}")


if __name__ == "__main__":
    asyncio.run(main())