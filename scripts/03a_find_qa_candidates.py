#!/usr/bin/env python3
"""Step 3a: Find candidate episode groups for Q&A generation (no LLM calls)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs, METADATA_DIR, PROCESSED_DIR, QA_PAIRS_PER_TYPE
from src.qa.qa_generator import load_all_metadata
from src.qa.type1_multihop import MultiHopGenerator
from src.qa.type2_comparative import ComparativeGenerator
from src.qa.type3_temporal import TemporalGenerator
from src.qa.type4_aggregation import AggregationGenerator

CANDIDATES_JSON = PROCESSED_DIR / "qa_candidates.json"


def main():
    ensure_dirs()

    metadata = load_all_metadata()
    if not metadata:
        print(f"Error: No metadata files found in {METADATA_DIR}")
        sys.exit(1)

    print(f"Loaded {len(metadata)} episode metadata files")
    print("=" * 60)

    # We pass client=None since Stage 1 doesn't use the LLM
    generators = [
        MultiHopGenerator(client=None, metadata=metadata),
        ComparativeGenerator(client=None, metadata=metadata),
        TemporalGenerator(client=None, metadata=metadata),
        AggregationGenerator(client=None, metadata=metadata),
    ]

    all_candidates = []
    for gen in generators:
        # Find 2x candidates (extras in case some fail in Stage 2)
        candidates = gen.find_candidates(count=QA_PAIRS_PER_TYPE * 2)
        all_candidates.extend(candidates)

        print(f"\n{gen.qa_type}: {len(candidates)} candidates")
        for i, c in enumerate(candidates):
            eps = c.episode_ids
            print(f"  {i+1}. Episodes {eps} — {c.connection[:90]}")

    # Save
    output = [c.to_dict() for c in all_candidates]
    with open(CANDIDATES_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_candidates)} candidates to {CANDIDATES_JSON}")


if __name__ == "__main__":
    main()
