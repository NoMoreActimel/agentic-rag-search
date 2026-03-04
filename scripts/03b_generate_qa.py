#!/usr/bin/env python3
"""Step 3b: Generate Q&A pairs from candidates using Gemini."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs, METADATA_DIR, PROCESSED_DIR, QA_PAIRS_JSON, QA_PAIRS_PER_TYPE
from src.qa.qa_generator import Candidate, QAPair, load_all_metadata
from src.qa.type1_multihop import MultiHopGenerator
from src.qa.type2_comparative import ComparativeGenerator
from src.qa.type3_temporal import TemporalGenerator
from src.qa.type4_aggregation import AggregationGenerator
from src.llm.gemini_client import GeminiClient

CANDIDATES_JSON = PROCESSED_DIR / "qa_candidates.json"


async def main():
    ensure_dirs()

    if not CANDIDATES_JSON.exists():
        print(f"Error: {CANDIDATES_JSON} not found.")
        print("Run 03a_find_qa_candidates.py first.")
        sys.exit(1)

    metadata = load_all_metadata()
    if not metadata:
        print(f"Error: No metadata files found in {METADATA_DIR}")
        sys.exit(1)

    with open(CANDIDATES_JSON) as f:
        raw_candidates = json.load(f)

    # Group candidates by type
    candidates_by_type: dict[str, list[Candidate]] = {}
    for c in raw_candidates:
        cand = Candidate(**c)
        if cand.qa_type not in candidates_by_type:
            candidates_by_type[cand.qa_type] = []
        candidates_by_type[cand.qa_type].append(cand)

    print(f"Loaded {len(raw_candidates)} candidates from {CANDIDATES_JSON}")
    print("=" * 60)

    client = GeminiClient()
    type_to_gen = {
        "multihop": MultiHopGenerator,
        "comparative": ComparativeGenerator,
        "temporal": TemporalGenerator,
        "aggregation": AggregationGenerator,
    }

    all_pairs: list[QAPair] = []

    for qa_type, candidates in candidates_by_type.items():
        gen_cls = type_to_gen.get(qa_type)
        if not gen_cls:
            print(f"Unknown type: {qa_type}, skipping")
            continue

        gen = gen_cls(client=client, metadata=metadata)
        print(f"\nGenerating {qa_type} ({len(candidates)} candidates, need {QA_PAIRS_PER_TYPE})...")

        pairs = []
        for cand in candidates:
            if len(pairs) >= QA_PAIRS_PER_TYPE:
                break
            pair = await gen.generate_from_candidate(cand)
            if pair is not None:
                pairs.append(pair)
                print(f"  [{len(pairs)}/{QA_PAIRS_PER_TYPE}] Q: {pair.question[:80]}...")

        all_pairs.extend(pairs)
        print(f"  Generated {len(pairs)}/{QA_PAIRS_PER_TYPE} {qa_type} pairs")

    # Save
    output = [p.to_dict() for p in all_pairs]
    with open(QA_PAIRS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_pairs)} Q&A pairs to {QA_PAIRS_JSON}")

    # Summary
    from collections import Counter
    type_counts = Counter(p.qa_type for p in all_pairs)
    for qt, cnt in sorted(type_counts.items()):
        print(f"  {qt}: {cnt} pairs")

    client.print_stats()


if __name__ == "__main__":
    asyncio.run(main())
