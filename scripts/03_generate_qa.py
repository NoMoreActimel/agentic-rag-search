#!/usr/bin/env python3
"""Step 3: Generate synthetic Q&A pairs using Gemini."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs, METADATA_DIR, QA_PAIRS_JSON
from src.qa.qa_generator import generate_all_qa_pairs, load_all_metadata


async def main():
    ensure_dirs()

    metadata = load_all_metadata()
    if not metadata:
        print(f"Error: No metadata files found in {METADATA_DIR}")
        print("Run 02_extract_metadata.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"Step 3: Generating Q&A pairs from {len(metadata)} episodes")
    print("=" * 60)

    pairs = await generate_all_qa_pairs(metadata=metadata)

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    print(f"Total Q&A pairs: {len(pairs)}")

    # Count by type
    from collections import Counter
    type_counts = Counter(p.qa_type for p in pairs)
    for qt, cnt in sorted(type_counts.items()):
        print(f"  {qt}: {cnt} pairs")

    # Preview first pair of each type
    seen_types = set()
    for p in pairs:
        if p.qa_type not in seen_types:
            seen_types.add(p.qa_type)
            print(f"\nSample {p.qa_type}:")
            print(f"  Q: {p.question[:120]}...")
            print(f"  A: {p.answer[:120]}...")
            print(f"  Episodes: {p.reference_episodes}")


if __name__ == "__main__":
    asyncio.run(main())
