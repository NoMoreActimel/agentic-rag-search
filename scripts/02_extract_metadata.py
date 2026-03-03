#!/usr/bin/env python3
"""Step 2: Extract structured metadata from podcast transcripts using Gemini."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs, METADATA_DIR, TRANSCRIPTS_SUBSET_PARQUET
from src.llm.metadata_extractor import extract_all_metadata


async def main():
    ensure_dirs()

    if not TRANSCRIPTS_SUBSET_PARQUET.exists():
        print(f"Error: {TRANSCRIPTS_SUBSET_PARQUET} not found.")
        print("Run 01_build_raw_dataset.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Step 2: Extracting metadata with Gemini")
    print("=" * 60)

    results = await extract_all_metadata()

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    metadata_files = list(METADATA_DIR.glob("*.json"))
    print(f"Metadata files: {len(metadata_files)} in {METADATA_DIR}")

    if results:
        # Check schema completeness
        required_keys = {"episode_id", "guest_info", "summary", "main_entities", "topics"}
        for r in results[:3]:
            missing = required_keys - set(r.keys())
            if missing:
                print(f"  Warning: Episode {r.get('episode_id')} missing keys: {missing}")
            else:
                print(f"  Episode {r['episode_id']}: schema OK "
                      f"({len(r.get('main_entities', []))} entities, "
                      f"{len(r.get('topics', []))} topics)")


if __name__ == "__main__":
    asyncio.run(main())
