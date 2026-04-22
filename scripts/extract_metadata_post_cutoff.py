#!/usr/bin/env python3
"""Extract metadata for post-cutoff episodes (>2025-01-31) missing from data/processed/metadata/.

Reads from transcripts.parquet (full corpus), filters by date, and invokes
`extract_episode_metadata` for each missing episode. Skips existing JSON files.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import METADATA_DIR, TRANSCRIPTS_PARQUET
from src.llm.gemini_client import GeminiClient
from src.llm.metadata_extractor import extract_episode_metadata

CUTOFF = pd.Timestamp("2025-01-31", tz="UTC")


async def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(TRANSCRIPTS_PARQUET).copy()
    df["dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    post = df[df["dt"] > CUTOFF].sort_values("episode_id")
    print(f"Post-cutoff episodes: {len(post)}")

    missing = [int(r.episode_id) for r in post.itertuples() if not (METADATA_DIR / f"{int(r.episode_id)}.json").exists()]
    print(f"Missing metadata: {len(missing)} → {missing}")
    if not missing:
        return

    client = GeminiClient()
    work = post[post["episode_id"].isin(missing)]

    async def extract_and_save(row):
        ep_id = int(row["episode_id"])
        meta = await extract_episode_metadata(
            client=client,
            episode_id=ep_id,
            title=str(row.get("title") or ""),
            guest=str(row.get("guest") or ""),
            transcript=str(row["full_transcript"]),
        )
        if meta:
            with open(METADATA_DIR / f"{ep_id}.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  saved ep {ep_id}")
        return ep_id

    tasks = [extract_and_save(row) for _, row in work.iterrows()]
    await asyncio.gather(*tasks, return_exceptions=True)
    client.print_stats()


if __name__ == "__main__":
    asyncio.run(main())
