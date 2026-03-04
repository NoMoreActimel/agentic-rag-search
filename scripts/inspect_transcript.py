#!/usr/bin/env python3
"""Quick script to inspect a transcript from the dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import TRANSCRIPTS_SUBSET_PARQUET

df = pd.read_parquet(TRANSCRIPTS_SUBSET_PARQUET)

# Pick a scraped episode (they have richer formatting with speaker labels)
scraped = df[df["source"] == "scraped"].iloc[0]

print(f"Episode #{int(scraped['episode_id'])}: {scraped['title']}")
print(f"Guest: {scraped['guest']}")
print(f"Date: {scraped['date']}")
print(f"Source: {scraped['source']}")
print(f"Length: {scraped['transcript_length']:,} chars")
print("=" * 80)
print(scraped["full_transcript"][:3000])
print("..." )
