#!/usr/bin/env python3
"""Step 1: Build raw transcript dataset from HuggingFace + web scraping."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ensure_dirs, TRANSCRIPTS_PARQUET, TRANSCRIPTS_SUBSET_PARQUET
from src.data.huggingface_loader import load_hf_transcripts
from src.data.web_scraper import scrape_all_transcripts
from src.data.dataset_builder import build_full_dataset, select_subset, save_datasets


def main():
    ensure_dirs()

    # Step 1a: Load HuggingFace data
    print("=" * 60)
    print("Step 1a: Loading HuggingFace dataset")
    print("=" * 60)
    hf_df = load_hf_transcripts()

    # Step 1b: Scrape website data
    print("\n" + "=" * 60)
    print("Step 1b: Scraping lexfridman.com transcripts")
    print("=" * 60)
    scraped_df = scrape_all_transcripts()

    # Step 1c: Merge and save
    print("\n" + "=" * 60)
    print("Step 1c: Building unified dataset")
    print("=" * 60)
    full_df = build_full_dataset(hf_df, scraped_df)
    subset_df = select_subset(full_df)
    save_datasets(full_df, subset_df)

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    print(f"Full dataset: {len(full_df)} episodes at {TRANSCRIPTS_PARQUET}")
    print(f"Subset: {len(subset_df)} episodes at {TRANSCRIPTS_SUBSET_PARQUET}")
    print(f"Episode ID range: {full_df['episode_id'].min()} - {full_df['episode_id'].max()}")
    print(f"Avg transcript length: {full_df['transcript_length'].mean():,.0f} chars")


if __name__ == "__main__":
    main()
