"""Merge HuggingFace and scraped data into unified parquet datasets."""

import pandas as pd

from config.settings import (
    SUBSET_HF_COUNT,
    SUBSET_SCRAPED_COUNT,
    TRANSCRIPTS_PARQUET,
    TRANSCRIPTS_SUBSET_PARQUET,
)


def build_full_dataset(hf_df: pd.DataFrame, scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge HuggingFace and scraped transcripts into a single dataset.

    For overlapping episode IDs, prefer the scraped version (more complete metadata).
    """
    # Handle empty DataFrames
    if scraped_df.empty:
        print("Warning: No scraped episodes — using HuggingFace data only")
        return hf_df.sort_values("episode_id").reset_index(drop=True)
    if hf_df.empty:
        print("Warning: No HuggingFace episodes — using scraped data only")
        return scraped_df.sort_values("episode_id").reset_index(drop=True)

    # Find overlapping episodes
    hf_ids = set(hf_df["episode_id"])
    scraped_ids = set(scraped_df["episode_id"])
    overlap = hf_ids & scraped_ids

    if overlap:
        print(f"Found {len(overlap)} overlapping episodes — preferring scraped versions")
        hf_df = hf_df[~hf_df["episode_id"].isin(overlap)]

    full = pd.concat([hf_df, scraped_df], ignore_index=True)
    full = full.sort_values("episode_id").reset_index(drop=True)

    print(f"Full dataset: {len(full)} episodes")
    print(f"  From HuggingFace: {(full['source'] == 'huggingface').sum()}")
    print(f"  From scraping: {(full['source'] == 'scraped').sum()}")

    return full


def select_subset(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select a balanced subset of episodes for the working dataset.

    Picks SUBSET_HF_COUNT from HuggingFace and SUBSET_SCRAPED_COUNT from scraped,
    choosing episodes with the longest transcripts for quality.
    """
    hf = full_df[full_df["source"] == "huggingface"]
    scraped = full_df[full_df["source"] == "scraped"]

    # Pick episodes with longest transcripts (likely highest quality/completeness)
    hf_subset = hf.nlargest(SUBSET_HF_COUNT, "transcript_length")
    scraped_subset = scraped.nlargest(SUBSET_SCRAPED_COUNT, "transcript_length")

    # If we don't have enough from one source, fill from the other
    hf_count = len(hf_subset)
    scraped_count = len(scraped_subset)

    if hf_count < SUBSET_HF_COUNT and scraped_count > SUBSET_SCRAPED_COUNT:
        extra_needed = SUBSET_HF_COUNT - hf_count
        remaining_scraped = scraped[~scraped["episode_id"].isin(scraped_subset["episode_id"])]
        extra = remaining_scraped.nlargest(extra_needed, "transcript_length")
        scraped_subset = pd.concat([scraped_subset, extra])
    elif scraped_count < SUBSET_SCRAPED_COUNT and hf_count > SUBSET_HF_COUNT:
        extra_needed = SUBSET_SCRAPED_COUNT - scraped_count
        remaining_hf = hf[~hf["episode_id"].isin(hf_subset["episode_id"])]
        extra = remaining_hf.nlargest(extra_needed, "transcript_length")
        hf_subset = pd.concat([hf_subset, extra])

    subset = pd.concat([hf_subset, scraped_subset], ignore_index=True)
    subset = subset.sort_values("episode_id").reset_index(drop=True)

    print(f"Subset: {len(subset)} episodes")
    print(f"  From HuggingFace: {(subset['source'] == 'huggingface').sum()}")
    print(f"  From scraping: {(subset['source'] == 'scraped').sum()}")

    return subset


def save_datasets(full_df: pd.DataFrame, subset_df: pd.DataFrame) -> None:
    """Save both datasets to parquet."""
    TRANSCRIPTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_parquet(TRANSCRIPTS_PARQUET, index=False)
    print(f"Saved full dataset to {TRANSCRIPTS_PARQUET}")

    subset_df.to_parquet(TRANSCRIPTS_SUBSET_PARQUET, index=False)
    print(f"Saved subset to {TRANSCRIPTS_SUBSET_PARQUET}")
