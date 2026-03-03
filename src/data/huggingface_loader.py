"""Load and consolidate Lex Fridman Podcast transcripts from HuggingFace."""

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config.settings import HF_DATASET_NAME, RAW_HF_DIR


def load_hf_transcripts(cache_dir: str | None = None) -> pd.DataFrame:
    """
    Load the nmac/lex_fridman_podcast dataset from HuggingFace.

    The dataset has ~803K rows with segment-level data.
    We group by episode, sort by start timestamp, and concatenate text
    into full transcripts.

    Returns:
        DataFrame with columns: episode_id, title, guest, full_transcript, source
    """
    cache_dir = cache_dir or str(RAW_HF_DIR)

    print("Loading HuggingFace dataset (this may take a moment on first run)...")
    ds = load_dataset(HF_DATASET_NAME, split="train", cache_dir=cache_dir)
    df = ds.to_pandas()

    print(f"Loaded {len(df)} segments across {df['id'].nunique()} episodes")

    episodes = []
    for ep_id, group in tqdm(df.groupby("id"), desc="Consolidating episodes"):
        group = group.sort_values("start")
        full_transcript = " ".join(group["text"].astype(str).tolist())

        # Extract title and guest from the first row
        first_row = group.iloc[0]
        title = str(first_row.get("title", ""))
        guest = str(first_row.get("guest", ""))

        episodes.append({
            "episode_id": int(ep_id),
            "title": title,
            "guest": guest,
            "date": None,  # Not available in HF dataset
            "full_transcript": full_transcript,
            "source": "huggingface",
            "transcript_length": len(full_transcript),
        })

    result = pd.DataFrame(episodes)
    result = result.sort_values("episode_id").reset_index(drop=True)
    print(f"Consolidated {len(result)} episodes from HuggingFace")
    return result
