#!/usr/bin/env python3
"""Restore: rebuild chunks.parquet + BM25 from full 426-ep transcripts.parquet.

Leaves embeddings untouched. Use after a git pull reverted chunks.parquet to 50-ep.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BM25_DIR, CHUNKS_PARQUET, TRANSCRIPTS_PARQUET
from src.indexing.bm25_index import build_bm25_index
from src.indexing.chunker import build_chunks


def main() -> None:
    print("Re-chunking from FULL transcripts.parquet ...")
    chunks_df = build_chunks(
        subset_path=str(TRANSCRIPTS_PARQUET),
        output_path=str(CHUNKS_PARQUET),
    )
    print(f"Chunks: {len(chunks_df)} over {chunks_df['episode_id'].nunique()} episodes")

    print("\nRebuilding BM25 ...")
    build_bm25_index(
        chunks_path=str(CHUNKS_PARQUET),
        output_dir=str(BM25_DIR),
    )
    print("Done.")


if __name__ == "__main__":
    main()
