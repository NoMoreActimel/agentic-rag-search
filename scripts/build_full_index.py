#!/usr/bin/env python3
"""One-off: build chunks, BM25, and embedding indices over the FULL 426-episode corpus.

Reads data/processed/transcripts.parquet (instead of transcripts_subset.parquet) and
writes into data/indices/ (overwriting the 50-episode artifacts — back them up first).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BM25_DIR,
    CHUNKS_PARQUET,
    EMBEDDINGS_DIR,
    TRANSCRIPTS_PARQUET,
    ensure_dirs,
)
from src.indexing.bm25_index import build_bm25_index
from src.indexing.chunker import build_chunks
from src.indexing.embedding_index import build_embedding_index


async def main() -> None:
    ensure_dirs()

    if not TRANSCRIPTS_PARQUET.exists():
        print(f"ERROR: {TRANSCRIPTS_PARQUET} not found.")
        sys.exit(1)

    print("=" * 60)
    print(f"Step 1: Chunking FULL corpus ({TRANSCRIPTS_PARQUET})")
    print("=" * 60)
    chunks_df = build_chunks(
        subset_path=str(TRANSCRIPTS_PARQUET),
        output_path=str(CHUNKS_PARQUET),
    )

    print("\n" + "=" * 60)
    print("Step 2: Building BM25 index")
    print("=" * 60)
    build_bm25_index(
        chunks_path=str(CHUNKS_PARQUET),
        output_dir=str(BM25_DIR),
    )

    print("\n" + "=" * 60)
    print("Step 3: Building embedding index (this is the long one)")
    print("=" * 60)
    idx = await build_embedding_index(
        chunks_path=str(CHUNKS_PARQUET),
        output_dir=str(EMBEDDINGS_DIR),
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Chunks: {len(chunks_df)} from {chunks_df['episode_id'].nunique()} episodes")
    print(f"FAISS vectors: {idx.ntotal}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
