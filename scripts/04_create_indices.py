#!/usr/bin/env python3
"""Step 4: Create BM25 and embedding search indices."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    ensure_dirs,
    BM25_DIR,
    CHUNKS_PARQUET,
    EMBEDDINGS_DIR,
    TRANSCRIPTS_SUBSET_PARQUET,
)
from src.indexing.chunker import build_chunks
from src.indexing.bm25_index import build_bm25_index, search_bm25
from src.indexing.embedding_index import build_embedding_index, search_embeddings


async def main():
    ensure_dirs()

    if not TRANSCRIPTS_SUBSET_PARQUET.exists():
        print(f"Error: {TRANSCRIPTS_SUBSET_PARQUET} not found.")
        print("Run 01_build_raw_dataset.py first.")
        sys.exit(1)

    # Step 4a: Chunk transcripts
    print("=" * 60)
    print("Step 4a: Chunking transcripts")
    print("=" * 60)
    chunks_df = build_chunks()

    # Step 4b: Build BM25 index
    print("\n" + "=" * 60)
    print("Step 4b: Building BM25 index")
    print("=" * 60)
    bm25 = build_bm25_index()

    # Step 4c: Build embedding index
    print("\n" + "=" * 60)
    print("Step 4c: Building embedding index")
    print("=" * 60)
    faiss_index = await build_embedding_index()

    # Verification: test search
    print("\n" + "=" * 60)
    print("Verification: Sample searches")
    print("=" * 60)

    test_queries = [
        "artificial intelligence safety",
        "consciousness and the brain",
        "future of robotics",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # BM25
        bm25_results = search_bm25(query, retriever=bm25, chunks_df=chunks_df, top_k=3)
        print("  BM25 top results:")
        for r in bm25_results:
            print(f"    Ep#{r['episode_id']} ({r['guest']}): "
                  f"score={r['score']:.3f} — {r['text'][:80]}...")

        # Embedding
        emb_results = await search_embeddings(
            query, index=faiss_index, chunks_df=chunks_df, top_k=3,
        )
        print("  Embedding top results:")
        for r in emb_results:
            print(f"    Ep#{r['episode_id']} ({r['guest']}): "
                  f"score={r['score']:.3f} — {r['text'][:80]}...")

    # Summary
    print("\n" + "=" * 60)
    print("Index Summary")
    print("=" * 60)
    print(f"Chunks: {len(chunks_df)} at {CHUNKS_PARQUET}")
    print(f"BM25 index: {BM25_DIR}")
    print(f"Embedding index: {EMBEDDINGS_DIR} ({faiss_index.ntotal} vectors)")


if __name__ == "__main__":
    asyncio.run(main())
