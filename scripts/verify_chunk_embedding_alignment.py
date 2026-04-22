#!/usr/bin/env python3
"""Verify chunks.parquet aligns with embeddings.npy after re-chunking.

Strategy: re-embed 5 sampled chunks via Gemini, cosine-compare to stored
embeddings at the same row index. If alignment holds, cos≈1.0.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CHUNKS_PARQUET, EMBEDDINGS_DIR
from src.llm.gemini_client import GeminiClient


async def main() -> None:
    chunks = pd.read_parquet(CHUNKS_PARQUET)
    emb = np.load(EMBEDDINGS_DIR / "embeddings.npy")

    print(f"chunks.parquet rows: {len(chunks):,}")
    print(f"embeddings.npy:      {emb.shape}")
    if len(chunks) != emb.shape[0]:
        print("MISMATCH: row counts differ → alignment impossible.")
        sys.exit(1)

    sample_indices = [0, 1, 1000, 33000, len(chunks) - 1]
    client = GeminiClient()
    texts = [chunks.iloc[i]["text"] for i in sample_indices]
    fresh = await client.embed(texts)
    fresh_arr = np.array(fresh, dtype=np.float32)
    fresh_arr /= np.linalg.norm(fresh_arr, axis=1, keepdims=True)

    print(f"\n{'idx':>8} {'ep':>5} {'cos_sim':>10}   {'verdict':<8}  text_preview")
    print("-" * 100)
    for i, si in enumerate(sample_indices):
        stored = emb[si]
        cos = float(np.dot(fresh_arr[i], stored))
        verdict = "OK" if cos > 0.99 else "MISMATCH"
        preview = str(chunks.iloc[si]["text"])[:60].replace("\n", " ")
        ep = int(chunks.iloc[si]["episode_id"])
        print(f"{si:>8} {ep:>5} {cos:>10.6f}   {verdict:<8}  {preview}")

    # Also: self-retrieval test — query FAISS with stored vector, should return same index.
    import faiss
    idx = faiss.read_index(str(EMBEDDINGS_DIR / "faiss.index"))
    print("\nSelf-retrieval check (FAISS top-1 of stored vector):")
    hits = idx.search(emb[sample_indices], 1)
    returned = hits[1].flatten().tolist()
    scores = hits[0].flatten().tolist()
    for si, ret, score in zip(sample_indices, returned, scores):
        ok = "OK" if ret == si else "FAIL"
        print(f"  query row {si:>6} → top-1 row {ret:>6}  score={score:.4f}  [{ok}]")


if __name__ == "__main__":
    asyncio.run(main())
