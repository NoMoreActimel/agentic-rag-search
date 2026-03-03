"""Embedding index using Gemini embeddings + FAISS for dense retrieval."""

import asyncio
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import (
    CHUNKS_PARQUET,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_DIR,
    GEMINI_EMBEDDING_DIMS,
)
from src.llm.gemini_client import GeminiClient

EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss.index"


async def build_embedding_index(
    chunks_path: str | None = None,
    output_dir: str | None = None,
) -> faiss.IndexFlatIP:
    """
    Build a FAISS embedding index from chunked transcripts.

    Uses Gemini embeddings (768 dims) with cosine similarity via
    normalized vectors + IndexFlatIP (inner product).
    """
    chunks_path = chunks_path or str(CHUNKS_PARQUET)
    output_dir = Path(output_dir or EMBEDDINGS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(chunks_path)
    texts = df["text"].tolist()

    print(f"Embedding {len(texts)} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    client = GeminiClient()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = await client.embed(batch)
        all_embeddings.extend(embeddings)

    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms

    # Save raw embeddings
    np.save(str(output_dir / EMBEDDINGS_FILE), embeddings_array)

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(GEMINI_EMBEDDING_DIMS)
    index.add(embeddings_array)

    # Save FAISS index
    faiss.write_index(index, str(output_dir / INDEX_FILE))

    print(f"FAISS index: {index.ntotal} vectors, dim={GEMINI_EMBEDDING_DIMS}")
    print(f"Saved to {output_dir}")

    client.print_stats()
    return index


def load_embedding_index(index_dir: str | None = None) -> faiss.IndexFlatIP:
    """Load a saved FAISS index."""
    index_dir = Path(index_dir or EMBEDDINGS_DIR)
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    return index


async def search_embeddings(
    query: str,
    index: faiss.IndexFlatIP | None = None,
    chunks_df: pd.DataFrame | None = None,
    client: GeminiClient | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search the embedding index with a query.

    Returns list of dicts with chunk metadata and scores.
    """
    if index is None:
        index = load_embedding_index()
    if chunks_df is None:
        chunks_df = pd.read_parquet(str(CHUNKS_PARQUET))
    if client is None:
        client = GeminiClient()

    # Embed query
    query_embedding = await client.embed([query])
    query_vec = np.array(query_embedding, dtype=np.float32)

    # Normalize
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Search
    scores, indices = index.search(query_vec, top_k)

    output = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(chunks_df):
            row = chunks_df.iloc[idx]
            output.append({
                "chunk_id": int(row["chunk_id"]),
                "episode_id": int(row["episode_id"]),
                "guest": row["guest"],
                "chunk_index": int(row["chunk_index"]),
                "text": row["text"],
                "score": float(score),
            })

    return output
