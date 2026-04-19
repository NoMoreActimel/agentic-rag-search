"""BM25 index using bm25s for fast sparse retrieval."""

from pathlib import Path

import bm25s
import pandas as pd
try:
    import Stemmer
except Exception:
    Stemmer = None

from config.settings import BM25_DIR, CHUNKS_PARQUET


def build_bm25_index(
    chunks_path: str | None = None,
    output_dir: str | None = None,
) -> bm25s.BM25:
    """
    Build a BM25 index from chunked transcripts.

    Uses bm25s with Snowball stemming for efficient retrieval.
    """
    chunks_path = chunks_path or str(CHUNKS_PARQUET)
    output_dir = Path(output_dir or BM25_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(chunks_path)
    corpus = df["text"].tolist()

    print(f"Building BM25 index over {len(corpus)} chunks...")

    # Tokenize with stemming
    stemmer = Stemmer.Stemmer("english") if Stemmer is not None else None
    tokenized = bm25s.tokenize(corpus, stemmer=stemmer)

    # Build index
    retriever = bm25s.BM25()
    retriever.index(tokenized)

    # Save
    retriever.save(str(output_dir))
    print(f"BM25 index saved to {output_dir}")

    return retriever


def load_bm25_index(index_dir: str | None = None) -> bm25s.BM25:
    """Load a saved BM25 index."""
    index_dir = index_dir or str(BM25_DIR)
    retriever = bm25s.BM25.load(index_dir)
    return retriever


def search_bm25(
    query: str,
    retriever: bm25s.BM25 | None = None,
    chunks_df: pd.DataFrame | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search the BM25 index.

    Returns list of dicts with chunk metadata and scores.
    """
    if retriever is None:
        retriever = load_bm25_index()
    if chunks_df is None:
        chunks_df = pd.read_parquet(str(CHUNKS_PARQUET))

    stemmer = Stemmer.Stemmer("english") if Stemmer is not None else None
    query_tokens = bm25s.tokenize([query], stemmer=stemmer)

    results, scores = retriever.retrieve(query_tokens, k=top_k)

    output = []
    for idx, score in zip(results[0], scores[0]):
        if idx < len(chunks_df):
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
