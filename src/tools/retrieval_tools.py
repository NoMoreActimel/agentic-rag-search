"""Retrieval tools: unified interface for grep, BM25, and embedding search.

Each tool takes a query string and returns a list of RetrievalResult objects.
The agent calls these tools during its iterative search loop.
"""

import re
from dataclasses import dataclass

import pandas as pd

from config.settings import CHUNKS_PARQUET, BM25_DIR, EMBEDDINGS_DIR
from src.indexing.bm25_index import load_bm25_index, search_bm25
from src.indexing.embedding_index import load_embedding_index, search_embeddings
from src.llm.gemini_client import GeminiClient


@dataclass
class RetrievalResult:
    """A single retrieved chunk with metadata."""

    chunk_id: int
    episode_id: int
    guest: str
    chunk_index: int
    text: str
    score: float
    source: str  # "grep", "bm25", or "embedding"

    def to_context_string(self) -> str:
        """Format for injection into agent context."""
        return (
            f"[Chunk {self.chunk_id} | Episode #{self.episode_id} | "
            f"Guest: {self.guest} | Score: {self.score:.3f} | Via: {self.source}]\n"
            f"{self.text}"
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "episode_id": self.episode_id,
            "guest": self.guest,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": self.score,
            "source": self.source,
        }


class GrepTool:
    """Exact-match retrieval using regex pattern matching over chunks."""

    def __init__(self, chunks_df: pd.DataFrame | None = None):
        self.chunks_df = chunks_df if chunks_df is not None else pd.read_parquet(str(CHUNKS_PARQUET))

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        Search for exact or near-exact matches using regex.

        The agent should reformulate queries to keywords/phrases for grep.
        Matches are ranked by number of query term matches found in the chunk.
        """
        # Split query into individual terms (lowercase)
        terms = [t.strip().lower() for t in re.split(r'\s+', query) if len(t.strip()) > 2]
        if not terms:
            return []

        # Score each chunk by number of matching terms
        scores = []
        for idx, row in self.chunks_df.iterrows():
            text_lower = row["text"].lower()
            match_count = sum(1 for t in terms if t in text_lower)
            if match_count > 0:
                scores.append((idx, match_count / len(terms)))  # Normalize to 0-1

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            row = self.chunks_df.iloc[idx]
            results.append(RetrievalResult(
                chunk_id=int(row["chunk_id"]),
                episode_id=int(row["episode_id"]),
                guest=str(row["guest"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                score=float(score),
                source="grep",
            ))

        return results


class BM25Tool:
    """BM25 sparse retrieval over chunked transcripts."""

    def __init__(
        self,
        chunks_df: pd.DataFrame | None = None,
        retriever=None,
    ):
        self.chunks_df = chunks_df if chunks_df is not None else pd.read_parquet(str(CHUNKS_PARQUET))
        self.retriever = retriever if retriever is not None else load_bm25_index()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search using BM25 ranking."""
        raw_results = search_bm25(
            query=query,
            retriever=self.retriever,
            chunks_df=self.chunks_df,
            top_k=top_k,
        )
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                episode_id=r["episode_id"],
                guest=r["guest"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                score=r["score"],
                source="bm25",
            )
            for r in raw_results
        ]


class EmbeddingTool:
    """Dense embedding retrieval using Gemini embeddings + FAISS."""

    def __init__(
        self,
        chunks_df: pd.DataFrame | None = None,
        index=None,
        client: GeminiClient | None = None,
    ):
        self.chunks_df = chunks_df if chunks_df is not None else pd.read_parquet(str(CHUNKS_PARQUET))
        self.index = index if index is not None else load_embedding_index()
        self.client = client

    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search using cosine similarity on Gemini embeddings."""
        raw_results = await search_embeddings(
            query=query,
            index=self.index,
            chunks_df=self.chunks_df,
            client=self.client,
            top_k=top_k,
        )
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                episode_id=r["episode_id"],
                guest=r["guest"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                score=r["score"],
                source="embedding",
            )
            for r in raw_results
        ]


class ToolRegistry:
    """Registry of all available retrieval tools for the agent."""

    def __init__(
        self,
        chunks_df: pd.DataFrame | None = None,
        bm25_retriever=None,
        faiss_index=None,
        gemini_client: GeminiClient | None = None,
    ):
        self.chunks_df = chunks_df if chunks_df is not None else pd.read_parquet(str(CHUNKS_PARQUET))
        self.grep = GrepTool(chunks_df=self.chunks_df)
        self.bm25 = BM25Tool(chunks_df=self.chunks_df, retriever=bm25_retriever)
        self.embedding = EmbeddingTool(
            chunks_df=self.chunks_df,
            index=faiss_index,
            client=gemini_client,
        )

    async def search(self, tool_name: str, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Dispatch search to the named tool."""
        if tool_name == "grep":
            return self.grep.search(query, top_k=top_k)
        elif tool_name == "bm25":
            return self.bm25.search(query, top_k=top_k)
        elif tool_name == "embedding":
            return await self.embedding.search(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown tool: {tool_name}. Use 'grep', 'bm25', or 'embedding'.")

    @staticmethod
    def available_tools() -> list[str]:
        return ["grep", "bm25", "embedding"]

    @staticmethod
    def tool_descriptions() -> dict[str, str]:
        return {
            "grep": (
                "Exact-match keyword search. Best for specific names, terms, or phrases. "
                "Use short, targeted keyword queries."
            ),
            "bm25": (
                "Probabilistic term-frequency search (BM25). Good for natural language queries "
                "with multiple relevant terms. Handles stemming automatically."
            ),
            "embedding": (
                "Semantic similarity search using neural embeddings. Best for conceptual or "
                "paraphrased queries where exact terms may not appear in the text."
            ),
        }