"""Chunk quality scoring: LLM-based static ranking for preprocessing the index.

This implements the "Static Ranking" method from the proposal:
LLM scores individual chunks on Integrity and Information Density.
Scores are computed offline and used to re-weight retrieval results.
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.settings import CHUNKS_PARQUET, PROCESSED_DIR
from src.llm.gemini_client import GeminiClient

QUALITY_SCORES_PATH = PROCESSED_DIR / "chunk_quality_scores.json"

SCORING_SYSTEM = """You are a content quality evaluator for podcast transcript chunks.
Score each chunk on two dimensions. Return ONLY valid JSON."""

SCORING_PROMPT = """Rate the following podcast transcript chunk on two quality dimensions.

Chunk (Episode #{episode_id}, Guest: {guest}):
---
{text}
---

Score on a scale of 1-5 for each:

1. **Integrity** (1-5): Clarity and lack of transcription errors.
   - 5: Clear, well-formed sentences, proper speaker attribution
   - 3: Mostly readable but some garbled text or missing context
   - 1: Heavily corrupted, unintelligible, or just timestamps/filler

2. **Information Density** (1-5): Ratio of meaningful content to filler.
   - 5: Dense with specific facts, opinions, technical details
   - 3: Mix of substantive and filler content
   - 1: Mostly ads, sponsors, intros, outros, or small talk with no substance

Return JSON:
{{
  "integrity": <1-5>,
  "information_density": <1-5>,
  "is_ad_or_filler": <true/false>,
  "brief_reason": "one sentence explanation"
}}"""


async def score_single_chunk(
    client: GeminiClient,
    chunk_id: int,
    episode_id: int,
    guest: str,
    text: str,
) -> dict | None:
    """Score a single chunk for quality."""
    prompt = SCORING_PROMPT.format(
        episode_id=episode_id,
        guest=guest,
        text=text[:1500],  # Truncate very long chunks
    )

    try:
        response = await client.generate_json(
            prompt=prompt,
            system_instruction=SCORING_SYSTEM,
            temperature=0.2,
        )
        data = json.loads(response)
        data["chunk_id"] = chunk_id
        return data
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "integrity": 3,
            "information_density": 3,
            "is_ad_or_filler": False,
            "brief_reason": f"Scoring failed: {e}",
        }


async def score_all_chunks(
    chunks_path: str | None = None,
    output_path: str | None = None,
    batch_size: int = 10,
) -> dict[int, dict]:
    """
    Score all chunks for quality. Implements checkpointing.

    Returns dict mapping chunk_id -> quality scores.
    """
    chunks_path = chunks_path or str(CHUNKS_PARQUET)
    output_path = Path(output_path or QUALITY_SCORES_PATH)

    df = pd.read_parquet(chunks_path)
    client = GeminiClient()

    # Load checkpoint if exists
    scores: dict[int, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            saved = json.load(f)
            scores = {int(k): v for k, v in saved.items()}
        print(f"Loaded {len(scores)} existing scores from checkpoint")

    # Find unscored chunks
    unscored = df[~df["chunk_id"].isin(scores.keys())]
    print(f"Scoring {len(unscored)} chunks ({len(scores)} already done)...")

    for i in tqdm(range(0, len(unscored), batch_size), desc="Scoring chunks"):
        batch = unscored.iloc[i : i + batch_size]
        tasks = [
            score_single_chunk(
                client=client,
                chunk_id=int(row["chunk_id"]),
                episode_id=int(row["episode_id"]),
                guest=str(row["guest"]),
                text=str(row["text"]),
            )
            for _, row in batch.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                scores[result["chunk_id"]] = result

        # Checkpoint every 100 chunks
        if (i // batch_size) % 10 == 0 and scores:
            with open(output_path, "w") as f:
                json.dump(scores, f)

        await asyncio.sleep(0.5)  # Rate limiting

    # Final save
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    client.print_stats()
    print(f"Scored {len(scores)} chunks. Saved to {output_path}")
    return scores


def load_quality_scores(path: str | None = None) -> dict[int, dict]:
    """Load precomputed quality scores."""
    path = Path(path or QUALITY_SCORES_PATH)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def compute_chunk_quality(score: dict) -> float:
    """
    Compute a single quality score from integrity and density ratings.

    Returns a value between 0 and 1.
    """
    integrity = score.get("integrity", 3)
    density = score.get("information_density", 3)
    is_ad = score.get("is_ad_or_filler", False)

    if is_ad:
        return 0.1  # Heavily penalize ads/filler

    # Weighted combination: density matters more for RAG
    combined = (0.4 * integrity + 0.6 * density) / 5.0
    return round(combined, 3)


def reweight_results(
    results: list[dict],
    quality_scores: dict[int, dict],
    quality_weight: float = 0.3,
) -> list[dict]:
    """
    Re-weight retrieval results using chunk quality scores.

    New score = (1 - quality_weight) * original_score + quality_weight * quality_score
    """
    for r in results:
        chunk_id = r.get("chunk_id", -1)
        q_score = quality_scores.get(chunk_id, {})
        quality = compute_chunk_quality(q_score)

        original = r.get("score", 0.0)
        # Normalize original score to 0-1 range (approximate)
        if original > 1:
            original = min(original / 30.0, 1.0)  # BM25 scores can be large

        r["quality_score"] = quality
        r["original_score"] = r["score"]
        r["score"] = (1 - quality_weight) * original + quality_weight * quality

    # Re-sort by new combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results