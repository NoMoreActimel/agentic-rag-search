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

from config.settings import (
    CHUNKS_PARQUET,
    GEMINI_QUALITY_CONCURRENT_LIMIT,
    GEMINI_QUALITY_MODEL,
    GEMINI_QUALITY_RPM_LIMIT,
    PROCESSED_DIR,
)
from src.llm.gemini_client import GeminiClient

QUALITY_SCORES_PATH = PROCESSED_DIR / "chunk_quality_scores.json"

SCORING_SYSTEM = """You are a content quality evaluator for podcast transcript chunks.
Score each chunk on three dimensions. Return ONLY valid JSON."""

SCORING_PROMPT = """Rate the following podcast transcript chunk on three quality dimensions.

This chunk is part of a retrieval index for a cross-episode Q&A benchmark. Questions in the
benchmark ask about: (a) specific named entities (people, projects, technologies) and their
attributes, (b) guest opinions and contrasting stances on shared topics, (c) how a guest's
predictions or claims changed over time, and (d) concrete numerical claims (probabilities,
years, amounts, counts). A chunk is useful if it contains evidence for SOME such question —
not a specific one.

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

3. **Retrieval Signal** (1-5): Does this chunk contain evidence that could answer SOME
   factual question about this corpus? Consider whether ANY of these are present:
     (a) a named entity (person, project, paper, organization, technology) discussed with
         enough context to say what it is or what the speaker thinks of it,
     (b) an explicit opinion, prediction, or stance the guest holds on an identifiable topic,
     (c) a concrete number, date, probability, timeline, or dollar amount tied to a topic,
     (d) a comparative claim ("X is better than Y", "unlike Z, we…").
   - 5: Two or more of (a)-(d) clearly present and attributable to the speaker.
   - 3: Exactly one of (a)-(d), or one present but weakly attributable.
   - 1: None of (a)-(d). Chunk is conversational filler, pleasantries, anecdote without
        a claim, intro/outro, or sponsor read.

For **is_ad_or_filler**, flag true when the chunk is predominantly ONE of:
  - a sponsor/ad read (product pitch, promo code, "this episode is brought to you by…"),
  - an intro or outro (guest announcement, sign-off, "subscribe on YouTube", podcast plug),
  - small talk / pleasantries with no substantive claim (greetings, "how are you", laughter),
  - a cross-reference to another podcast episode or an unrelated recommendation.
Do not flag chunks that contain a substantive claim mixed with some filler.

Return JSON:
{{
  "integrity": <1-5>,
  "information_density": <1-5>,
  "retrieval_signal": <1-5>,
  "is_ad_or_filler": <true/false>,
  "brief_reason": "one sentence citing the specific feature that drove the score"
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
        # Normalize types (model may return numeric fields as strings).
        data["integrity"] = _rating_1_to_5(data.get("integrity", 3))
        data["information_density"] = _rating_1_to_5(data.get("information_density", 3))
        data["retrieval_signal"] = _rating_1_to_5(data.get("retrieval_signal", 3))
        data["is_ad_or_filler"] = _truthy_ad(data.get("is_ad_or_filler", False))
        return data
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "integrity": 3,
            "information_density": 3,
            "retrieval_signal": 3,
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
    client = GeminiClient(
        model=GEMINI_QUALITY_MODEL,
        rpm_limit=GEMINI_QUALITY_RPM_LIMIT,
        concurrent_limit=GEMINI_QUALITY_CONCURRENT_LIMIT,
    )

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

        # No manual sleep here: the GeminiClient rate limiter enforces
        # GEMINI_QUALITY_RPM_LIMIT and handles pacing for us.

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


def _rating_1_to_5(value, default: int = 3) -> int:
    """Coerce model/JSON output to int in [1, 5] (Gemini sometimes returns strings)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        n = int(round(float(value)))
    else:
        s = str(value).strip()
        try:
            n = int(round(float(s)))
        except ValueError:
            return default
    return max(1, min(5, n))


def _truthy_ad(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def compute_chunk_quality(score: dict) -> float:
    """
    Compute a single quality score from integrity, density, and retrieval-signal ratings.

    Returns a value between 0 and 1. Falls back to the 2-dimension blend if
    retrieval_signal is missing (e.g. scores produced before the field was added).
    """
    integrity = _rating_1_to_5(score.get("integrity", 3))
    density = _rating_1_to_5(score.get("information_density", 3))
    is_ad = _truthy_ad(score.get("is_ad_or_filler", False))

    if is_ad:
        return 0.1  # Heavily penalize ads/filler

    if "retrieval_signal" in score:
        retrieval = _rating_1_to_5(score["retrieval_signal"])
        combined = (0.2 * integrity + 0.3 * density + 0.5 * retrieval) / 5.0
    else:
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
        original = r.get("score", 0.0)
        r["original_score"] = original

        q_score = quality_scores.get(chunk_id)
        if q_score is None:
            r["quality_score"] = None
            continue

        quality = compute_chunk_quality(q_score)

        # Normalize original score to 0-1 range (approximate)
        if original > 1:
            original = min(original / 30.0, 1.0)  # BM25 scores can be large

        r["quality_score"] = quality
        r["score"] = (1 - quality_weight) * original + quality_weight * quality

    # Re-sort by new combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results