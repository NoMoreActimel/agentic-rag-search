"""Evaluation metrics for agentic RAG experiment runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from src.llm.gemini_client import GeminiClient

EVAL_SYSTEM = """You are a strict evaluator for question answering over retrieved transcript chunks.
Return ONLY valid JSON."""

EVAL_PROMPT = """Evaluate this QA attempt.

QUESTION:
{question}

GROUND_TRUTH_ANSWER:
{ground_truth}

MODEL_ANSWER:
{predicted_answer}

REFERENCE_EPISODES:
{reference_episodes}

RETRIEVED_CHUNKS:
{retrieved_chunks}

Return JSON with this schema:
{{
  "judge_score": <integer 1-5>,
  "is_correct": <true/false>,
  "retrieval_precision": <float 0-1>,
  "hallucination_rate": <float 0-1>,
  "reason": "<short rationale>"
}}

Definitions:
- judge_score: overall answer quality against ground truth (5 best).
- is_correct: true only if answer is substantially correct.
- retrieval_precision: fraction of retrieved chunks that are relevant for solving this question.
- hallucination_rate: fraction of answer content not supported by retrieved chunks.
"""


@dataclass
class ExampleMetrics:
    """Single-example metrics emitted by evaluator."""

    judge_score: int
    is_correct: bool
    retrieval_precision: float
    hallucination_rate: float
    lexical_f1: float
    exact_match: bool
    success: bool
    reason: str
    reference_episode_recall: float

    def to_dict(self) -> dict:
        return {
            "judge_score": self.judge_score,
            "is_correct": self.is_correct,
            "retrieval_precision": self.retrieval_precision,
            "hallucination_rate": self.hallucination_rate,
            "lexical_f1": self.lexical_f1,
            "exact_match": self.exact_match,
            "success": self.success,
            "reason": self.reason,
            "reference_episode_recall": self.reference_episode_recall,
        }


def _normalize_text(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _normalize_text(gold).split()
    pred_tokens = _normalize_text(pred).split()
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((gold_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(gold: str, pred: str) -> bool:
    return _normalize_text(gold) == _normalize_text(pred)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _reference_episode_recall(reference_episodes: list[int], found_episodes: list[int]) -> float:
    if not reference_episodes:
        return 0.0
    ref = set(reference_episodes)
    found = set(found_episodes)
    return len(ref.intersection(found)) / len(ref)


def _format_retrieved_chunks(trajectory: dict, max_chunks: int = 15, max_chars_per_chunk: int = 260) -> str:
    entries: list[str] = []
    chunk_counter = 0
    for step in trajectory.get("steps", []):
        for chunk in step.get("results", []):
            if chunk_counter >= max_chunks:
                break
            text = str(chunk.get("text", ""))[:max_chars_per_chunk]
            entries.append(
                f"- chunk_id={chunk.get('chunk_id')} ep={chunk.get('episode_id')} "
                f"score={chunk.get('score')}: {text}"
            )
            chunk_counter += 1
        if chunk_counter >= max_chunks:
            break
    if not entries:
        return "(no chunks retrieved)"
    return "\n".join(entries)


async def evaluate_example_metrics(
    client: GeminiClient,
    question: str,
    ground_truth: str,
    predicted_answer: str,
    reference_episodes: list[int],
    trajectory: dict,
) -> ExampleMetrics:
    """Compute LLM-judge metrics + lexical sanity checks for one example."""
    prompt = EVAL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
        reference_episodes=reference_episodes,
        retrieved_chunks=_format_retrieved_chunks(trajectory),
    )

    judge_score = 1
    is_correct = False
    retrieval_precision = 0.0
    hallucination_rate = 1.0
    reason = "Fallback values used due to evaluator failure."

    try:
        raw = await client.generate_json(
            prompt=prompt,
            system_instruction=EVAL_SYSTEM,
            temperature=0.1,
        )
        parsed = json.loads(raw)
        judge_score = int(parsed.get("judge_score", 1))
        judge_score = min(5, max(1, judge_score))
        is_correct = bool(parsed.get("is_correct", False))
        retrieval_precision = _clip01(float(parsed.get("retrieval_precision", 0.0)))
        hallucination_rate = _clip01(float(parsed.get("hallucination_rate", 1.0)))
        reason = str(parsed.get("reason", "")).strip() or reason
    except Exception as exc:
        reason = f"Evaluator failed: {exc}"

    lexical_f1 = _token_f1(ground_truth, predicted_answer)
    exact_match = _exact_match(ground_truth, predicted_answer)
    success = is_correct or lexical_f1 >= 0.55
    episode_recall = _reference_episode_recall(
        reference_episodes=reference_episodes,
        found_episodes=trajectory.get("unique_episodes_found", []),
    )

    return ExampleMetrics(
        judge_score=judge_score,
        is_correct=is_correct,
        retrieval_precision=retrieval_precision,
        hallucination_rate=hallucination_rate,
        lexical_f1=lexical_f1,
        exact_match=exact_match,
        success=success,
        reason=reason,
        reference_episode_recall=episode_recall,
    )
