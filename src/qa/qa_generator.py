"""Base class and orchestrator for Q&A pair generation."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from config.settings import METADATA_DIR, QA_PAIRS_JSON, QA_PAIRS_PER_TYPE
from src.llm.gemini_client import GeminiClient


@dataclass
class Candidate:
    """A group of episodes identified as a potential Q&A source."""

    qa_type: str
    episode_ids: list[int]
    connection: str  # Why these episodes are grouped (shared entity, topic, etc.)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAPair:
    """A single question-answer pair with metadata."""

    question: str
    answer: str
    qa_type: str
    reference_episodes: list[int]
    reasoning_steps: list[str]
    era: str = ""  # "old" (<=2025-01 cutoff) or "new" (post-cutoff); "" if unspecified

    def to_dict(self) -> dict:
        return asdict(self)


# Shared prompt constraints appended to all Q&A generation prompts
PROMPT_CONSTRAINTS = """
STRICT RULES:
- The question must NEVER mention episode numbers, episode IDs, or "#NNN".
- Instead, reference guests by name, topics, concepts, or projects they discussed.
- The answer must be 2-3 sentences max: a direct factual answer followed by brief justification.
- The answer must NOT mention episode numbers either.
- The question should require genuine retrieval — a reader must search the transcript corpus to answer it.
"""


class QATypeGenerator(ABC):
    """Abstract base class for Q&A type generators."""

    qa_type: str = ""

    def __init__(self, client: GeminiClient, metadata: list[dict]):
        self.client = client
        self.metadata = metadata
        self.episodes = {m["episode_id"]: m for m in metadata}

    @abstractmethod
    def find_candidates(self, count: int = QA_PAIRS_PER_TYPE) -> list[Candidate]:
        """Stage 1: Find candidate episode groups from metadata."""
        ...

    @abstractmethod
    async def generate_from_candidate(self, candidate: Candidate) -> QAPair | None:
        """Stage 2: Generate a Q&A pair from a candidate group."""
        ...

    async def generate(self, count: int = QA_PAIRS_PER_TYPE) -> list[QAPair]:
        """Run both stages: find candidates, then generate Q&A for each."""
        # Stage 1: find more candidates than needed (some may fail generation)
        candidates = self.find_candidates(count=count * 2)
        print(f"  Found {len(candidates)} candidates")

        # Stage 2: generate Q&A from each candidate until we have enough
        results = []
        for candidate in candidates:
            if len(results) >= count:
                break
            pair = await self.generate_from_candidate(candidate)
            if pair is not None:
                results.append(pair)

        return results[:count]

    @staticmethod
    def _get_guest_name(episode: dict) -> str:
        """Extract guest name, handling both dict and list guest_info."""
        gi = episode.get("guest_info", {})
        if isinstance(gi, list):
            return ", ".join(g.get("name", "") for g in gi if g.get("name"))
        return gi.get("name", "")

    def _get_episodes_by_entity(self) -> dict[str, list[int]]:
        """Build entity -> episode_id index."""
        entity_eps: dict[str, list[int]] = {}
        for m in self.metadata:
            ep_id = m["episode_id"]
            for entity in m.get("main_entities", []):
                name = entity["name"].lower().strip()
                if name not in entity_eps:
                    entity_eps[name] = []
                entity_eps[name].append(ep_id)
        return entity_eps

    def _get_episodes_by_topic(self) -> dict[str, list[int]]:
        """Build topic -> episode_id index."""
        topic_eps: dict[str, list[int]] = {}
        for m in self.metadata:
            ep_id = m["episode_id"]
            for topic in m.get("topics", []):
                topic_key = topic.lower().strip()
                if topic_key not in topic_eps:
                    topic_eps[topic_key] = []
                topic_eps[topic_key].append(ep_id)
        return topic_eps

    def _build_episode_summary(self, ep_id: int) -> str:
        """Build a text summary of an episode for use in prompts."""
        ep = self.episodes.get(ep_id)
        if not ep:
            return ""
        guest = self._get_guest_name(ep) or "Unknown"
        summary = ep.get("summary", "")
        entities = ", ".join(e["name"] for e in ep.get("main_entities", [])[:8])
        topics = ", ".join(ep.get("topics", [])[:6])
        details = "\n".join(
            f"  - {d['fact']}" for d in ep.get("key_details", [])[:6]
        )
        return (
            f"Guest: {guest}\n"
            f"Summary: {summary}\n"
            f"Key entities: {entities}\n"
            f"Topics: {topics}\n"
            f"Key details:\n{details}"
        )


def load_all_metadata(metadata_dir=None) -> list[dict]:
    """Load all metadata JSON files."""
    metadata_dir = metadata_dir or METADATA_DIR
    metadata = []
    for f in sorted(metadata_dir.glob("*.json")):
        with open(f) as fh:
            metadata.append(json.load(fh))
    return metadata


async def generate_all_qa_pairs(
    metadata: list[dict] | None = None,
    output_path=None,
    count_per_type: int | None = None,
    era: str = "",
    client: GeminiClient | None = None,
    save: bool = True,
) -> list[QAPair]:
    """Orchestrate Q&A generation across all 4 types.

    Args:
        metadata: Episode metadata list (filters the candidate pool). Loaded from
            METADATA_DIR if None.
        output_path: Where to write the JSON. Defaults to QA_PAIRS_JSON.
        count_per_type: Number of pairs per QA type. Defaults to QA_PAIRS_PER_TYPE.
        era: Optional cohort tag stamped on every generated pair (e.g. "old" / "new").
        client: Shared GeminiClient; created if None.
        save: If False, skip writing output_path (caller aggregates and saves).
    """
    from src.qa.type1_multihop import MultiHopGenerator
    from src.qa.type2_comparative import ComparativeGenerator
    from src.qa.type3_temporal import TemporalGenerator
    from src.qa.type4_aggregation import AggregationGenerator

    output_path = output_path or QA_PAIRS_JSON
    count_per_type = count_per_type or QA_PAIRS_PER_TYPE

    if metadata is None:
        metadata = load_all_metadata()

    if not metadata:
        print("No metadata found. Run 02_extract_metadata.py first.")
        return []

    if client is None:
        client = GeminiClient()
    generators = [
        MultiHopGenerator(client, metadata),
        ComparativeGenerator(client, metadata),
        TemporalGenerator(client, metadata),
        AggregationGenerator(client, metadata),
    ]

    all_pairs: list[QAPair] = []
    for gen in generators:
        print(f"\nGenerating {gen.qa_type} questions (era={era or 'unset'}) ...")
        print(f"  Stage 1: Finding candidates...")
        pairs = await gen.generate(count=count_per_type)
        for p in pairs:
            p.era = era
        all_pairs.extend(pairs)
        print(f"  Generated {len(pairs)} {gen.qa_type} pairs")

    if save:
        output = [p.to_dict() for p in all_pairs]
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(all_pairs)} Q&A pairs to {output_path}")
        client.print_stats()
    return all_pairs
