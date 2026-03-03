"""Base class and orchestrator for Q&A pair generation."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from config.settings import METADATA_DIR, QA_PAIRS_JSON, QA_PAIRS_PER_TYPE
from src.llm.gemini_client import GeminiClient


@dataclass
class QAPair:
    """A single question-answer pair with metadata."""

    question: str
    answer: str
    qa_type: str
    reference_episodes: list[int]
    reasoning_steps: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class QATypeGenerator(ABC):
    """Abstract base class for Q&A type generators."""

    qa_type: str = ""

    def __init__(self, client: GeminiClient, metadata: list[dict]):
        self.client = client
        self.metadata = metadata
        # Build episode lookup
        self.episodes = {m["episode_id"]: m for m in metadata}

    @abstractmethod
    async def generate(self, count: int = QA_PAIRS_PER_TYPE) -> list[QAPair]:
        """Generate Q&A pairs of this type."""
        ...

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
) -> list[QAPair]:
    """
    Orchestrate Q&A generation across all 4 types.

    Generates QA_PAIRS_PER_TYPE pairs for each type.
    """
    from src.qa.type1_multihop import MultiHopGenerator
    from src.qa.type2_comparative import ComparativeGenerator
    from src.qa.type3_temporal import TemporalGenerator
    from src.qa.type4_aggregation import AggregationGenerator

    output_path = output_path or QA_PAIRS_JSON

    if metadata is None:
        metadata = load_all_metadata()

    if not metadata:
        print("No metadata found. Run 02_extract_metadata.py first.")
        return []

    client = GeminiClient()
    generators = [
        MultiHopGenerator(client, metadata),
        ComparativeGenerator(client, metadata),
        TemporalGenerator(client, metadata),
        AggregationGenerator(client, metadata),
    ]

    all_pairs: list[QAPair] = []
    for gen in generators:
        print(f"\nGenerating {gen.qa_type} questions...")
        pairs = await gen.generate(count=QA_PAIRS_PER_TYPE)
        all_pairs.extend(pairs)
        print(f"  Generated {len(pairs)} {gen.qa_type} pairs")

    # Save
    output = [p.to_dict() for p in all_pairs]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_pairs)} Q&A pairs to {output_path}")

    client.print_stats()
    return all_pairs
