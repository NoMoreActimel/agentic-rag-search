"""Type 1: Multi-Hop Bridge questions — require entity-linking across episodes."""

import json

from src.llm.gemini_client import GeminiClient
from src.qa.qa_generator import QAPair, QATypeGenerator

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create multi-hop bridge questions that require connecting information from two different podcast episodes.
The question should require the reader to first identify one piece of information from one episode,
then use that to find related information in another episode.
The question should NOT directly name the connecting entity — the reader must discover it through retrieval."""

GENERATION_PROMPT = """Create a multi-hop bridge question based on these two podcast episodes.

Episode A: #{ep_a_id} — "{ep_a_title}" with {ep_a_guest}
Summary: {ep_a_summary}
Key entities: {ep_a_entities}

Episode B: #{ep_b_id} — "{ep_b_title}" with {ep_b_guest}
Summary: {ep_b_summary}
Key entities: {ep_b_entities}

Shared entity/connection: {shared_entity}

Create a question that:
1. Requires finding information about the shared entity in Episode A
2. Then using that to find related information in Episode B
3. Does NOT directly name both guests — the reader must discover the connection

Return JSON:
{{
  "question": "The multi-hop question",
  "answer": "A detailed answer citing both episodes",
  "reasoning_steps": [
    "Step 1: Find X in episode about Y",
    "Step 2: Use X to identify Z in episode about W",
    "Step 3: Combine to answer"
  ]
}}"""


class MultiHopGenerator(QATypeGenerator):
    qa_type = "multihop"

    def _find_bridge_pairs(self) -> list[tuple[int, int, str]]:
        """Find pairs of episodes connected by shared entities."""
        entity_eps = self._get_episodes_by_entity()

        pairs = []
        for entity, ep_ids in entity_eps.items():
            if len(ep_ids) >= 2:
                # Pick the first two distinct episodes
                unique_eps = sorted(set(ep_ids))
                for i in range(len(unique_eps)):
                    for j in range(i + 1, len(unique_eps)):
                        pairs.append((unique_eps[i], unique_eps[j], entity))

        # Also check for guest cross-references
        for m in self.metadata:
            ep_id = m["episode_id"]
            guest_name = m.get("guest_info", {}).get("name", "").lower()
            for other in m.get("other_persons_mentioned", []):
                other_name = other.get("name", "").lower()
                # Check if this mentioned person is a guest in another episode
                for other_m in self.metadata:
                    if other_m["episode_id"] != ep_id:
                        other_guest = other_m.get("guest_info", {}).get("name", "").lower()
                        if other_name and other_guest and other_name in other_guest:
                            pairs.append((ep_id, other_m["episode_id"], other_name))

        # Deduplicate
        seen = set()
        unique_pairs = []
        for a, b, e in pairs:
            key = (min(a, b), max(a, b), e)
            if key not in seen:
                seen.add(key)
                unique_pairs.append((a, b, e))

        return unique_pairs

    async def generate(self, count: int = 5) -> list[QAPair]:
        bridge_pairs = self._find_bridge_pairs()

        if not bridge_pairs:
            print("  Warning: No bridge pairs found for multi-hop questions")
            return []

        # Take up to `count` pairs
        selected = bridge_pairs[:count * 2]  # Generate extras in case some fail
        results = []

        for ep_a_id, ep_b_id, shared in selected:
            if len(results) >= count:
                break

            ep_a = self.episodes.get(ep_a_id)
            ep_b = self.episodes.get(ep_b_id)
            if not ep_a or not ep_b:
                continue

            entities_a = ", ".join(
                e["name"] for e in ep_a.get("main_entities", [])[:5]
            )
            entities_b = ", ".join(
                e["name"] for e in ep_b.get("main_entities", [])[:5]
            )

            prompt = GENERATION_PROMPT.format(
                ep_a_id=ep_a_id,
                ep_a_title=ep_a.get("guest_info", {}).get("name", "Unknown"),
                ep_a_guest=ep_a.get("guest_info", {}).get("name", "Unknown"),
                ep_a_summary=ep_a.get("summary", ""),
                ep_a_entities=entities_a,
                ep_b_id=ep_b_id,
                ep_b_title=ep_b.get("guest_info", {}).get("name", "Unknown"),
                ep_b_guest=ep_b.get("guest_info", {}).get("name", "Unknown"),
                ep_b_summary=ep_b.get("summary", ""),
                ep_b_entities=entities_b,
                shared_entity=shared,
            )

            try:
                response = await self.client.generate_json(
                    prompt=prompt,
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.5,
                )
                data = json.loads(response)
                results.append(QAPair(
                    question=data["question"],
                    answer=data["answer"],
                    qa_type=self.qa_type,
                    reference_episodes=[ep_a_id, ep_b_id],
                    reasoning_steps=data.get("reasoning_steps", []),
                ))
            except Exception as e:
                print(f"  Failed to generate multi-hop Q&A for eps {ep_a_id},{ep_b_id}: {e}")

        return results[:count]
