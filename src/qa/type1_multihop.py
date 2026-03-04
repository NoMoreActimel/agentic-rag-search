"""Type 1: Multi-Hop Bridge questions — require entity-linking across episodes."""

import json

from src.qa.qa_generator import Candidate, QAPair, QATypeGenerator, PROMPT_CONSTRAINTS

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create multi-hop bridge questions that require connecting information across two different episodes.
The question must force the reader to first retrieve one piece of information, then use it to find related information in a different episode."""

GENERATION_PROMPT = """Create a multi-hop bridge question based on these two podcast episodes connected by: {connection}

Episode A:
{episode_a}

Episode B:
{episode_b}

Create a question that:
1. Requires finding information about the connecting element in one episode
2. Then using that to find related information in the other episode
3. References guests/topics/concepts — NOT episode numbers
4. Cannot be answered from a single episode alone

{constraints}

Return JSON:
{{
  "question": "The multi-hop question (no episode numbers!)",
  "answer": "Direct factual answer in 2-3 sentences with brief justification.",
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}"""


class MultiHopGenerator(QATypeGenerator):
    qa_type = "multihop"

    def find_candidates(self, count: int = 10) -> list[Candidate]:
        """Find pairs of episodes connected by shared entities or guest cross-references."""
        pairs = []
        seen_pairs = set()

        # Strategy 1: Guest cross-references (guest A mentions person who is guest B)
        # Require at least 2 words in the mentioned name to avoid "Ryan", "Matt" false matches
        for m in self.metadata:
            ep_id = m["episode_id"]
            for other in m.get("other_persons_mentioned", []):
                other_name = other.get("name", "").strip()
                if not other_name or len(other_name.split()) < 2:
                    continue  # Skip single-word names (too ambiguous)
                other_name_lower = other_name.lower()
                for other_m in self.metadata:
                    if other_m["episode_id"] == ep_id:
                        continue
                    other_guest = self._get_guest_name(other_m).lower()
                    if other_guest and other_name_lower in other_guest:
                        key = (min(ep_id, other_m["episode_id"]),
                               max(ep_id, other_m["episode_id"]))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            guest_a = self._get_guest_name(m)
                            pairs.append(Candidate(
                                qa_type=self.qa_type,
                                episode_ids=[ep_id, other_m["episode_id"]],
                                connection=f"{guest_a} mentions {other['name']}, who is the guest in another episode",
                            ))

        # Strategy 2: Shared entities across different guests
        entity_eps = self._get_episodes_by_entity()
        for entity, ep_ids in entity_eps.items():
            unique_eps = sorted(set(ep_ids))
            if len(unique_eps) < 2:
                continue
            # Only pair episodes with different guests
            for i in range(len(unique_eps)):
                for j in range(i + 1, len(unique_eps)):
                    ep_a, ep_b = unique_eps[i], unique_eps[j]
                    guest_a = self._get_guest_name(self.episodes.get(ep_a, {}))
                    guest_b = self._get_guest_name(self.episodes.get(ep_b, {}))
                    if guest_a.lower() == guest_b.lower():
                        continue  # Same guest — not a bridge
                    key = (ep_a, ep_b)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pairs.append(Candidate(
                            qa_type=self.qa_type,
                            episode_ids=[ep_a, ep_b],
                            connection=f"Shared entity '{entity}' discussed by {guest_a} and {guest_b}",
                        ))

        # Prioritize guest cross-references (stronger bridges), then entity-based
        # Deduplicate by episode pair to ensure diversity
        return pairs[:count]

    async def generate_from_candidate(self, candidate: Candidate) -> QAPair | None:
        ep_a_id, ep_b_id = candidate.episode_ids[:2]

        prompt = GENERATION_PROMPT.format(
            connection=candidate.connection,
            episode_a=self._build_episode_summary(ep_a_id),
            episode_b=self._build_episode_summary(ep_b_id),
            constraints=PROMPT_CONSTRAINTS,
        )

        try:
            response = await self.client.generate_json(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.5,
            )
            data = json.loads(response)
            return QAPair(
                question=data["question"],
                answer=data["answer"],
                qa_type=self.qa_type,
                reference_episodes=[ep_a_id, ep_b_id],
                reasoning_steps=data.get("reasoning_steps", []),
            )
        except Exception as e:
            print(f"    Failed for episodes {ep_a_id},{ep_b_id}: {e}")
            return None
