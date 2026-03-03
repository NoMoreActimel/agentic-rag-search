"""Type 3: Temporal Evolution questions — track how views change over time."""

import json

from src.llm.gemini_client import GeminiClient
from src.qa.qa_generator import QAPair, QATypeGenerator

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create temporal evolution questions that track how a guest's or topic's perspective
has evolved across multiple podcast appearances or discussions over time.
Focus on genuine shifts in thinking, updated predictions, or revised views."""

GENERATION_PROMPT = """Create a temporal evolution question based on these episodes.

{episode_descriptions}

The connection: {connection_description}

Create a question that:
1. Asks about how a view, prediction, or stance has evolved over time
2. Requires retrieving from multiple time periods
3. Has an answer that traces the evolution

Return JSON:
{{
  "question": "The temporal evolution question",
  "answer": "A detailed answer tracing how the view evolved across episodes",
  "reasoning_steps": [
    "Step 1: Find earlier view/prediction",
    "Step 2: Find later updated view",
    "Step 3: Describe the evolution"
  ]
}}"""


class TemporalGenerator(QATypeGenerator):
    qa_type = "temporal"

    def _find_returning_guests(self) -> list[tuple[str, list[int]]]:
        """Find guests who appeared in multiple episodes."""
        guest_eps: dict[str, list[int]] = {}
        for m in self.metadata:
            guest = m.get("guest_info", {}).get("name", "").strip()
            if guest:
                guest_key = guest.lower()
                if guest_key not in guest_eps:
                    guest_eps[guest_key] = []
                guest_eps[guest_key].append(m["episode_id"])

        returning = [
            (guest, sorted(eps))
            for guest, eps in guest_eps.items()
            if len(eps) >= 2
        ]
        returning.sort(key=lambda x: len(x[1]), reverse=True)
        return returning

    def _find_evolving_topics(self) -> list[tuple[str, list[int]]]:
        """Find topics discussed across time-separated episodes."""
        topic_eps = self._get_episodes_by_topic()

        evolving = []
        for topic, ep_ids in topic_eps.items():
            unique_eps = sorted(set(ep_ids))
            if len(unique_eps) >= 2:
                # Check if episodes are reasonably spread apart (by ID as proxy for time)
                spread = unique_eps[-1] - unique_eps[0]
                if spread >= 50:  # At least ~50 episodes apart
                    evolving.append((topic, unique_eps))

        evolving.sort(key=lambda x: x[1][-1] - x[1][0], reverse=True)
        return evolving

    async def generate(self, count: int = 5) -> list[QAPair]:
        results = []

        # Try returning guests first
        returning_guests = self._find_returning_guests()
        for guest, ep_ids in returning_guests:
            if len(results) >= count:
                break

            episode_descs = []
            for eid in ep_ids[:3]:
                ep = self.episodes.get(eid)
                if not ep:
                    continue
                summary = ep.get("summary", "No summary")
                topics = ", ".join(ep.get("topics", [])[:5])
                episode_descs.append(
                    f"Episode #{eid}:\n  Summary: {summary}\n  Topics: {topics}"
                )

            if len(episode_descs) < 2:
                continue

            prompt = GENERATION_PROMPT.format(
                episode_descriptions="\n\n".join(episode_descs),
                connection_description=f"Guest '{guest}' appeared in episodes: {ep_ids}",
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
                    reference_episodes=ep_ids[:3],
                    reasoning_steps=data.get("reasoning_steps", []),
                ))
            except Exception as e:
                print(f"  Failed to generate temporal Q&A for guest '{guest}': {e}")

        # Fill remaining with evolving topics
        if len(results) < count:
            evolving_topics = self._find_evolving_topics()
            for topic, ep_ids in evolving_topics:
                if len(results) >= count:
                    break

                selected_eps = [ep_ids[0], ep_ids[-1]]  # First and last
                episode_descs = []
                for eid in selected_eps:
                    ep = self.episodes.get(eid)
                    if not ep:
                        continue
                    guest = ep.get("guest_info", {}).get("name", "Unknown")
                    summary = ep.get("summary", "No summary")
                    episode_descs.append(
                        f"Episode #{eid} with {guest}:\n  Summary: {summary}"
                    )

                if len(episode_descs) < 2:
                    continue

                prompt = GENERATION_PROMPT.format(
                    episode_descriptions="\n\n".join(episode_descs),
                    connection_description=f"Topic '{topic}' discussed across episodes {selected_eps}",
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
                        reference_episodes=selected_eps,
                        reasoning_steps=data.get("reasoning_steps", []),
                    ))
                except Exception as e:
                    print(f"  Failed to generate temporal Q&A for topic '{topic}': {e}")

        return results[:count]
