"""Type 2: Comparative Viewpoint questions — compare perspectives across episodes."""

import json

from src.qa.qa_generator import Candidate, QAPair, QATypeGenerator, PROMPT_CONSTRAINTS

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create comparative questions that require synthesizing and contrasting viewpoints
from different podcast guests on the same topic.
Focus on genuine differences in perspective, not just different facts."""

GENERATION_PROMPT = """Create a comparative viewpoint question for these episodes that share the topic "{topic}".

{episode_descriptions}

Create a question that:
1. Asks how different guests view or approach this shared topic
2. Requires retrieving relevant segments from multiple episodes
3. References guests by name or by their ideas — NOT by episode number
4. Highlights a genuine contrast or complementary perspectives

{constraints}

Return JSON:
{{
  "question": "The comparative question (no episode numbers!)",
  "answer": "Direct factual comparison in 2-3 sentences with key contrasts.",
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}"""


class ComparativeGenerator(QATypeGenerator):
    qa_type = "comparative"

    def find_candidates(self, count: int = 10) -> list[Candidate]:
        """Find groups of episodes sharing a topic with different guests."""
        topic_eps = self._get_episodes_by_topic()

        candidates = []
        seen_ep_sets = set()

        # Sort topics by number of episodes (most shared first)
        shared = [
            (topic, sorted(set(eps)))
            for topic, eps in topic_eps.items()
            if len(set(eps)) >= 2
        ]
        shared.sort(key=lambda x: len(x[1]), reverse=True)

        for topic, ep_ids in shared:
            # Pick 2-3 episodes with different guests
            selected = []
            seen_guests = set()
            for eid in ep_ids:
                guest = self._get_guest_name(self.episodes.get(eid, {})).lower()
                if guest and guest not in seen_guests:
                    seen_guests.add(guest)
                    selected.append(eid)
                if len(selected) >= 3:
                    break

            if len(selected) < 2:
                continue

            ep_key = tuple(sorted(selected))
            if ep_key in seen_ep_sets:
                continue
            seen_ep_sets.add(ep_key)

            guest_names = [self._get_guest_name(self.episodes.get(e, {})) for e in selected]
            candidates.append(Candidate(
                qa_type=self.qa_type,
                episode_ids=selected,
                connection=f"Topic '{topic}' discussed by {', '.join(guest_names)}",
            ))

        return candidates[:count]

    async def generate_from_candidate(self, candidate: Candidate) -> QAPair | None:
        topic = candidate.connection.split("'")[1]  # Extract topic from connection string

        episode_descs = []
        for eid in candidate.episode_ids:
            episode_descs.append(self._build_episode_summary(eid))

        prompt = GENERATION_PROMPT.format(
            topic=topic,
            episode_descriptions="\n\n---\n\n".join(episode_descs),
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
                reference_episodes=candidate.episode_ids,
                reasoning_steps=data.get("reasoning_steps", []),
            )
        except Exception as e:
            print(f"    Failed for topic '{topic}': {e}")
            return None
