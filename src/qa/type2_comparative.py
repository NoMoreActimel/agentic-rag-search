"""Type 2: Comparative Viewpoint questions — compare perspectives across episodes."""

import json

from src.llm.gemini_client import GeminiClient
from src.qa.qa_generator import QAPair, QATypeGenerator

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create comparative questions that require synthesizing and contrasting viewpoints
from different podcast guests on the same topic.
Questions should highlight genuine differences in perspective, not just different facts."""

GENERATION_PROMPT = """Create a comparative viewpoint question based on these episodes that share the topic "{topic}".

{episode_descriptions}

Create a question that:
1. Asks about how different guests view or approach this topic
2. Requires retrieving relevant segments from multiple episodes
3. Has an answer that synthesizes the contrasting viewpoints

Return JSON:
{{
  "question": "The comparative question",
  "answer": "A detailed answer comparing the viewpoints from each episode",
  "reasoning_steps": [
    "Step 1: Retrieve Guest A's view on topic",
    "Step 2: Retrieve Guest B's view on topic",
    "Step 3: Compare and contrast"
  ]
}}"""


class ComparativeGenerator(QATypeGenerator):
    qa_type = "comparative"

    def _find_shared_topics(self) -> list[tuple[str, list[int]]]:
        """Find topics discussed across multiple episodes."""
        topic_eps = self._get_episodes_by_topic()

        # Filter to topics with 2+ episodes, sort by number of episodes
        shared = [
            (topic, sorted(set(eps)))
            for topic, eps in topic_eps.items()
            if len(set(eps)) >= 2
        ]
        shared.sort(key=lambda x: len(x[1]), reverse=True)
        return shared

    async def generate(self, count: int = 5) -> list[QAPair]:
        shared_topics = self._find_shared_topics()

        if not shared_topics:
            print("  Warning: No shared topics found for comparative questions")
            return []

        results = []

        for topic, ep_ids in shared_topics:
            if len(results) >= count:
                break

            # Take up to 3 episodes per topic for comparison
            selected_eps = ep_ids[:3]
            episode_descs = []

            for eid in selected_eps:
                ep = self.episodes.get(eid)
                if not ep:
                    continue
                guest = ep.get("guest_info", {}).get("name", "Unknown")
                summary = ep.get("summary", "No summary available")
                episode_descs.append(
                    f"Episode #{eid} with {guest}:\n  Summary: {summary}"
                )

            if len(episode_descs) < 2:
                continue

            prompt = GENERATION_PROMPT.format(
                topic=topic,
                episode_descriptions="\n\n".join(episode_descs),
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
                print(f"  Failed to generate comparative Q&A for topic '{topic}': {e}")

        return results[:count]
