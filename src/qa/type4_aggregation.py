"""Type 4: Quantitative Aggregation questions — require exhaustive retrieval of numeric claims."""

import json

from src.llm.gemini_client import GeminiClient
from src.qa.qa_generator import QAPair, QATypeGenerator

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create quantitative aggregation questions that require collecting and comparing
numerical claims, predictions, or metrics from across multiple podcast episodes.
Questions should require exhaustive search — missing one relevant episode would give an incomplete answer.
Focus on concrete numbers: probabilities, timelines, dollar amounts, statistics, rankings."""

GENERATION_PROMPT = """Create a quantitative aggregation question based on these podcast episodes.

Episodes with relevant quantitative details:
{episode_descriptions}

Topic area: {topic}

Create a question that:
1. Requires collecting specific numbers/predictions/timelines from multiple episodes
2. Asks for comparison, ranking, or aggregation of these numbers
3. Would give an incomplete answer if any episode were missed

Return JSON:
{{
  "question": "The quantitative aggregation question",
  "answer": "A detailed answer aggregating the quantitative claims from each episode",
  "reasoning_steps": [
    "Step 1: Retrieve quantitative claims from episode X",
    "Step 2: Retrieve quantitative claims from episode Y",
    "Step 3: Aggregate/compare the numbers"
  ]
}}"""


class AggregationGenerator(QATypeGenerator):
    qa_type = "aggregation"

    def _find_quantitative_clusters(self) -> list[tuple[str, list[tuple[int, str]]]]:
        """Find clusters of episodes with related quantitative claims."""
        # Collect all key_details that contain numbers
        topic_details: dict[str, list[tuple[int, str]]] = {}

        for m in self.metadata:
            ep_id = m["episode_id"]
            for detail in m.get("key_details", []):
                fact = detail.get("fact", "")
                topic = detail.get("topic", "general")

                # Check if the fact contains numbers or quantitative language
                has_numbers = any(c.isdigit() for c in fact)
                quant_words = ["percent", "%", "billion", "million", "thousand",
                               "probability", "timeline", "years", "months",
                               "prediction", "estimate", "forecast"]
                has_quant = any(w in fact.lower() for w in quant_words)

                if has_numbers or has_quant:
                    topic_key = topic.lower().strip()
                    if topic_key not in topic_details:
                        topic_details[topic_key] = []
                    topic_details[topic_key].append((ep_id, fact))

        # Filter to topics with details from 2+ episodes
        clusters = []
        for topic, details in topic_details.items():
            unique_eps = set(d[0] for d in details)
            if len(unique_eps) >= 2:
                clusters.append((topic, details))

        clusters.sort(key=lambda x: len(set(d[0] for d in x[1])), reverse=True)
        return clusters

    async def generate(self, count: int = 5) -> list[QAPair]:
        clusters = self._find_quantitative_clusters()
        results = []

        # If not enough quantitative clusters, fall back to topic-based aggregation
        if len(clusters) < count:
            topic_eps = self._get_episodes_by_topic()
            for topic, ep_ids in topic_eps.items():
                unique_eps = sorted(set(ep_ids))
                if len(unique_eps) >= 3:
                    # Create pseudo-cluster from topic
                    details = []
                    for eid in unique_eps[:4]:
                        ep = self.episodes.get(eid)
                        if ep:
                            summary = ep.get("summary", "")
                            details.append((eid, summary[:200]))
                    clusters.append((topic, details))

        for topic, details in clusters:
            if len(results) >= count:
                break

            # Build episode descriptions
            episode_descs = []
            seen_eps = set()
            for ep_id, fact in details[:6]:
                if ep_id in seen_eps:
                    continue
                seen_eps.add(ep_id)

                ep = self.episodes.get(ep_id)
                if not ep:
                    continue
                guest = ep.get("guest_info", {}).get("name", "Unknown")
                episode_descs.append(
                    f"Episode #{ep_id} with {guest}:\n  Detail: {fact}"
                )

            if len(episode_descs) < 2:
                continue

            prompt = GENERATION_PROMPT.format(
                episode_descriptions="\n\n".join(episode_descs),
                topic=topic,
            )

            try:
                response = await self.client.generate_json(
                    prompt=prompt,
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.5,
                )
                data = json.loads(response)
                ref_eps = sorted(seen_eps)
                results.append(QAPair(
                    question=data["question"],
                    answer=data["answer"],
                    qa_type=self.qa_type,
                    reference_episodes=ref_eps,
                    reasoning_steps=data.get("reasoning_steps", []),
                ))
            except Exception as e:
                print(f"  Failed to generate aggregation Q&A for topic '{topic}': {e}")

        return results[:count]
