"""Type 4: Quantitative Aggregation questions — require exhaustive retrieval of numeric claims."""

import json

from src.qa.qa_generator import Candidate, QAPair, QATypeGenerator, PROMPT_CONSTRAINTS

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create quantitative aggregation questions that require collecting and comparing
specific numerical claims, predictions, or metrics from multiple podcast episodes.
Questions should require exhaustive search — missing one relevant episode gives an incomplete answer.
Focus on concrete numbers: probabilities, timelines, dollar amounts, statistics."""

GENERATION_PROMPT = """Create a quantitative aggregation question from these episodes.

Topic area: {topic}

{episode_descriptions}

Create a question that:
1. Asks to collect or compare specific numbers/predictions/timelines from multiple guests
2. Requires searching across multiple episodes to find all relevant data points
3. References guests or topics by name — NOT by episode number
4. Would give an incomplete answer if any relevant episode were missed

{constraints}

Return JSON:
{{
  "question": "The quantitative aggregation question (no episode numbers!)",
  "answer": "Direct factual answer in 2-3 sentences aggregating the numbers.",
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}"""


class AggregationGenerator(QATypeGenerator):
    qa_type = "aggregation"

    def find_candidates(self, count: int = 10) -> list[Candidate]:
        """Find clusters of episodes with related quantitative claims."""
        candidates = []
        seen = set()

        # Scan key_details for quantitative content
        topic_details: dict[str, list[tuple[int, str]]] = {}
        quant_words = [
            "percent", "%", "billion", "million", "thousand",
            "probability", "timeline", "years", "months",
            "prediction", "estimate", "forecast", "$",
        ]

        for m in self.metadata:
            ep_id = m["episode_id"]
            for detail in m.get("key_details", []):
                fact = detail.get("fact", "")
                topic = detail.get("topic", "general")
                has_numbers = any(c.isdigit() for c in fact)
                has_quant = any(w in fact.lower() for w in quant_words)
                if has_numbers or has_quant:
                    topic_key = topic.lower().strip()
                    if topic_key not in topic_details:
                        topic_details[topic_key] = []
                    topic_details[topic_key].append((ep_id, fact))

        # Build candidates from topics with quantitative details across 2+ episodes
        for topic, details in topic_details.items():
            unique_eps = sorted(set(d[0] for d in details))
            if len(unique_eps) < 2:
                continue
            key = tuple(unique_eps[:4])
            if key in seen:
                continue
            seen.add(key)

            guest_names = [
                self._get_guest_name(self.episodes.get(e, {}))
                for e in unique_eps[:4]
            ]
            facts_preview = "; ".join(d[1][:80] for d in details[:4])
            candidates.append(Candidate(
                qa_type=self.qa_type,
                episode_ids=unique_eps[:4],
                connection=f"Quantitative claims about '{topic}' by {', '.join(guest_names)}: {facts_preview}",
            ))

        # Fallback: topic-based aggregation if not enough quantitative clusters
        if len(candidates) < count:
            topic_eps = self._get_episodes_by_topic()
            for topic, ep_ids in topic_eps.items():
                unique_eps = sorted(set(ep_ids))
                if len(unique_eps) < 3:
                    continue
                key = tuple(unique_eps[:4])
                if key in seen:
                    continue
                seen.add(key)
                guest_names = [
                    self._get_guest_name(self.episodes.get(e, {}))
                    for e in unique_eps[:4]
                ]
                candidates.append(Candidate(
                    qa_type=self.qa_type,
                    episode_ids=unique_eps[:4],
                    connection=f"Topic '{topic}' discussed by {', '.join(guest_names)}",
                ))

        # Prioritize genuine quantitative clusters
        candidates.sort(key=lambda c: "Quantitative" in c.connection, reverse=True)
        return candidates[:count]

    async def generate_from_candidate(self, candidate: Candidate) -> QAPair | None:
        # Extract topic from connection
        if "'" in candidate.connection:
            topic = candidate.connection.split("'")[1]
        else:
            topic = "general"

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
