"""Type 3: Temporal Evolution questions — track how views change over time."""

import json
from functools import lru_cache

import pandas as pd

from config.settings import TRANSCRIPTS_PARQUET
from src.qa.qa_generator import Candidate, QAPair, QATypeGenerator, PROMPT_CONSTRAINTS


@lru_cache(maxsize=1)
def _load_episode_dates() -> dict[int, pd.Timestamp | None]:
    """Map episode_id -> publication date (UTC timestamp or None if undated)."""
    df = pd.read_parquet(TRANSCRIPTS_PARQUET)
    dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return {
        int(e): (d if pd.notna(d) else None)
        for e, d in zip(df["episode_id"], dt)
    }


_MIN_DATE_SPREAD = pd.Timedelta(days=180)   # ≥ 6 months of real time
_MIN_ID_SPREAD_FALLBACK = 50                # used when either endpoint is undated

SYSTEM_INSTRUCTION = """You are a Q&A dataset creator for podcast transcripts.
Create temporal evolution questions that track how a guest's views or a topic's treatment
has evolved across multiple podcast appearances or discussions over time.
Focus on genuine shifts in thinking, updated predictions, or revised views."""

GENERATION_PROMPT = """Create a temporal evolution question based on these episodes.

Connection: {connection}

{episode_descriptions}

Create a question that:
1. Asks about how a view, prediction, or stance has evolved over time
2. Requires retrieving from multiple time periods / appearances
3. References the guest or topic by name — NOT by episode number
4. Focuses on a genuine change or development, not just different facts

{constraints}

Return JSON:
{{
  "question": "The temporal evolution question (no episode numbers!)",
  "answer": "Direct factual answer in 2-3 sentences tracing the evolution.",
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}"""


class TemporalGenerator(QATypeGenerator):
    qa_type = "temporal"

    def find_candidates(self, count: int = 10) -> list[Candidate]:
        """Find returning guests or topics discussed across time-separated episodes."""
        candidates = []
        seen = set()

        # Strategy 1: Returning guests (strongest temporal signal)
        guest_eps: dict[str, list[int]] = {}
        for m in self.metadata:
            guest = self._get_guest_name(m).strip().lower()
            if guest:
                if guest not in guest_eps:
                    guest_eps[guest] = []
                guest_eps[guest].append(m["episode_id"])

        for guest, eps in guest_eps.items():
            if len(eps) >= 2:
                eps_sorted = sorted(eps)
                key = tuple(eps_sorted)
                if key not in seen:
                    seen.add(key)
                    # Get the proper-cased name
                    proper_name = self._get_guest_name(self.episodes.get(eps_sorted[0], {}))
                    candidates.append(Candidate(
                        qa_type=self.qa_type,
                        episode_ids=eps_sorted,
                        connection=f"Returning guest: {proper_name} appeared in {len(eps)} episodes",
                    ))

        # Strategy 2: Same topic discussed in time-separated episodes.
        # Prefer real publication dates (accurate for scraped eps); fall back to
        # episode-ID spread when either endpoint is undated (HF archive).
        ep_dates = _load_episode_dates()
        topic_eps = self._get_episodes_by_topic()
        for topic, ep_ids in topic_eps.items():
            unique_eps = sorted(set(ep_ids))
            if len(unique_eps) < 2:
                continue
            first_ep, last_ep = unique_eps[0], unique_eps[-1]
            d_first, d_last = ep_dates.get(first_ep), ep_dates.get(last_ep)

            if d_first is not None and d_last is not None:
                date_spread = d_last - d_first
                if date_spread < _MIN_DATE_SPREAD:
                    continue
                conn = f"Topic '{topic}' discussed ~{date_spread.days} days apart"
            else:
                id_spread = last_ep - first_ep
                if id_spread < _MIN_ID_SPREAD_FALLBACK:
                    continue
                conn = f"Topic '{topic}' discussed ~{id_spread} episodes apart"

            selected = [first_ep, last_ep]
            key = tuple(selected)
            if key not in seen:
                seen.add(key)
                candidates.append(Candidate(
                    qa_type=self.qa_type,
                    episode_ids=selected,
                    connection=conn,
                ))

        # Prioritize returning guests, then topics with largest real time spread.
        def _rank(c: Candidate) -> tuple[bool, float]:
            is_returning = "Returning guest" in c.connection
            d_min = ep_dates.get(c.episode_ids[0])
            d_max = ep_dates.get(c.episode_ids[-1])
            if d_min is not None and d_max is not None:
                spread_score = (d_max - d_min).days
            else:
                spread_score = c.episode_ids[-1] - c.episode_ids[0]
            return (is_returning, spread_score)

        candidates.sort(key=_rank, reverse=True)
        return candidates[:count]

    async def generate_from_candidate(self, candidate: Candidate) -> QAPair | None:
        episode_descs = []
        for eid in candidate.episode_ids:
            episode_descs.append(self._build_episode_summary(eid))

        prompt = GENERATION_PROMPT.format(
            connection=candidate.connection,
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
            print(f"    Failed for {candidate.connection}: {e}")
            return None
