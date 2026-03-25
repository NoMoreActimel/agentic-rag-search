"""Judge modules: provide process-level feedback to the search agent.

Three judge configurations:
1. NoJudge — no feedback (baseline)
2. ProcessJudge — reviews retrieved chunks and provides feedback
3. OracleJudge — same as ProcessJudge but with ground-truth answer (ceiling)
"""

from abc import ABC, abstractmethod

from src.llm.gemini_client import GeminiClient
from src.tools.retrieval_tools import RetrievalResult


class BaseJudge(ABC):
    """Abstract base class for judges."""

    @abstractmethod
    async def evaluate_step(
        self,
        question: str,
        query: str,
        retrieved_chunks: list[RetrievalResult],
        step_number: int,
    ) -> str | None:
        """Evaluate a retrieval step and return feedback for the agent."""
        ...


class NoJudge(BaseJudge):
    """No feedback — baseline configuration."""

    async def evaluate_step(self, **kwargs) -> str | None:
        return None


# ── Process Judge ─────────────────────────────────────────────────────────────

PROCESS_JUDGE_SYSTEM = """You are a retrieval quality judge for a podcast search system.
Your job is to evaluate retrieved transcript chunks and provide actionable feedback
to help the search agent find better information.

Be specific and concise. Your feedback directly guides the agent's next search query."""

PROCESS_JUDGE_PROMPT = """QUESTION being investigated:
{question}

SEARCH QUERY used in this step:
{query}

RETRIEVED CHUNKS (Step {step_number}):
{chunks_text}

Evaluate these chunks and provide feedback in 2-3 sentences:
1. Are any chunks clearly irrelevant (ads, filler, off-topic)? If so, say which to ignore.
2. Do the chunks contain useful information toward answering the question?
3. What should the agent search for next to fill gaps in the information?

Be direct and actionable. Focus on guiding the next search query."""


class ProcessJudge(BaseJudge):
    """Judge-in-Process: reviews chunks and provides natural language feedback."""

    def __init__(self, client: GeminiClient):
        self.client = client

    async def evaluate_step(
        self,
        question: str,
        query: str,
        retrieved_chunks: list[RetrievalResult],
        step_number: int,
    ) -> str | None:
        if not retrieved_chunks:
            return "No chunks were retrieved. Try a different or broader search query."

        chunks_text = "\n\n".join(
            f"Chunk {i+1} [Ep#{r.episode_id}, Guest: {r.guest}, Score: {r.score:.3f}]:\n{r.text[:500]}"
            for i, r in enumerate(retrieved_chunks)
        )

        prompt = PROCESS_JUDGE_PROMPT.format(
            question=question,
            query=query,
            step_number=step_number,
            chunks_text=chunks_text,
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_instruction=PROCESS_JUDGE_SYSTEM,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            return f"Judge error: {e}"


# ── Oracle Judge ──────────────────────────────────────────────────────────────

ORACLE_JUDGE_SYSTEM = """You are an oracle retrieval quality judge for a podcast search system.
You have access to the ground-truth answer and can provide highly targeted feedback
to help the search agent find the exact information needed.

Be specific and concise. Your feedback directly guides the agent's next search query.
Do NOT reveal the answer directly — instead, guide the agent toward the right chunks."""

ORACLE_JUDGE_PROMPT = """QUESTION being investigated:
{question}

GROUND-TRUTH ANSWER (for your reference only — do NOT reveal this to the agent):
{ground_truth}

SEARCH QUERY used in this step:
{query}

RETRIEVED CHUNKS (Step {step_number}):
{chunks_text}

Evaluate these chunks against the ground truth and provide feedback in 2-3 sentences:
1. Are any chunks clearly irrelevant or misleading? If so, say which to ignore.
2. How close are the retrieved chunks to containing the needed information?
3. What specific topic, name, or concept should the agent search for next?

Be direct and actionable. Guide toward the right information WITHOUT revealing the answer."""


class OracleJudge(BaseJudge):
    """Oracle Judge: has ground-truth answer for maximum feedback quality."""

    def __init__(self, client: GeminiClient, ground_truth: str):
        self.client = client
        self.ground_truth = ground_truth

    async def evaluate_step(
        self,
        question: str,
        query: str,
        retrieved_chunks: list[RetrievalResult],
        step_number: int,
    ) -> str | None:
        if not retrieved_chunks:
            return "No chunks were retrieved. Try a different or broader search query."

        chunks_text = "\n\n".join(
            f"Chunk {i+1} [Ep#{r.episode_id}, Guest: {r.guest}, Score: {r.score:.3f}]:\n{r.text[:500]}"
            for i, r in enumerate(retrieved_chunks)
        )

        prompt = ORACLE_JUDGE_PROMPT.format(
            question=question,
            ground_truth=self.ground_truth,
            query=query,
            step_number=step_number,
            chunks_text=chunks_text,
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_instruction=ORACLE_JUDGE_SYSTEM,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            return f"Oracle judge error: {e}"