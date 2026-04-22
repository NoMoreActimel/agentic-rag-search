"""Agentic search loop: iteratively retrieves, reasons, and answers questions.

The agent takes a question, decides which retrieval tool to call and with what
query, accumulates retrieved chunks in context, and decides when it has enough
information to generate a final answer. Optionally receives judge feedback
after each retrieval step.
"""

import json
import time
from dataclasses import dataclass, field

from src.llm.gemini_client import GeminiClient
from src.tools.retrieval_tools import RetrievalResult, ToolRegistry


@dataclass
class AgentStep:
    """Record of a single agent retrieval step."""

    step_number: int
    tool_used: str
    query: str
    num_results: int
    results: list[dict]
    judge_feedback: str | None = None
    timestamp: float = 0.0


@dataclass
class AgentTrajectory:
    """Full trajectory of an agent's search session."""

    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_chunks_retrieved: int = 0
    unique_episodes_found: set = field(default_factory=set)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "steps": [
                {
                    "step_number": s.step_number,
                    "tool_used": s.tool_used,
                    "query": s.query,
                    "num_results": s.num_results,
                    "results": s.results,
                    "judge_feedback": s.judge_feedback,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "unique_episodes_found": list(self.unique_episodes_found),
            "num_steps": len(self.steps),
            "elapsed_seconds": self.elapsed_seconds,
        }


# ── System prompt for the agent ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a research agent that answers questions about the Lex Fridman Podcast.
You have access to a search tool that retrieves relevant transcript chunks from the podcast corpus.

Your task: iteratively search the corpus to find all information needed to answer the question accurately.

AVAILABLE TOOL: {tool_name}
{tool_description}

At each step, you must decide:
1. SEARCH: Issue a new search query to retrieve more information.
2. ANSWER: You have enough information to provide a final answer.

Respond in JSON format:
{{
  "action": "search" or "answer",
  "query": "your search query (only if action=search)",
  "reasoning": "brief explanation of why you chose this action",
  "answer": "your final answer (only if action=answer)"
}}

GUIDELINES:
- Use targeted, specific search queries — not the full question.
- Vary your queries across steps to find different aspects of the answer.
- Pay attention to which episodes and guests appear in results.
- If you receive judge feedback, incorporate it into your next search strategy.
- When answering, base your response ONLY on retrieved information — do not use prior knowledge.
- Provide a direct, concise answer in 2-3 sentences."""


AGENT_STEP_PROMPT = """QUESTION: {question}

{history}=== STEP {step_number} of {max_steps} (current) ===
What is your next action?

Respond in JSON:
{{
  "action": "search" or "answer",
  "query": "search query (if action=search)",
  "reasoning": "why this action",
  "answer": "final answer (if action=answer)"
}}"""

# ── Separate system prompt for forced answers (plain text, no JSON) ──────────

FORCE_ANSWER_SYSTEM = """You are a research agent that answers questions about the Lex Fridman Podcast.
Based ONLY on the retrieved transcript chunks provided, give a direct answer in 2-3 sentences.
Do NOT use any prior knowledge. If the context is insufficient, say so.
Respond in plain text only — no JSON, no formatting."""


class SearchAgent:
    """Iterative search agent that uses retrieval tools to answer questions."""

    def __init__(
        self,
        client: GeminiClient,
        tool_registry: ToolRegistry,
        tool_name: str = "bm25",
        max_steps: int = 5,
        top_k_per_step: int = 5,
        judge=None,
    ):
        """
        Args:
            client: Gemini client for agent reasoning.
            tool_registry: Registry of available retrieval tools.
            tool_name: Which retrieval tool to use ("grep", "bm25", "embedding").
            max_steps: Maximum number of retrieval iterations.
            top_k_per_step: Number of chunks to retrieve per search.
            judge: Optional judge instance for process feedback.
        """
        self.client = client
        self.tools = tool_registry
        self.tool_name = tool_name
        self.max_steps = max_steps
        self.top_k = top_k_per_step
        self.judge = judge

    async def run(self, question: str) -> AgentTrajectory:
        """
        Run the full agentic search loop for a question.

        Returns an AgentTrajectory recording all steps and the final answer.
        """
        start_time = time.time()
        trajectory = AgentTrajectory(question=question)

        # Build system prompt with tool info
        tool_desc = ToolRegistry.tool_descriptions().get(self.tool_name, "")
        system_prompt = AGENT_SYSTEM_PROMPT.format(
            tool_name=self.tool_name,
            tool_description=tool_desc,
        )

        # Accumulated context from all retrieval steps
        all_retrieved: list[RetrievalResult] = []
        seen_chunk_ids: set[int] = set()

        for step_num in range(1, self.max_steps + 1):
            # Per-step history: chunks + judge feedback grouped by the step that produced them.
            history_str = self._build_history_string(trajectory.steps)

            # Ask agent for next action
            step_prompt = AGENT_STEP_PROMPT.format(
                question=question,
                history=history_str,
                step_number=step_num,
                max_steps=self.max_steps,
            )

            try:
                response = await self.client.generate_json(
                    prompt=step_prompt,
                    system_instruction=system_prompt,
                    temperature=0.3,
                )
                decision = json.loads(response)
            except (json.JSONDecodeError, Exception) as e:
                # If parsing fails, force a plain-text answer with what we have
                context_str = self._build_context_string(all_retrieved)
                trajectory.final_answer = await self._force_answer(question, context_str)
                trajectory.steps.append(AgentStep(
                    step_number=step_num,
                    tool_used="none",
                    query="",
                    num_results=0,
                    results=[],
                    timestamp=time.time() - start_time,
                ))
                break

            action = decision.get("action", "answer")

            if action == "search":
                query = decision.get("query", question)

                # Execute retrieval
                results = await self.tools.search(
                    tool_name=self.tool_name,
                    query=query,
                    top_k=self.top_k,
                )

                # Deduplicate against previously seen chunks
                new_results = []
                for r in results:
                    if r.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(r.chunk_id)
                        new_results.append(r)
                        trajectory.unique_episodes_found.add(r.episode_id)

                all_retrieved.extend(new_results)
                trajectory.total_chunks_retrieved += len(new_results)

                # Record step
                step = AgentStep(
                    step_number=step_num,
                    tool_used=self.tool_name,
                    query=query,
                    num_results=len(new_results),
                    results=[r.to_dict() for r in new_results],
                    timestamp=time.time() - start_time,
                )

                # Get judge feedback if judge is configured
                if self.judge is not None and new_results:
                    feedback = await self.judge.evaluate_step(
                        question=question,
                        query=query,
                        retrieved_chunks=new_results,
                        step_number=step_num,
                    )
                    step.judge_feedback = feedback

                trajectory.steps.append(step)

            elif action == "answer":
                # Extract just the answer text
                answer_text = decision.get("answer", "")
                if not answer_text:
                    # Fallback: force a plain-text answer
                    context_str = self._build_context_string(all_retrieved)
                    answer_text = await self._force_answer(question, context_str)

                trajectory.final_answer = answer_text

                # Record a terminal step
                trajectory.steps.append(AgentStep(
                    step_number=step_num,
                    tool_used="none",
                    query="",
                    num_results=0,
                    results=[],
                    timestamp=time.time() - start_time,
                ))
                break

        # If agent used all steps without answering, force an answer
        if not trajectory.final_answer:
            context_str = self._build_context_string(all_retrieved)
            trajectory.final_answer = await self._force_answer(question, context_str)

        trajectory.elapsed_seconds = time.time() - start_time
        return trajectory

    def _build_context_string(self, results: list[RetrievalResult]) -> str:
        """Build a formatted context string from all retrieved chunks."""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"--- Retrieved Chunk {i} ---\n{r.to_context_string()}")
        return "\n\n".join(parts)

    def _build_history_string(self, steps: list[AgentStep]) -> str:
        """Render prior steps as labeled blocks: query + new chunks + judge feedback per step."""
        blocks: list[str] = []
        for s in steps:
            if s.tool_used == "none":
                continue
            chunks_block = self._render_step_chunks(s.results)
            block = (
                f"=== STEP {s.step_number} ===\n"
                f"Query: {s.query}\n"
                f"Retrieved chunks:\n{chunks_block}"
            )
            if s.judge_feedback:
                block += f"\n\nJudge feedback (step {s.step_number}): {s.judge_feedback}"
            blocks.append(block)
        if not blocks:
            return ""
        return "\n\n".join(blocks) + "\n\n"

    @staticmethod
    def _render_step_chunks(chunk_dicts: list[dict]) -> str:
        if not chunk_dicts:
            return "(no new chunks retrieved)"
        parts: list[str] = []
        for i, c in enumerate(chunk_dicts, 1):
            meta = (
                f"[Chunk {c.get('chunk_id')} | Episode #{c.get('episode_id')} | "
                f"Guest: {c.get('guest')} | Score: {float(c.get('score', 0.0)):.3f} | "
                f"Via: {c.get('source')}]"
            )
            parts.append(f"--- Chunk {i} ---\n{meta}\n{c.get('text', '')}")
        return "\n\n".join(parts)

    async def _force_answer(self, question: str, context: str) -> str:
        """Force the agent to produce a plain-text answer from current context."""
        force_prompt = (
            f"QUESTION: {question}\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"You have used all available search steps. "
            f"Based ONLY on the retrieved context above, provide your best answer "
            f"in 2-3 sentences. If the context is insufficient, say so."
        )
        try:
            response = await self.client.generate(
                prompt=force_prompt,
                system_instruction=FORCE_ANSWER_SYSTEM,
                temperature=0.3,
            )
            return response.strip()
        except Exception:
            return "Unable to generate answer due to an error."