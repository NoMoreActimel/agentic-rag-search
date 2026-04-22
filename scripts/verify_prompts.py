#!/usr/bin/env python3
"""Verify that judge feedback + previously retrieved chunks appear in agent prompts.

Uses a mock Gemini client so no API calls are made. Prints the exact prompt that
would be sent on iterations 1, 2, 3, and the judge prompt that would be sent in
between steps.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.search_agent import SearchAgent
from src.judge.judges import PROCESS_JUDGE_PROMPT, PROCESS_JUDGE_SYSTEM, ProcessJudge
from src.tools.retrieval_tools import RetrievalResult


BAR = "=" * 90
SUB = "-" * 90


class MockClient:
    """Stand-in for GeminiClient that records prompts and returns canned responses."""

    def __init__(self):
        self.agent_calls: list[tuple[str, str]] = []
        self.judge_calls: list[tuple[str, str]] = []
        self._agent_step = 0

    async def generate_json(self, prompt: str, system_instruction: str | None = None, temperature: float = 0.3) -> str:
        self._agent_step += 1
        self.agent_calls.append((system_instruction or "", prompt))
        print(f"\n{BAR}\n[AGENT CALL #{self._agent_step}] system_instruction ↓\n{SUB}\n{system_instruction}")
        print(f"{SUB}\n[AGENT CALL #{self._agent_step}] user prompt ↓\n{SUB}\n{prompt}\n{BAR}\n")
        if self._agent_step < 3:
            return json.dumps({
                "action": "search",
                "query": f"mock query for step {self._agent_step}",
                "reasoning": f"need more info, step {self._agent_step}",
            })
        return json.dumps({
            "action": "answer",
            "answer": "Mock final answer based on accumulated context.",
            "reasoning": "I have enough information now.",
        })

    async def generate(self, prompt: str, system_instruction: str | None = None, temperature: float = 0.7, response_mime_type=None) -> str:
        self.judge_calls.append((system_instruction or "", prompt))
        is_judge = (system_instruction or "").startswith("You are a retrieval quality judge")
        label = "JUDGE" if is_judge else "FORCE-ANSWER"
        print(f"\n{BAR}\n[{label} CALL] system_instruction ↓\n{SUB}\n{system_instruction}")
        print(f"{SUB}\n[{label} CALL] user prompt ↓\n{SUB}\n{prompt}\n{BAR}\n")
        if is_judge:
            n = len([c for c in self.judge_calls if c[0] == system_instruction])
            return f"[MOCK JUDGE FEEDBACK #{n}] ignore chunk 3 (ad). Chunks 1-2 mention the topic; next query should target 'bit rate Noland Arbaugh Neuralink'."
        return "Mock forced plain-text answer."


class MockToolRegistry:
    """Returns canned RetrievalResults that look like grep/bm25 output."""

    def __init__(self):
        self._counter = 0

    async def search(self, tool_name: str, query: str, top_k: int = 5) -> list[RetrievalResult]:
        results = []
        for i in range(top_k):
            self._counter += 1
            results.append(
                RetrievalResult(
                    chunk_id=self._counter,
                    episode_id=100 + self._counter,
                    guest=f"Guest{self._counter}",
                    chunk_index=i,
                    text=(
                        f"[Mock chunk #{self._counter} text for query='{query}'] "
                        f"Lorem ipsum about topic. Includes names like Noland Arbaugh, "
                        f"Neuralink, bit rate 8.5 bps. (...truncated for brevity...)"
                    ),
                    score=0.9 - 0.1 * i,
                    source=tool_name,
                )
            )
        return results

    @staticmethod
    def tool_descriptions():
        from src.tools.retrieval_tools import ToolRegistry
        return ToolRegistry.tool_descriptions()


async def main():
    client = MockClient()
    registry = MockToolRegistry()
    # patch ToolRegistry.tool_descriptions access on instance
    registry.tool_descriptions = MockToolRegistry.tool_descriptions  # type: ignore

    question = (
        "Ryan Hall mentions a tech entrepreneur in his discussion on the mindset of warriors; "
        "in a separate conversation with that entrepreneur about brain-computer interfaces, "
        "what was the specific bit rate achieved by the first human participant, Noland Arbaugh?"
    )

    judge = ProcessJudge(client=client)
    agent = SearchAgent(
        client=client,
        tool_registry=registry,
        tool_name="bm25",
        max_steps=3,
        top_k_per_step=3,
        judge=judge,
    )

    # Fix: SearchAgent uses ToolRegistry.tool_descriptions() (staticmethod on the class).
    # Our mock registry doesn't inherit; patch the call site by monkeying the real class method access.
    # SearchAgent.run reads `ToolRegistry.tool_descriptions()` — we imported it directly there.
    # It works because SearchAgent references the real ToolRegistry class symbol, not our mock.

    trajectory = await agent.run(question)

    print(f"\n{BAR}\nFINAL TRAJECTORY SUMMARY\n{BAR}")
    print(f"Steps recorded: {len(trajectory.steps)}")
    for s in trajectory.steps:
        fb = (s.judge_feedback or "")[:80]
        print(f"  Step {s.step_number}: tool={s.tool_used} query='{s.query}' results={s.num_results} feedback='{fb}...'")
    print(f"Final answer: {trajectory.final_answer}")
    print(f"Total agent calls: {len(client.agent_calls)}")
    print(f"Total judge/force calls: {len(client.judge_calls)}")


if __name__ == "__main__":
    asyncio.run(main())
