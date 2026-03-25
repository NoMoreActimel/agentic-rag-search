#!/usr/bin/env python3
"""Quick test: run the agent on a single question to verify everything works."""

import asyncio
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import QA_PAIRS_JSON
from src.llm.gemini_client import GeminiClient
from src.tools.retrieval_tools import ToolRegistry
from src.agent.search_agent import SearchAgent
from src.judge.judges import ProcessJudge


async def main():
    # Load one question
    with open(QA_PAIRS_JSON) as f:
        qa_pairs = json.load(f)

    qa = qa_pairs[0]
    print(f"Question: {qa['question'][:100]}...")
    print(f"Expected: {qa['answer'][:100]}...")
    print(f"Type: {qa['qa_type']}")
    print("=" * 60)

    client = GeminiClient()
    tool_registry = ToolRegistry(gemini_client=client)

    # Test 1: BM25, no judge
    print("\n--- Test 1: BM25 + No Judge (3 steps) ---")
    try:
        agent = SearchAgent(
            client=client,
            tool_registry=tool_registry,
            tool_name="bm25",
            max_steps=3,
            judge=None,
        )
        t1 = await agent.run(qa["question"])

        print(f"Steps: {len(t1.steps)}")
        for s in t1.steps:
            print(f"  Step {s.step_number}: tool={s.tool_used}, query='{s.query[:50]}', results={s.num_results}")
        print(f"Answer: {t1.final_answer[:200]}")
        print(f"Episodes: {t1.unique_episodes_found}")
        print("TEST 1: PASSED")
    except Exception as e:
        print(f"TEST 1: FAILED — {e}")
        traceback.print_exc()

    # Wait before Test 2 to avoid rate limits
    print("\nWaiting 10 seconds before Test 2 (rate limit cooldown)...")
    await asyncio.sleep(10)

    # Test 2: BM25 + ProcessJudge
    print("\n--- Test 2: BM25 + Process Judge (3 steps) ---")
    try:
        judge = ProcessJudge(client=client)
        agent2 = SearchAgent(
            client=client,
            tool_registry=tool_registry,
            tool_name="bm25",
            max_steps=3,
            judge=judge,
        )
        t2 = await agent2.run(qa["question"])

        print(f"Steps: {len(t2.steps)}")
        for s in t2.steps:
            print(f"  Step {s.step_number}: tool={s.tool_used}, query='{s.query[:50]}', results={s.num_results}")
            if s.judge_feedback:
                print(f"    Judge: {s.judge_feedback[:120]}...")
        print(f"Answer: {t2.final_answer[:200]}")
        print("TEST 2: PASSED")
    except Exception as e:
        print(f"TEST 2: FAILED — {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    client.print_stats()


if __name__ == "__main__":
    asyncio.run(main())