"""Unit tests for retrieval quality reweighting behavior."""

import asyncio
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.retrieval_tools import ToolRegistry  # noqa: E402


class TestQualityReweight(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {
                    "chunk_id": 1,
                    "episode_id": 100,
                    "guest": "Alice",
                    "chunk_index": 0,
                    "text": "Neural networks and transformers are discussed here.",
                },
                {
                    "chunk_id": 2,
                    "episode_id": 101,
                    "guest": "Bob",
                    "chunk_index": 1,
                    "text": "This chunk is mostly unrelated filler text.",
                },
            ]
        )

    def test_reweight_missing_scores_keeps_baseline(self) -> None:
        async def run() -> None:
            base_registry = ToolRegistry(
                chunks_df=self.df,
                bm25_retriever=object(),
                faiss_index=object(),
                quality_reweight=False,
            )
            weighted_registry = ToolRegistry(
                chunks_df=self.df,
                bm25_retriever=object(),
                faiss_index=object(),
                quality_reweight=True,
                quality_scores={999: {"integrity": 5, "information_density": 5}},
            )

            baseline = await base_registry.search("grep", "transformers neural", top_k=2)
            weighted = await weighted_registry.search("grep", "transformers neural", top_k=2)

            self.assertEqual(len(baseline), len(weighted))
            self.assertEqual(baseline[0].chunk_id, weighted[0].chunk_id)
            self.assertEqual(baseline[0].score, weighted[0].score)
            self.assertIsNone(weighted[0].quality_score)
            self.assertEqual(weighted[0].original_score, baseline[0].score)

        asyncio.run(run())

    def test_reweight_with_scores_changes_ordering_signal(self) -> None:
        async def run() -> None:
            weighted_registry = ToolRegistry(
                chunks_df=self.df,
                bm25_retriever=object(),
                faiss_index=object(),
                quality_reweight=True,
                quality_weight=0.9,
                quality_scores={
                    1: {"integrity": 1, "information_density": 1, "is_ad_or_filler": True},
                    2: {"integrity": 5, "information_density": 5, "is_ad_or_filler": False},
                },
            )
            weighted = await weighted_registry.search("grep", "transformers neural filler", top_k=2)
            self.assertEqual(len(weighted), 2)
            self.assertIsNotNone(weighted[0].quality_score)
            self.assertIsNotNone(weighted[1].quality_score)
            self.assertGreater(weighted[0].score, weighted[1].score)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
