"""Unit tests for deterministic metric helpers."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import _exact_match, _reference_episode_recall, _token_f1  # noqa: E402


class TestMetricsUtils(unittest.TestCase):
    def test_exact_match_normalizes(self) -> None:
        self.assertTrue(_exact_match("Answer: 42.", "answer 42"))
        self.assertFalse(_exact_match("forty two", "forty one"))

    def test_token_f1(self) -> None:
        score = _token_f1("the cat sat on mat", "cat sat")
        self.assertGreater(score, 0.4)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(_token_f1("abc", "xyz"), 0.0)

    def test_reference_episode_recall(self) -> None:
        recall = _reference_episode_recall([1, 2, 3], [2, 3, 9])
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertEqual(_reference_episode_recall([], [1]), 0.0)


if __name__ == "__main__":
    unittest.main()
