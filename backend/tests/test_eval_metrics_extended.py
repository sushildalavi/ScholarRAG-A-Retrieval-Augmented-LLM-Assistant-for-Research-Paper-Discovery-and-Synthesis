"""
Extended tests for backend/eval_metrics.py covering edge cases and
the aggregate_metrics() aggregation logic.
"""
import unittest

from backend.eval_metrics import recall_at_k, mrr, ndcg_at_k, aggregate_metrics


class RecallAtKTests(unittest.TestCase):
    def test_found_in_top_k(self):
        self.assertEqual(recall_at_k([1, 2, 3, 4], gold_doc_id=2, k=5), 1.0)

    def test_not_found_in_top_k(self):
        self.assertEqual(recall_at_k([1, 2, 3], gold_doc_id=99, k=3), 0.0)

    def test_found_exactly_at_k(self):
        self.assertEqual(recall_at_k([1, 2, 3, 4, 5], gold_doc_id=5, k=5), 1.0)

    def test_found_beyond_k(self):
        """Doc at position 6 should not count for Recall@5."""
        self.assertEqual(recall_at_k([1, 2, 3, 4, 5, 6], gold_doc_id=6, k=5), 0.0)

    def test_none_gold_returns_zero(self):
        self.assertEqual(recall_at_k([1, 2, 3], gold_doc_id=None, k=5), 0.0)

    def test_empty_pred_returns_zero(self):
        self.assertEqual(recall_at_k([], gold_doc_id=1, k=5), 0.0)


class MrrTests(unittest.TestCase):
    def test_first_position(self):
        self.assertAlmostEqual(mrr([5, 2, 3], gold_doc_id=5), 1.0)

    def test_second_position(self):
        self.assertAlmostEqual(mrr([1, 5, 3], gold_doc_id=5), 0.5)

    def test_fifth_position(self):
        self.assertAlmostEqual(mrr([1, 2, 3, 4, 5], gold_doc_id=5), 0.2)

    def test_not_found(self):
        self.assertAlmostEqual(mrr([1, 2, 3], gold_doc_id=99), 0.0)

    def test_none_gold(self):
        self.assertAlmostEqual(mrr([1, 2, 3], gold_doc_id=None), 0.0)


class NdcgAtKTests(unittest.TestCase):
    def test_first_position_is_perfect(self):
        score = ndcg_at_k([1, 2, 3], gold_doc_id=1, k=3)
        self.assertAlmostEqual(score, 1.0)

    def test_not_found_is_zero(self):
        score = ndcg_at_k([1, 2, 3], gold_doc_id=99, k=3)
        self.assertAlmostEqual(score, 0.0)

    def test_lower_position_gives_lower_score(self):
        score_pos1 = ndcg_at_k([5, 1, 2, 3, 4], gold_doc_id=5, k=5)
        score_pos3 = ndcg_at_k([1, 2, 5, 3, 4], gold_doc_id=5, k=5)
        self.assertGreater(score_pos1, score_pos3)

    def test_none_gold_is_zero(self):
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], gold_doc_id=None, k=3), 0.0)


class AggregateMetricsTests(unittest.TestCase):
    def test_empty_returns_zeros(self):
        result = aggregate_metrics([])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["recall_at"]["5"], 0.0)
        self.assertEqual(result["ndcg_at"]["10"], 0.0)

    def test_single_perfect_row(self):
        rows = [{"pred_doc_ids": [1, 2, 3], "gold_doc_id": 1}]
        result = aggregate_metrics(rows)
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["recall_at"]["1"], 1.0)

    def test_aggregation_averages_correctly(self):
        rows = [
            {"pred_doc_ids": [1, 2, 3], "gold_doc_id": 1},   # MRR=1.0
            {"pred_doc_ids": [2, 1, 3], "gold_doc_id": 1},   # MRR=0.5
        ]
        result = aggregate_metrics(rows)
        self.assertAlmostEqual(result["mrr"], 0.75)

    def test_recall_at_different_k(self):
        rows = [{"pred_doc_ids": [2, 3, 4, 5, 6, 1], "gold_doc_id": 1}]
        result = aggregate_metrics(rows)
        # gold_doc_id=1 is at position 6
        self.assertAlmostEqual(result["recall_at"]["1"], 0.0)
        self.assertAlmostEqual(result["recall_at"]["5"], 0.0)
        self.assertAlmostEqual(result["recall_at"]["10"], 1.0)

    def test_output_keys_always_present(self):
        result = aggregate_metrics([{"pred_doc_ids": [1], "gold_doc_id": 1}])
        self.assertIn("count", result)
        self.assertIn("recall_at", result)
        self.assertIn("mrr", result)
        self.assertIn("ndcg_at", result)
        for k in ("1", "3", "5", "10"):
            self.assertIn(k, result["recall_at"])
        for k in ("3", "5", "10"):
            self.assertIn(k, result["ndcg_at"])

    def test_scores_are_rounded(self):
        rows = [{"pred_doc_ids": [1, 2, 3], "gold_doc_id": 2}]
        result = aggregate_metrics(rows)
        # Check that values are rounded to 3 decimal places
        mrr_val = result["mrr"]
        self.assertAlmostEqual(mrr_val, round(mrr_val, 3))


if __name__ == "__main__":
    unittest.main()
