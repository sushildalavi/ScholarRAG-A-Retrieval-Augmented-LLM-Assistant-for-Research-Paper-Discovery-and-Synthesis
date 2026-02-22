import unittest

from backend.eval_metrics import aggregate_metrics, recall_at_k, mrr, ndcg_at_k


class EvalMetricsTests(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertEqual(recall_at_k([4, 2, 1], 2, 1), 0.0)
        self.assertEqual(recall_at_k([4, 2, 1], 2, 3), 1.0)

    def test_mrr(self):
        self.assertAlmostEqual(mrr([9, 7, 5], 7), 0.5)
        self.assertEqual(mrr([9, 7, 5], 1), 0.0)

    def test_ndcg(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1], 3, 3), 1.0)
        self.assertEqual(ndcg_at_k([3, 2, 1], 5, 3), 0.0)

    def test_aggregate(self):
        out = aggregate_metrics([
            {"pred_doc_ids": [1, 2, 3], "gold_doc_id": 1},
            {"pred_doc_ids": [4, 2, 1], "gold_doc_id": 2},
        ])
        self.assertEqual(out["count"], 2)
        self.assertGreaterEqual(out["recall_at"]["1"], 0.5)


if __name__ == "__main__":
    unittest.main()
