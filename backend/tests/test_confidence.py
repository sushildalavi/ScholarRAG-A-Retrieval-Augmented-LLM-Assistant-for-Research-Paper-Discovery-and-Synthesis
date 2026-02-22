import unittest

from backend.confidence import build_confidence


class ConfidenceTests(unittest.TestCase):
    def test_confidence_shape_and_label(self):
        c = build_confidence(
            top_sim=0.9,
            top_rerank_norm=0.8,
            citation_coverage=1.0,
            evidence_margin=0.4,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
        )
        self.assertIn("score", c)
        self.assertIn("label", c)
        self.assertIn("factors", c)
        self.assertTrue(0.0 <= c["score"] <= 1.0)
        self.assertEqual(c["label"], "High")

    def test_ambiguity_penalty_lowers_confidence(self):
        hi = build_confidence(
            top_sim=0.85,
            top_rerank_norm=0.8,
            citation_coverage=0.9,
            evidence_margin=0.4,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
        )
        lo = build_confidence(
            top_sim=0.85,
            top_rerank_norm=0.8,
            citation_coverage=0.9,
            evidence_margin=0.4,
            ambiguity_penalty=1.0,
            insufficiency_penalty=0.0,
        )
        self.assertLess(lo["score"], hi["score"])


if __name__ == "__main__":
    unittest.main()
