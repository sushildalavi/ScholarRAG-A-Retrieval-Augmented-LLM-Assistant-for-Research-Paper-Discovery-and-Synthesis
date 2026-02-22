import unittest

from backend.confidence import build_confidence


class ScoreSerializationTests(unittest.TestCase):
    def test_precision_and_fields(self):
        c = build_confidence(
            top_sim=0.742156,
            top_rerank_norm=0.913492,
            citation_coverage=0.666666,
            evidence_margin=0.21991,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
        )
        self.assertIn("score", c)
        self.assertIn("factors", c)
        self.assertIn("top_sim", c["factors"])
        self.assertIn("top_rerank_norm", c["factors"])
        # 4 decimal serialization behavior
        self.assertEqual(c["factors"]["top_sim"], round(c["factors"]["top_sim"], 4))
        self.assertEqual(c["factors"]["top_rerank_norm"], round(c["factors"]["top_rerank_norm"], 4))


if __name__ == "__main__":
    unittest.main()
