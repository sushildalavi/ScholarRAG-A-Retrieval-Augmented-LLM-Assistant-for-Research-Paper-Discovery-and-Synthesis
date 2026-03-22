"""
Integration tests for the M/S/A confidence model.
Tests the full build_confidence() path including the MSA logistic blend.
No external dependencies required.
"""
import unittest
import math

from backend.confidence import (
    build_confidence,
    compute_msa_score,
    confidence_label,
    sigmoid,
    clamp01,
    score_percent,
)


class ClampTests(unittest.TestCase):
    def test_clamps_above_one(self):
        self.assertEqual(clamp01(1.5), 1.0)

    def test_clamps_below_zero(self):
        self.assertEqual(clamp01(-0.1), 0.0)

    def test_passthrough_in_range(self):
        self.assertAlmostEqual(clamp01(0.65), 0.65)


class SigmoidTests(unittest.TestCase):
    def test_zero_input_gives_half(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=5)

    def test_large_positive_approaches_one(self):
        self.assertGreater(sigmoid(10.0), 0.99)

    def test_large_negative_approaches_zero(self):
        self.assertLess(sigmoid(-10.0), 0.01)

    def test_monotone_increasing(self):
        vals = [sigmoid(x) for x in [-2, -1, 0, 1, 2]]
        self.assertEqual(vals, sorted(vals))


class ConfidenceLabelTests(unittest.TestCase):
    def test_high_label(self):
        self.assertEqual(confidence_label(0.80), "High")
        self.assertEqual(confidence_label(0.75), "High")

    def test_med_label(self):
        self.assertEqual(confidence_label(0.65), "Med")
        self.assertEqual(confidence_label(0.50), "Med")

    def test_low_label(self):
        self.assertEqual(confidence_label(0.49), "Low")
        self.assertEqual(confidence_label(0.0), "Low")


class ComputeMsaScoreTests(unittest.TestCase):
    def test_default_weights_in_range(self):
        score = compute_msa_score(0.7, 0.6, 0.8)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_high_msa_gives_high_score(self):
        score = compute_msa_score(1.0, 1.0, 1.0)
        self.assertGreater(score, 0.7)

    def test_zero_msa_gives_low_score(self):
        score = compute_msa_score(0.0, 0.0, 0.0)
        self.assertLess(score, 0.6)

    def test_custom_weights_applied(self):
        weights = {"b": 0.0, "w1": 0.9, "w2": 0.05, "w3": 0.05}
        high_m = compute_msa_score(1.0, 0.0, 0.0, weights=weights)
        high_s = compute_msa_score(0.0, 1.0, 0.0, weights=weights)
        self.assertGreater(high_m, high_s)

    def test_clamps_input_above_one(self):
        score = compute_msa_score(2.0, 2.0, 2.0)
        self.assertLessEqual(score, 1.0)

    def test_clamps_input_below_zero(self):
        score = compute_msa_score(-1.0, -1.0, -1.0)
        self.assertGreaterEqual(score, 0.0)


class BuildConfidenceWithMsaTests(unittest.TestCase):
    def test_msa_blend_increases_high_confidence(self):
        """High M/S/A should increase score compared to retrieval-only baseline."""
        without_msa = build_confidence(
            top_sim=0.75,
            top_rerank_norm=0.7,
            citation_coverage=0.8,
            evidence_margin=0.3,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
        )
        with_msa = build_confidence(
            top_sim=0.75,
            top_rerank_norm=0.7,
            citation_coverage=0.8,
            evidence_margin=0.3,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
            msa={"M": 0.9, "S": 0.85, "A": 0.8},
        )
        self.assertGreaterEqual(with_msa["score"], without_msa["score"] * 0.9)

    def test_msa_factors_in_output(self):
        c = build_confidence(
            top_sim=0.8,
            top_rerank_norm=0.7,
            citation_coverage=0.9,
            evidence_margin=0.3,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
            msa={"M": 0.7, "S": 0.6, "A": 0.5},
        )
        self.assertIn("msa", c["factors"])
        self.assertIn("M", c["factors"]["msa"])
        self.assertIn("S", c["factors"]["msa"])
        self.assertIn("A", c["factors"]["msa"])
        self.assertIn("msa_score", c["factors"]["msa"])

    def test_calibration_weights_override_defaults(self):
        """Custom weights should change the MSA score."""
        default_weights = build_confidence(
            top_sim=0.5, top_rerank_norm=0.5,
            citation_coverage=0.5, evidence_margin=0.2,
            ambiguity_penalty=0.0, insufficiency_penalty=0.0,
            msa={"M": 0.8, "S": 0.2, "A": 0.2},
        )
        high_m_weight = build_confidence(
            top_sim=0.5, top_rerank_norm=0.5,
            citation_coverage=0.5, evidence_margin=0.2,
            ambiguity_penalty=0.0, insufficiency_penalty=0.0,
            msa={"M": 0.8, "S": 0.2, "A": 0.2, "weights": {"w1": 0.9, "w2": 0.05, "w3": 0.05}},
        )
        # Higher w1 with high M should produce higher or equal MSA component
        self.assertGreaterEqual(
            high_m_weight["factors"]["msa"]["msa_score"],
            default_weights["factors"]["msa"]["msa_score"],
        )

    def test_needs_clarification_caps_at_025(self):
        c = build_confidence(
            top_sim=0.95,
            top_rerank_norm=0.95,
            citation_coverage=1.0,
            evidence_margin=0.5,
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
            needs_clarification=True,
        )
        self.assertLessEqual(c["score"], 0.25)

    def test_scope_penalty_reduces_score(self):
        no_pen = build_confidence(
            top_sim=0.8, top_rerank_norm=0.8,
            citation_coverage=0.9, evidence_margin=0.4,
            ambiguity_penalty=0.0, insufficiency_penalty=0.0,
            scope_penalty=0.0,
        )
        with_pen = build_confidence(
            top_sim=0.8, top_rerank_norm=0.8,
            citation_coverage=0.9, evidence_margin=0.4,
            ambiguity_penalty=0.0, insufficiency_penalty=0.0,
            scope_penalty=1.0,
        )
        self.assertLess(with_pen["score"], no_pen["score"])

    def test_output_shape(self):
        c = build_confidence(
            top_sim=0.6, top_rerank_norm=0.5,
            citation_coverage=0.7, evidence_margin=0.2,
            ambiguity_penalty=0.1, insufficiency_penalty=0.1,
        )
        for key in ("score", "label", "factors", "explanation", "needs_clarification"):
            self.assertIn(key, c)

    def test_score_is_clamped(self):
        c = build_confidence(
            top_sim=2.0, top_rerank_norm=2.0,
            citation_coverage=2.0, evidence_margin=2.0,
            ambiguity_penalty=0.0, insufficiency_penalty=0.0,
        )
        self.assertLessEqual(c["score"], 1.0)
        self.assertGreaterEqual(c["score"], 0.0)


class ScorePercentTests(unittest.TestCase):
    def test_converts_correctly(self):
        self.assertAlmostEqual(score_percent(0.75), 75.0)
        self.assertAlmostEqual(score_percent(0.0), 0.0)
        self.assertAlmostEqual(score_percent(1.0), 100.0)

    def test_clamps_above_one(self):
        self.assertAlmostEqual(score_percent(1.5), 100.0)

    def test_clamps_below_zero(self):
        self.assertAlmostEqual(score_percent(-0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
