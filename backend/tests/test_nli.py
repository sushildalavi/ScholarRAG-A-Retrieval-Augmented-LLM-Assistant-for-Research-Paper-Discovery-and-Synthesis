"""
Tests for pure-function helpers in backend/services/nli.py.
No OpenAI API calls are made — only the heuristic and parsing paths are tested.
"""
import unittest

from backend.services.nli import (
    _tokens,
    _heuristic_entailment_prob,
    _parse_prob_text,
)


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        result = _tokens("Hello World NLP")
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("nlp", result)

    def test_excludes_short_tokens(self):
        """Tokens with length <= 2 should be filtered out."""
        result = _tokens("I am a NLP model")
        self.assertNotIn("i", result)
        self.assertNotIn("am", result)
        self.assertNotIn("a", result)

    def test_empty_string(self):
        self.assertEqual(_tokens(""), set())

    def test_none_input(self):
        self.assertEqual(_tokens(None), set())

    def test_numbers_included(self):
        result = _tokens("model 123 was trained")
        self.assertIn("123", result)

    def test_punctuation_stripped(self):
        result = _tokens("attention, is all you need!")
        self.assertIn("attention", result)
        self.assertIn("need", result)
        self.assertNotIn("attention,", result)


class HeuristicEntailmentTests(unittest.TestCase):
    def test_identical_text_is_high(self):
        text = "the transformer model achieves state of the art results"
        score = _heuristic_entailment_prob(text, text)
        self.assertGreater(score, 0.8)

    def test_disjoint_text_is_low(self):
        h = "neural networks learn representations"
        p = "historical linguistics studies language change over time"
        score = _heuristic_entailment_prob(h, p)
        self.assertLess(score, 0.2)

    def test_partial_overlap_is_between(self):
        h = "attention mechanism improves translation quality"
        p = "attention in neural machine translation is effective"
        score = _heuristic_entailment_prob(h, p)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_hypothesis_returns_zero(self):
        self.assertEqual(_heuristic_entailment_prob("", "some premise text"), 0.0)

    def test_empty_premise_returns_zero(self):
        self.assertEqual(_heuristic_entailment_prob("some hypothesis", ""), 0.0)

    def test_output_clamped_to_01(self):
        text = "word " * 100
        score = _heuristic_entailment_prob(text, text)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class ParseProbTextTests(unittest.TestCase):
    def test_parses_labeled_probabilities(self):
        text = "entailment: 0.8\nneutral: 0.15\ncontradiction: 0.05"
        e, n, c = _parse_prob_text(text)
        self.assertAlmostEqual(e + n + c, 1.0, places=5)
        self.assertGreater(e, n)
        self.assertGreater(e, c)

    def test_parses_bare_number(self):
        """If only a number is found, treat it as entailment probability."""
        e, n, c = _parse_prob_text("0.9")
        self.assertAlmostEqual(e, 0.9, places=5)

    def test_parses_json_style(self):
        text = '{"entailment": 0.75, "neutral": 0.20, "contradiction": 0.05}'
        e, n, c = _parse_prob_text(text)
        self.assertAlmostEqual(e + n + c, 1.0, places=5)
        self.assertGreater(e, 0.5)

    def test_empty_string_returns_defaults(self):
        e, n, c = _parse_prob_text("")
        # Should return fallback, not raise
        self.assertGreaterEqual(e, 0.0)
        self.assertLessEqual(e, 1.0)

    def test_scores_sum_to_one_after_normalize(self):
        text = "entailment: 2.0\nneutral: 1.0\ncontradiction: 1.0"
        e, n, c = _parse_prob_text(text)
        self.assertAlmostEqual(e + n + c, 1.0, places=5)

    def test_invalid_text_does_not_raise(self):
        try:
            result = _parse_prob_text("this is not a probability")
            self.assertEqual(len(result), 3)
        except Exception as exc:
            self.fail(f"_parse_prob_text raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
