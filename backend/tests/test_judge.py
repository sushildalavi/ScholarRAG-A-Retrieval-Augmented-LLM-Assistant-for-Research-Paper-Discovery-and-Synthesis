"""
Tests for pure-function helpers in backend/services/judge.py.
No OpenAI API calls are made.
"""
import unittest

from backend.services.judge import (
    _split_sentences,
    _safe_int,
    _safe_float,
    _parse_judge_json,
    _fallback_report,
)


class SplitSentencesTests(unittest.TestCase):
    def test_basic_split(self):
        text = "Attention is all you need. The model achieves SOTA. Results are strong."
        sentences = _split_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_empty_string(self):
        self.assertEqual(_split_sentences(""), [])

    def test_none_input(self):
        self.assertEqual(_split_sentences(None), [])

    def test_single_sentence_no_split(self):
        text = "The model outperforms the baseline."
        result = _split_sentences(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_strips_whitespace(self):
        text = "  First sentence.  Second sentence.  "
        result = _split_sentences(text)
        for s in result:
            self.assertEqual(s, s.strip())

    def test_question_and_exclamation(self):
        text = "Does the model work? Yes! It does."
        result = _split_sentences(text)
        self.assertGreaterEqual(len(result), 2)


class SafeIntTests(unittest.TestCase):
    def test_converts_string_to_int(self):
        self.assertEqual(_safe_int("5"), 5)

    def test_converts_float_string(self):
        self.assertEqual(_safe_int("3.7"), 3)

    def test_returns_default_on_invalid(self):
        self.assertEqual(_safe_int("abc", default=0), 0)

    def test_returns_default_on_none(self):
        self.assertEqual(_safe_int(None, default=-1), -1)

    def test_passthrough_int(self):
        self.assertEqual(_safe_int(42), 42)


class SafeFloatTests(unittest.TestCase):
    def test_converts_string_to_float(self):
        self.assertAlmostEqual(_safe_float("0.75"), 0.75)

    def test_returns_default_on_invalid(self):
        self.assertAlmostEqual(_safe_float("not_a_number", default=0.5), 0.5)

    def test_returns_default_on_none(self):
        self.assertAlmostEqual(_safe_float(None, default=0.0), 0.0)

    def test_passthrough_float(self):
        self.assertAlmostEqual(_safe_float(0.33), 0.33)


class ParseJudgeJsonTests(unittest.TestCase):
    def test_parses_clean_json(self):
        raw = '{"overall_score": 0.85, "coverage": 0.9}'
        result = _parse_judge_json(raw)
        self.assertAlmostEqual(result["overall_score"], 0.85)

    def test_extracts_json_from_markdown(self):
        raw = 'Here is the result:\n```json\n{"overall_score": 0.7}\n```'
        result = _parse_judge_json(raw)
        # Should extract the JSON object even from markdown wrapping
        self.assertIsInstance(result, dict)

    def test_returns_empty_dict_on_invalid(self):
        result = _parse_judge_json("not json at all")
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_empty_string(self):
        result = _parse_judge_json("")
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_none(self):
        result = _parse_judge_json(None)
        self.assertEqual(result, {})


class FallbackReportTests(unittest.TestCase):
    def test_all_cited_sentences(self):
        sentences = [
            "The model achieves high accuracy [S1].",
            "Recall is also improved [S2].",
        ]
        citations = [{"id": 1, "text": "..."}, {"id": 2, "text": "..."}]
        report = _fallback_report(sentences, citations)
        self.assertAlmostEqual(report["overall_score"], 1.0)
        self.assertIn("coverage_by_citation_id", report)

    def test_no_citations_gives_zero_score(self):
        sentences = [
            "The model works well.",
            "Results are impressive.",
        ]
        report = _fallback_report(sentences, [])
        self.assertAlmostEqual(report["overall_score"], 0.0)

    def test_partial_coverage(self):
        sentences = [
            "First claim [S1].",
            "Second claim with no citation.",
            "Third claim [S2].",
        ]
        citations = [{"id": 1}, {"id": 2}]
        report = _fallback_report(sentences, citations)
        self.assertAlmostEqual(report["overall_score"], 2 / 3, places=3)

    def test_empty_sentences_returns_zero(self):
        report = _fallback_report([], [])
        self.assertAlmostEqual(report["overall_score"], 0.0)

    def test_report_contains_required_keys(self):
        sentences = ["A claim [S1]."]
        citations = [{"id": 1}]
        report = _fallback_report(sentences, citations)
        for key in ("overall_score", "coverage_by_citation_id", "unsupported"):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
