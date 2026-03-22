"""
Tests for pure-function helpers in backend/services/embeddings.py.
No network calls or database connections are required.
"""
import unittest

from backend.services.embeddings import (
    _trim_or_pad,
    _prepare_text,
    _validate_embedding_payload,
    _is_context_length_error,
)


class TrimOrPadTests(unittest.TestCase):
    def test_exact_dim_passthrough(self):
        vec = [0.1, 0.2, 0.3]
        self.assertEqual(_trim_or_pad(vec, 3), [0.1, 0.2, 0.3])

    def test_pads_short_vector(self):
        vec = [0.5, 0.5]
        result = _trim_or_pad(vec, 5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[-3:], [0.0, 0.0, 0.0])

    def test_trims_long_vector(self):
        vec = [float(i) for i in range(10)]
        result = _trim_or_pad(vec, 4)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [0.0, 1.0, 2.0, 3.0])

    def test_padding_to_1536_from_1024(self):
        """Simulates mxbai-embed-large 1024-d → 1536-d padding."""
        vec = [0.1] * 1024
        result = _trim_or_pad(vec, 1536)
        self.assertEqual(len(result), 1536)
        self.assertEqual(result[1024:], [0.0] * 512)

    def test_json_string_input(self):
        """Handles JSON-encoded vector strings from legacy DB rows."""
        result = _trim_or_pad("[0.1, 0.2, 0.3]", 3)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)

    def test_empty_vector_pads_to_dim(self):
        result = _trim_or_pad([], 4)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0])


class PrepareTextTests(unittest.TestCase):
    def test_whitespace_normalization(self):
        text = "  hello   world  "
        result = _prepare_text(text, "query")
        self.assertEqual(result, "hello world")

    def test_query_truncation_at_max_words(self):
        """Queries should be truncated at EMBEDDING_MAX_QUERY_WORDS (128)."""
        long_text = " ".join(["word"] * 200)
        result = _prepare_text(long_text, "query")
        self.assertLessEqual(len(result.split()), 128)

    def test_document_truncation_at_max_words(self):
        """Documents should be truncated at EMBEDDING_MAX_DOC_WORDS (256)."""
        long_text = " ".join(["word"] * 400)
        result = _prepare_text(long_text, "document")
        self.assertLessEqual(len(result.split()), 256)

    def test_short_text_not_truncated(self):
        text = "short text"
        self.assertEqual(_prepare_text(text, "query"), "short text")

    def test_empty_string(self):
        self.assertEqual(_prepare_text("", "query"), "")

    def test_none_coerced(self):
        result = _prepare_text(None, "query")
        self.assertEqual(result, "")


class ValidateEmbeddingPayloadTests(unittest.TestCase):
    def test_valid_embedding_key(self):
        payload = {"embedding": [0.1] * 256}
        result = _validate_embedding_payload(payload)
        self.assertEqual(len(result), 256)

    def test_valid_embeddings_key(self):
        """Handles /api/embed response format that uses 'embeddings'."""
        payload = {"embeddings": [[0.1] * 256]}
        result = _validate_embedding_payload(payload)
        self.assertEqual(len(result), 256)

    def test_raises_on_missing_key(self):
        with self.assertRaises(RuntimeError):
            _validate_embedding_payload({})

    def test_raises_on_empty_list(self):
        with self.assertRaises(RuntimeError):
            _validate_embedding_payload({"embedding": []})

    def test_raises_on_too_short(self):
        with self.assertRaises(RuntimeError):
            _validate_embedding_payload({"embedding": [0.1] * 10})

    def test_raises_on_non_numeric(self):
        with self.assertRaises(RuntimeError):
            _validate_embedding_payload({"embedding": ["a", "b"] * 100})

    def test_all_floats_coerced(self):
        payload = {"embedding": [1, 2, 3] * 50}
        result = _validate_embedding_payload(payload)
        self.assertTrue(all(isinstance(v, float) for v in result))


class ContextLengthErrorTests(unittest.TestCase):
    def test_context_length_detected(self):
        self.assertTrue(_is_context_length_error("context length exceeded"))
        self.assertTrue(_is_context_length_error("Input length exceeds maximum"))

    def test_other_errors_not_flagged(self):
        self.assertFalse(_is_context_length_error("connection refused"))
        self.assertFalse(_is_context_length_error("timeout"))
        self.assertFalse(_is_context_length_error(""))


if __name__ == "__main__":
    unittest.main()
