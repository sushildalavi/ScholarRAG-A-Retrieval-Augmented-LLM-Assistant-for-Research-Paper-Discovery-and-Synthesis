import unittest

from backend.sense_resolver import resolve_sense


class SenseResolverTests(unittest.TestCase):
    def test_transformer_ambiguity(self):
        chunks = [
            {"title": "Transformers in Medical Imaging", "snippet": "transformer models for segmentation"},
            {"title": "Transformer Condition Monitoring", "snippet": "electrical transformer thermal monitoring"},
        ]
        out = resolve_sense("tell me about transformers", chunks)
        self.assertTrue(out["is_ambiguous"])
        self.assertGreaterEqual(len(out.get("options", [])), 2)


if __name__ == "__main__":
    unittest.main()
