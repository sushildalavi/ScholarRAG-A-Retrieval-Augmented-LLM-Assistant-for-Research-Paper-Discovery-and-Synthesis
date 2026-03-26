import unittest

from backend.services.assistant_utils import _prune_uploaded_citations, _rank_and_trim_citations


class UploadedMultiDocTests(unittest.TestCase):
    def test_prune_uploaded_citations_preserves_multi_doc_selections(self):
        citations = [
            {
                "source": "uploaded",
                "doc_id": 1,
                "chunk_id": 101,
                "title": "Doc A",
                "snippet": "contrastive training objective and evaluation results",
                "confidence": 0.92,
            },
            {
                "source": "uploaded",
                "doc_id": 1,
                "chunk_id": 102,
                "title": "Doc A",
                "snippet": "additional details about contrastive pretraining",
                "confidence": 0.88,
            },
            {
                "source": "uploaded",
                "doc_id": 2,
                "chunk_id": 201,
                "title": "Doc B",
                "snippet": "reinforcement learning objective and ablation findings",
                "confidence": 0.67,
            },
        ]

        pruned = _prune_uploaded_citations(
            "compare the training objectives",
            citations,
            doc_ids=[1, 2],
        )

        self.assertEqual(len(pruned), 3)
        self.assertEqual({int(c["doc_id"]) for c in pruned}, {1, 2})

    def test_rank_and_trim_citations_keeps_one_hit_per_selected_doc(self):
        citations = [
            {
                "source": "uploaded",
                "doc_id": 1,
                "chunk_id": 101,
                "title": "Doc A",
                "snippet": "training objective uses contrastive loss and hard negatives",
                "confidence": 0.94,
                "sim_score": 0.92,
            },
            {
                "source": "uploaded",
                "doc_id": 1,
                "chunk_id": 102,
                "title": "Doc A",
                "snippet": "method section on contrastive retrieval with strong overlap",
                "confidence": 0.91,
                "sim_score": 0.89,
            },
            {
                "source": "uploaded",
                "doc_id": 2,
                "chunk_id": 201,
                "title": "Doc B",
                "snippet": "training objective uses reinforcement learning with reward shaping",
                "confidence": 0.61,
                "sim_score": 0.52,
            },
        ]

        reranked = _rank_and_trim_citations(
            "compare the training objectives used by these papers",
            citations,
            k=2,
            doc_ids=[1, 2],
        )

        self.assertEqual(len(reranked), 2)
        self.assertEqual({int(c["doc_id"]) for c in reranked}, {1, 2})


if __name__ == "__main__":
    unittest.main()
