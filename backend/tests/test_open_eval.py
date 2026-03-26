import json
import tempfile
import unittest
from pathlib import Path

from backend.open_eval import _assistant_payload, build_claim_rows, load_query_set
from backend.open_eval_metrics import aggregate_query_metrics, ranked_doc_ids, relevant_doc_gains
from backend.open_eval_spreadsheet import (
    build_calibration_records_from_claim_csv,
    build_claim_annotation_rows,
    build_corpus_doc_rows,
    build_query_summary_rows,
    build_retrieval_annotation_rows,
    dump_csv_rows,
    load_retrieval_annotations_csv,
)


class OpenEvalQueryTests(unittest.TestCase):
    def test_load_query_set_accepts_object_schema(self):
        payload = {
            "queries": [
                {"query_id": "q1", "query": "What method is proposed?", "doc_scope": "uploaded", "doc_id": 12},
                {"query_id": "q2", "query": "Compare the papers", "doc_scope": "uploaded", "doc_ids": [12, 14]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.json"
            path.write_text(json.dumps(payload))
            loaded = load_query_set(path)
        self.assertEqual(len(loaded["queries"]), 2)
        self.assertEqual(loaded["queries"][0]["doc_id"], 12)
        self.assertEqual(loaded["queries"][1]["doc_ids"], [12, 14])


class OpenEvalClaimTests(unittest.TestCase):
    def test_build_claim_rows_preserves_evidence_and_averages_msa(self):
        citations = [
            {
                "citation_id": "S1",
                "citation_index": 1,
                "evidence_id": "uploaded:1:11:2",
                "snippet": "The model uses contrastive loss during training.",
                "msa": {"M": 0.8, "S": 0.7, "A": 0.6},
            },
            {
                "citation_id": "S2",
                "citation_index": 2,
                "evidence_id": "uploaded:2:22:5",
                "snippet": "It reports improved recall compared with the baseline.",
                "msa": {"M": 0.4, "S": 0.6, "A": 0.8},
            },
        ]
        answer = (
            "The method uses contrastive loss [S1]. "
            "It improves recall compared with the baseline [S1][S2]."
        )

        claims = build_claim_rows("q1", answer, citations)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["evidence_ids"], ["uploaded:1:11:2"])
        self.assertEqual(claims[1]["citation_ids"], ["S1", "S2"])
        self.assertAlmostEqual(claims[1]["msa"]["M"], 0.6, places=4)
        self.assertIn("contrastive loss", claims[1]["evidence_text"])
        self.assertIn("improved recall", claims[1]["evidence_text"])

    def test_assistant_payload_defaults_to_all_ready_docs_for_unscoped_query(self):
        payload = _assistant_payload(
            {"query_id": "q1", "query": "Summarize the papers.", "doc_scope": "uploaded"},
            k=8,
            compute_msa=False,
            run_judge_llm=False,
            all_docs=[
                {"doc_id": 12, "title": "Paper A"},
                {"doc_id": 14, "title": "Paper B"},
            ],
        )
        self.assertEqual(payload["doc_ids"], [12, 14])


class OpenEvalMetricTests(unittest.TestCase):
    def test_ranked_doc_ids_prefers_retrieved_docs_list(self):
        query_row = {
            "retrieved_docs": [
                {"rank": 2, "doc_id": 14},
                {"rank": 1, "doc_id": 12},
            ]
        }
        self.assertEqual(ranked_doc_ids(query_row), [12, 14])

    def test_relevant_doc_gains_reads_corpus_labels(self):
        query_row = {
            "corpus_docs": [
                {"doc_id": 12, "relevance_label": "relevant"},
                {"doc_id": 14, "relevance_label": "partially_relevant"},
                {"doc_id": 16, "relevance_label": "not_relevant"},
            ]
        }
        self.assertEqual(relevant_doc_gains(query_row), {12: 2.0, 14: 1.0})

    def test_aggregate_query_metrics_handles_partial_relevance(self):
        query_rows = [
            {
                "query_id": "q1",
                "query": "Compare the methods.",
                "retrieved_docs": [
                    {"rank": 1, "doc_id": 16},
                    {"rank": 2, "doc_id": 12},
                    {"rank": 3, "doc_id": 14},
                ],
                "corpus_docs": [
                    {"doc_id": 12, "relevance_label": "relevant"},
                    {"doc_id": 14, "relevance_label": "partially_relevant"},
                    {"doc_id": 16, "relevance_label": "not_relevant"},
                ],
            }
        ]

        metrics = aggregate_query_metrics(query_rows)

        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(metrics["recall_at"]["1"], 0.0, places=4)
        self.assertAlmostEqual(metrics["recall_at"]["3"], 1.0, places=4)
        self.assertAlmostEqual(metrics["mrr"], 0.5, places=4)
        self.assertGreater(metrics["ndcg_at"]["10"], 0.0)


class OpenEvalSpreadsheetTests(unittest.TestCase):
    def test_build_csv_rows_from_exports(self):
        retrieval_rows = [
            {
                "query_id": "q1",
                "query": "What method is proposed?",
                "retrieved": [
                    {
                        "rank": 1,
                        "doc_id": 12,
                        "title": "Paper A",
                        "chunk_id": 101,
                        "page": 3,
                        "score": 0.91,
                        "chunk_text": "Chunk A",
                        "relevance_label": None,
                    }
                ],
                "corpus_docs": [
                    {"doc_id": 12, "title": "Paper A", "relevance_label": None},
                    {"doc_id": 14, "title": "Paper B", "relevance_label": None},
                ],
            }
        ]
        answer_rows = [
            {
                "query_id": "q1",
                "query": "What method is proposed?",
                "answer": "The paper proposes X [S1].",
                "claims": [
                    {
                        "claim_id": "q1_c1",
                        "text": "The paper proposes X.",
                        "evidence_ids": ["uploaded:12:101:3"],
                        "evidence_text": "We propose X.",
                        "label": None,
                        "citation_correct": None,
                        "msa": {"M": 0.9, "S": 0.6, "A": 0.7},
                    }
                ],
            }
        ]

        query_summary = build_query_summary_rows(answer_rows)
        retrieval_csv = build_retrieval_annotation_rows(retrieval_rows)
        corpus_csv = build_corpus_doc_rows(retrieval_rows)
        claim_csv = build_claim_annotation_rows(answer_rows)

        self.assertEqual(query_summary[0]["generated_answer"], "The paper proposes X [S1].")
        self.assertEqual(retrieval_csv[0]["document_title"], "Paper A")
        self.assertEqual(corpus_csv[1]["document_title"], "Paper B")
        self.assertEqual(claim_csv[0]["evidence_ids"], "uploaded:12:101:3")
        self.assertEqual(claim_csv[0]["msa_M"], 0.9)

    def test_load_retrieval_annotations_csv_prefers_corpus_labels(self):
        retrieval_rows = [
            {
                "query_id": "q1",
                "query": "Compare the methods.",
                "rank": 1,
                "doc_id": 16,
                "document_title": "Paper C",
                "chunk_id": 301,
                "page": 2,
                "retrieval_score": 0.88,
                "chunk_text": "Chunk C",
                "relevance_label": "not_relevant",
            },
            {
                "query_id": "q1",
                "query": "Compare the methods.",
                "rank": 2,
                "doc_id": 12,
                "document_title": "Paper A",
                "chunk_id": 101,
                "page": 3,
                "retrieval_score": 0.77,
                "chunk_text": "Chunk A",
                "relevance_label": "relevant",
            },
        ]
        corpus_rows = [
            {"query_id": "q1", "query": "Compare the methods.", "doc_id": 12, "document_title": "Paper A", "relevance_label": "relevant"},
            {"query_id": "q1", "query": "Compare the methods.", "doc_id": 14, "document_title": "Paper B", "relevance_label": "partially_relevant"},
            {"query_id": "q1", "query": "Compare the methods.", "doc_id": 16, "document_title": "Paper C", "relevance_label": "not_relevant"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            retrieval_path = Path(tmpdir) / "retrieval.csv"
            corpus_path = Path(tmpdir) / "corpus.csv"
            dump_csv_rows(
                retrieval_path,
                ["query_id", "query", "rank", "doc_id", "document_title", "chunk_id", "page", "retrieval_score", "chunk_text", "relevance_label"],
                retrieval_rows,
            )
            dump_csv_rows(
                corpus_path,
                ["query_id", "query", "doc_id", "document_title", "relevance_label"],
                corpus_rows,
            )
            queries = load_retrieval_annotations_csv(retrieval_path, corpus_csv_path=corpus_path)

        metrics = aggregate_query_metrics(queries)
        self.assertEqual(len(queries), 1)
        self.assertAlmostEqual(metrics["recall_at"]["1"], 0.0, places=4)
        self.assertAlmostEqual(metrics["recall_at"]["3"], 0.5, places=4)
        self.assertAlmostEqual(metrics["mrr"], 0.5, places=4)

    def test_build_calibration_records_from_claim_csv(self):
        rows = [
            {
                "query_id": "q1",
                "query": "What method is proposed?",
                "claim_id": "q1_c1",
                "claim_text": "The paper proposes X.",
                "evidence_ids": "uploaded:12:101:3",
                "evidence_text": "We propose X.",
                "msa_M": "0.9",
                "msa_S": "0.6",
                "msa_A": "0.7",
                "support_label": "supported",
                "citation_correct": "true",
                "annotator_notes": "",
            },
            {
                "query_id": "q1",
                "query": "What method is proposed?",
                "claim_id": "q1_c2",
                "claim_text": "The paper solves all problems.",
                "evidence_ids": "uploaded:12:101:3",
                "evidence_text": "We evaluate on one task.",
                "msa_M": "",
                "msa_S": "",
                "msa_A": "",
                "support_label": "unsupported",
                "citation_correct": "false",
                "annotator_notes": "",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            claim_path = Path(tmpdir) / "claims.csv"
            dump_csv_rows(
                claim_path,
                ["query_id", "query", "claim_id", "claim_text", "evidence_ids", "evidence_text", "msa_M", "msa_S", "msa_A", "support_label", "citation_correct", "annotator_notes"],
                rows,
            )
            records = build_calibration_records_from_claim_csv(claim_path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["label"], "supported")
        self.assertEqual(records[1]["label"], "unsupported")
        self.assertEqual(records[0]["msa"]["M"], 0.9)


if __name__ == "__main__":
    unittest.main()
