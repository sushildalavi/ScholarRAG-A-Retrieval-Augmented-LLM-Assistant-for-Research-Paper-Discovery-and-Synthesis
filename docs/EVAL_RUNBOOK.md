# ScholarRAG Evaluation Runbook

Step-by-step guide for running the full evaluation pipeline: retrieval quality, synthesis faithfulness, and M/S/A confidence calibration.

## Prerequisites

- Backend running at `http://localhost:8000`
- All 15 papers uploaded (see Step 1)
- Python 3.11+, `jq` installed for JSON slicing
- `eval_data/` and `scripts/` directories present in repo root

---

## Step 1 — Upload Papers and Record doc_ids

Upload each paper via the UI or API. After uploading, list all documents to record their assigned database IDs.

```bash
# List all uploaded documents
curl -s http://localhost:8000/documents | jq '.[] | {id, title}'
```

You should see output like:

```json
{"id": 3, "title": "Dense Passage Retrieval for Open-Domain Question Answering"}
{"id": 7, "title": "ColBERT: Efficient and Effective Passage Search..."}
...
```

Build a mapping of `_doc_key` → `id`. The keys used in all eval files are:

| `_doc_key`          | Paper title (substring match)                                 |
|---------------------|---------------------------------------------------------------|
| `dpr`               | Dense Passage Retrieval for Open-Domain Question Answering    |
| `colbert`           | ColBERT                                                       |
| `rag`               | Retrieval-Augmented Generation for Knowledge-Intensive NLP    |
| `beir`              | BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation      |
| `squad`             | SQuAD: 100,000+ Questions for Machine Comprehension           |
| `natural_questions` | Natural Questions: A Benchmark for Question Answering         |
| `drqa`              | Reading Wikipedia to Answer Open-Domain Questions (DrQA)      |
| `pegasus`           | PEGASUS: Pre-training with Extracted Gap-sentences            |
| `bart`              | BART: Denoising Sequence-to-Sequence Pre-training             |
| `factscore`         | FActScore: Fine-grained Atomic Evaluation                     |
| `bert`              | BERT: Pre-training of Deep Bidirectional Transformers         |
| `transformer`       | Attention Is All You Need                                     |
| `cot`               | Chain-of-Thought Prompting Elicits Reasoning                  |
| `instructgpt`       | Training language models to follow instructions (InstructGPT) |
| `llm_judge`         | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena        |

---

## Step 2 — Apply doc_ids to Eval Files

Run the mapping script to replace all `null` placeholders with real database IDs.

```bash
python scripts/apply_doc_ids.py \
  --mapping '{"dpr": 3, "colbert": 7, "rag": 2, ...}' \
  --files \
    eval_data/queries_120_master.json \
    eval_data/retrieval_eval_cases.json \
    eval_data/judge_eval_cases.json \
    eval_data/msa_labeling_template.jsonl
```

If `apply_doc_ids.py` does not exist yet, apply the mapping manually or with a one-liner:

```bash
# Example: replace doc_key "dpr" with id 3 in retrieval cases
# (repeat for each paper)
python - <<'EOF'
import json, pathlib

MAPPING = {
    "dpr": 3,
    "colbert": 7,
    "rag": 2,
    # ... fill in your actual IDs
}

path = pathlib.Path("eval_data/retrieval_eval_cases.json")
data = json.loads(path.read_text())
for case in data["cases"]:
    key = case.get("_doc_key")
    if key and key in MAPPING:
        case["expected_doc_id"] = MAPPING[key]
path.write_text(json.dumps(data, indent=2))
print("Done.")
EOF
```

For `judge_eval_cases.json`, multi-doc cases use `doc_ids` (list):

```python
for case in data["cases"]:
    if case.get("judge_type") == "single":
        case["doc_id"] = MAPPING.get(case["_doc_key"])
    elif case.get("judge_type") == "multi":
        case["doc_ids"] = [MAPPING[k] for k in case["_doc_keys"] if k in MAPPING]
```

---

## Step 3 — Run Retrieval Evaluation

Send the 80 retrieval cases to `/eval/run`. This searches across **all** uploaded documents for each query and measures whether the correct paper appears in the top-k results.

```bash
# Build the full payload
python - <<'EOF'
import json, pathlib

cases_file = pathlib.Path("eval_data/retrieval_eval_cases.json")
cases_data = json.loads(cases_file.read_text())

payload = {
    "name": "Retrieval Eval — 80 queries",
    "scope": "uploaded",
    "k": 10,
    "cases": cases_data["cases"]
}

pathlib.Path("eval_data/_retrieval_payload.json").write_text(json.dumps(payload))
print("Payload ready.")
EOF

# Submit
curl -s -X POST http://localhost:8000/eval/run \
     -H "Content-Type: application/json" \
     -d @eval_data/_retrieval_payload.json \
     -o eval_data/retrieval_results.json

echo "Exit code: $?"
```

### Inspect results

```bash
# Summary metrics
jq '{
  count,
  retrieval_recall_at_1:  .metrics_retrieval_only.recall_at["1"],
  retrieval_recall_at_5:  .metrics_retrieval_only.recall_at["5"],
  retrieval_recall_at_10: .metrics_retrieval_only.recall_at["10"],
  retrieval_mrr:          .metrics_retrieval_only.mrr,
  rerank_recall_at_5:     .metrics_retrieval_rerank.recall_at["5"],
  rerank_ndcg_at_10:      .metrics_retrieval_rerank.ndcg_at["10"]
}' eval_data/retrieval_results.json

# Cases where the correct doc did NOT appear in top-10
jq '[.details[] | select(.hit_rerank == false) | {query, expected_doc_id}]' \
  eval_data/retrieval_results.json
```

### Metrics returned by `/eval/run`

The response has two metric blocks, each with the same structure:

```json
{
  "count": 80,
  "recall_at": {"1": 0.72, "3": 0.86, "5": 0.90, "10": 0.95},
  "mrr": 0.81,
  "ndcg_at": {"3": 0.84, "5": 0.87, "10": 0.89}
}
```

- `metrics_retrieval_only` — before re-ranking
- `metrics_retrieval_rerank` — after cross-encoder re-ranking

---

## Step 4 — Run Judge Evaluation

Send the 40 synthesis cases to `/eval/judge`. The endpoint calls `assistant_answer()` for each case (generates a RAG answer), then evaluates faithfulness via LLM judge.

**Estimated runtime**: ~2–5 minutes for 40 cases depending on API latency.

```bash
# Build the full payload
python - <<'EOF'
import json, pathlib

cases_file = pathlib.Path("eval_data/judge_eval_cases.json")
cases_data = json.loads(cases_file.read_text())

payload = {
    "scope": "uploaded",
    "k": 10,
    "run_judge_llm": True,
    "cases": cases_data["cases"]
}

pathlib.Path("eval_data/_judge_payload.json").write_text(json.dumps(payload))
print("Payload ready.")
EOF

# Submit
curl -s -X POST http://localhost:8000/eval/judge \
     -H "Content-Type: application/json" \
     -d @eval_data/_judge_payload.json \
     -o eval_data/judge_results.json

echo "Exit code: $?"
```

### Inspect results

```bash
# Aggregate faithfulness metrics
jq '.metrics' eval_data/judge_results.json

# Per-case scores
jq '[.details[] | {
  query: .query[:60],
  overall_score: .faithfulness.overall_score,
  citation_coverage: .faithfulness.citation_coverage,
  supported: .faithfulness.supported_count,
  unsupported: .faithfulness.unsupported_count
}]' eval_data/judge_results.json

# All unsupported claims across all cases
jq '[.details[].faithfulness.unsupported[] | {sentence, reason}]' \
  eval_data/judge_results.json
```

### Faithfulness report structure per case

```json
{
  "overall_score": 0.83,
  "citation_coverage": 0.75,
  "supported_count": 5,
  "unsupported_count": 1,
  "sentence_count": 6,
  "claims": [
    {
      "sentence_id": "S1",
      "sentence": "RAG combines parametric and non-parametric memory.",
      "supported": true,
      "evidence_ids": ["chunk_42"],
      "reason": "Directly stated in the retrieved passage."
    }
  ],
  "unsupported": [...],
  "method": "llm"
}
```

---

## Step 5 — Export Calibration Records

Convert the judge results into calibration records for `POST /confidence/calibrate`.

```bash
python scripts/export_msa_records.py \
  eval_data/judge_results.json \
  -o eval_data/calibration_records_auto.jsonl

# Check label balance
jq -r '.label' eval_data/calibration_records_auto.jsonl | sort | uniq -c
```

Expected output:
```
 142 supported
  58 unsupported
```

If the split is severely imbalanced (>90% one class), consider collecting more judge cases on difficult/edge-case queries before calibrating.

---

## Step 6 — Human Annotation (Optional but Recommended)

The file `eval_data/msa_labeling_template.jsonl` contains 250 pre-structured annotation slots. Fill in the `sentence`, `evidence_text`, and `label` fields for each slot. `S` and `A` can be left `null`; the API will use default values (0.5).

Template record format:
```json
{
  "id": 1,
  "source_query_id": 81,
  "_doc_key": "rag",
  "query": "Synthesize the argument in the RAG paper...",
  "sentence": "",
  "evidence_text": "",
  "S": null,
  "A": null,
  "M": null,
  "label": null,
  "annotator_notes": ""
}
```

Fill `sentence` and `evidence_text` from actual judge output. Set `label` to exactly `"supported"` or `"unsupported"` (no other values are accepted).

Merge auto-exported records with human-annotated records before calibrating:

```bash
cat eval_data/calibration_records_auto.jsonl \
    eval_data/msa_labeling_template_filled.jsonl \
    > eval_data/calibration_records_merged.jsonl

wc -l eval_data/calibration_records_merged.jsonl
```

---

## Step 7 — Run Confidence Calibration

Post the merged records to `/confidence/calibrate`. The endpoint runs logistic regression (2200 iterations, lr=0.38, L2=0.001) to fit weights `w1`, `w2`, `w3`, `b` for the M/S/A blend.

```bash
# Build payload
python - <<'EOF'
import json, pathlib

records = [json.loads(l) for l in
           pathlib.Path("eval_data/calibration_records_merged.jsonl").read_text().splitlines()
           if l.strip()]

payload = {"records": records}
pathlib.Path("eval_data/_calibrate_payload.json").write_text(json.dumps(payload))
print(f"Records: {len(records)}")
EOF

# Submit
curl -s -X POST http://localhost:8000/confidence/calibrate \
     -H "Content-Type: application/json" \
     -d @eval_data/_calibrate_payload.json \
     | jq .
```

### Expected response

```json
{
  "weights": {"w1": 0.61, "w2": 0.19, "w3": 0.21, "b": -0.08},
  "metrics": {
    "n": 200,
    "accuracy": 0.84,
    "brier": 0.12,
    "method": "gradient_logistic"
  }
}
```

The fitted weights are stored in the `confidence_calibration` table and used for subsequent confidence scoring.

### Minimum record count

The calibration endpoint will succeed with as few as 1 record but is unreliable below ~50 records. Aim for at least 100 records with a roughly balanced label split (30–70% range).

---

## Step 8 — Interpreting Results

### Retrieval quality targets

| Metric              | Acceptable | Good  | Excellent |
|---------------------|-----------|-------|-----------|
| Recall@1            | > 0.55    | > 0.70 | > 0.85   |
| Recall@5            | > 0.75    | > 0.87 | > 0.95   |
| MRR                 | > 0.60    | > 0.75 | > 0.85   |
| nDCG@10             | > 0.65    | > 0.80 | > 0.90   |

Low Recall@1 with decent Recall@10 suggests the re-ranker needs tuning.

### Faithfulness targets

| Metric           | Acceptable | Good  | Excellent |
|------------------|-----------|-------|-----------|
| overall_score    | > 0.65    | > 0.78 | > 0.88   |
| citation_coverage | > 0.60   | > 0.75 | > 0.88   |

### Calibration targets

| Metric   | Target  |
|----------|---------|
| accuracy | > 0.78  |
| brier    | < 0.18  |

---

## Known Mismatches: Paper vs. Code

The midterm report uses language that does not match the current implementation in several places. Use the code-accurate terminology below when writing evaluation results.

| # | Report language | Code reality | Impact |
|---|-----------------|--------------|--------|
| 1 | "Supported / Partially Supported / Not Supported" (3-class) | `supported: bool` in `judge.py` — binary only. There is no "Partially Supported" output. | Do not report a "partially supported" count; the code cannot produce it. |
| 2 | "22% partially supported" | This percentage cannot come from the current code. It may have come from a prior prototype or external annotation. | Remove or qualify this claim. |
| 3 | "relevant chunk IDs manually identified" | `/eval/run` uses only `expected_doc_id` (document-level). Chunk-level relevance is only used in the offline `scripts/eval_retrieval.py` harness, not the API. | If chunk-level eval is desired, use `eval_retrieval.py` with a `relevant_chunk_ids` field. |
| 4 | "faithfulness_score" | The actual response field is `overall_score` inside the `faithfulness` object. | Use `faithfulness.overall_score` in all references. |
| 5 | "per_sentence" | The actual field is `faithfulness.claims` (a list). Individual entries have `sentence_id`, `sentence`, `supported`, `evidence_ids`, `reason`. | Use `claims` not `per_sentence`. |
| 6 | "partially supported" label in calibration | `_judge_label_to_binary()` maps `"moderate"/"medium"` to `0`, not a third class. Only `"supported"` → 1, everything else → 0. | Use only `"supported"` or `"unsupported"` as label values in calibration records. |
| 7 | "alignment tax" / performance degradation numbers for ScholarRAG | No alignment-tax measurement exists in this codebase. This concept is from the InstructGPT paper, not a ScholarRAG measurement. | Attribute clearly as a property of InstructGPT, not a ScholarRAG result. |

---

## File Reference

| File | Purpose |
|------|---------|
| `eval_data/queries_120_master.json` | Master dataset: 120 queries with full metadata, `_doc_key` identifiers, and null placeholder IDs |
| `eval_data/retrieval_eval_cases.json` | 80 cases for `POST /eval/run` — cross-document retrieval evaluation |
| `eval_data/judge_eval_cases.json` | 40 cases for `POST /eval/judge` — single-doc and multi-doc synthesis evaluation |
| `eval_data/msa_labeling_template.jsonl` | 250 annotation slots for human-labeled M/S/A calibration data |
| `scripts/export_msa_records.py` | Converts `/eval/judge` response JSON → calibration JSONL for `POST /confidence/calibrate` |
| `scripts/eval_retrieval.py` | Offline retrieval harness (chunk-level scoring, requires `relevant_chunk_ids`) |
| `docs/EVALUATION.md` | High-level evaluation overview |
| `docs/EVAL_RUNBOOK.md` | This file — step-by-step commands |

---

## Quick Reference: API Constraints

```
POST /eval/run
  scope: "uploaded" only (hard error on any other value)
  required fields per case: query, expected_doc_id
  optional: doc_id, doc_ids (omit to search all documents)

POST /eval/judge
  scope: "uploaded" or "public"
  required fields per case: query
  optional: doc_id (int), doc_ids (list of int), answer, citations
  run_judge_llm: true = LLM judge, false = heuristic fallback

POST /confidence/calibrate
  required: records (list)
  per record: sentence + evidence_text (M computed via NLI)
              OR msa: {M, S, A} (pre-computed)
              + label: "supported" or "unsupported" (required, no other values)
  minimum recommended: 50+ records, roughly balanced labels
```
