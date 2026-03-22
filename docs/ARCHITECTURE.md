# ScholarRAG — System Architecture

## 1. Overview

ScholarRAG solves three distinct but related problems:

1. **Document Q&A**: Given a set of user-uploaded PDFs and a natural-language question, retrieve the most relevant passages and generate a citation-grounded answer with calibrated confidence.
2. **Public Literature Discovery**: Given a research query, aggregate results from 7 live scholarly APIs, deduplicate, re-score with a hybrid dense+sparse method, and synthesize an answer from the top results.
3. **Answer Faithfulness Evaluation**: Given a generated answer and its citations, measure whether the answer is actually supported by the retrieved evidence (LLM-as-judge + NLI entailment).

The system is intentionally split into a stateless React SPA (frontend) and a stateful FastAPI service (backend) that owns all ML inference and database interactions.

---

## 2. Component Map

| Module | File | Responsibility |
|--------|------|----------------|
| **API Orchestration** | `backend/app.py` | FastAPI app setup, CORS, router includes, startup schema migrations |
| **RAG Core** | `backend/app.py` (`/assistant/answer`) | Query routing, retrieval dispatch, context building, generation, confidence assembly |
| **PDF Ingestion** | `backend/pdf_ingest.py` | PDF text extraction (page-aware), structure-aware chunking, batch embedding, pgvector upsert |
| **Public Aggregation** | `backend/public_search.py` | ThreadPoolExecutor fan-out to 7 APIs, hybrid scoring, deduplication |
| **Embedding Service** | `backend/services/embeddings.py` | Centralized Ollama HTTP contract, batching, retries, cache, versioning |
| **NLI Service** | `backend/services/nli.py` | LLM-based NLI entailment scoring with `lru_cache(maxsize=2000)` |
| **Judge Service** | `backend/services/judge.py` | LLM-as-judge faithfulness evaluation; heuristic fallback |
| **Confidence Model** | `backend/confidence.py` | M/S/A logistic blend; calibration weight injection from Postgres |
| **Eval Metrics** | `backend/eval_metrics.py` | Pure-function Recall@K, MRR, nDCG@K |
| **Sense Resolver** | `backend/sense_resolver.py` | Query WSD using curated ambiguous-term lexicon |
| **Research Feed** | `backend/services/research_feed.py` | Latest research aggregation with TTL cache |
| **DB Layer** | `backend/services/db.py` | Connection pooling, `fetchall`, `fetchone`, `execute` helpers |
| **Answer Utils** | `backend/services/assistant_utils.py` | 40+ helpers for context building, citation pruning, fallback detection |
| **Auth** | `backend/auth.py` | Supabase JWT validation middleware |
| **Chat** | `backend/chat.py` | Session management, message history persistence |
| **Frontend App** | `frontend/src/App.tsx` | React state machine: upload, document selection, query, evidence panel |
| **API Client** | `frontend/src/api/client.ts` | Typed HTTP client for all backend endpoints |

---

## 3. Data Model

### Core Tables (PostgreSQL 16 + pgvector)

```sql
-- Uploaded documents
documents (
  id SERIAL PRIMARY KEY,
  title TEXT,
  path TEXT,
  hash TEXT UNIQUE,       -- SHA-256 for deduplication
  status TEXT,            -- 'indexed' | 'processing' | 'error'
  doc_type TEXT,
  created_at TIMESTAMP
)

-- Text chunks extracted from documents
chunks (
  id SERIAL PRIMARY KEY,
  doc_id INT REFERENCES documents(id),
  content TEXT,
  page_number INT,
  heading_path TEXT,      -- e.g. "2 > Background > 2.1 Related Work"
  token_count INT
)

-- Versioned embeddings per chunk
chunk_embeddings (
  id SERIAL PRIMARY KEY,
  chunk_id INT REFERENCES chunks(id),
  embedding VECTOR(1536), -- padded from 1024-d raw
  provider TEXT,          -- 'ollama'
  model TEXT,             -- 'mxbai-embed-large'
  embedding_version TEXT, -- 'mxbai-embed-large-v1'
  dim INT                 -- 1024 (raw), stored at 1536
)

-- Embedding cache (query + document, keyed by hash)
embedding_cache (
  id SERIAL PRIMARY KEY,
  cache_key TEXT UNIQUE,
  embedding VECTOR(1536),
  embedding_type TEXT,    -- 'query' | 'document'
  provider TEXT,
  model TEXT,
  version TEXT
)

-- M/S/A calibration weights
confidence_calibration (
  id SERIAL PRIMARY KEY,
  model_name TEXT,        -- 'msa_logistic_v1'
  label TEXT,             -- 'default'
  weights JSONB,          -- {"b": 0.0, "w1": 0.58, "w2": 0.22, "w3": 0.20}
  metrics JSONB,
  dataset_size INT
)

-- Per-citation evidence scores
evidence_scores (
  id SERIAL PRIMARY KEY,
  request_id TEXT,
  citation_id INT,
  m_score REAL,           -- entailment probability
  s_score REAL,           -- retrieval stability
  a_score REAL,           -- multi-source agreement
  score REAL              -- blended final score
)

-- Retrieval evaluation runs
eval_runs (
  id SERIAL PRIMARY KEY,
  name TEXT,
  scope TEXT,
  k INT,
  case_count INT,
  metrics_retrieval_only JSONB,
  metrics_retrieval_rerank JSONB,
  latency_breakdown JSONB,
  details JSONB
)

-- Judge evaluation runs
evaluation_judge_runs (
  id SERIAL PRIMARY KEY,
  scope TEXT,
  query_count INT,
  metrics JSONB,
  details JSONB
)
```

### Key Design Decisions in the Schema

**Embedding versioning**: The `chunk_embeddings` table stores `provider`, `model`, `embedding_version`, and `dim` on every row. At query time, retrieval filters on the *active* contract (from `EMBEDDING_VERSION` env var). This prevents silent vector mixing when you switch from `mxbai-embed-large-v1` to any future model — old embeddings are queryable but not mixed with new ones unless an explicit reindex migration runs.

**Dimension padding**: `mxbai-embed-large` outputs 1024-d vectors. The pgvector column is 1536-d for backward compatibility with an earlier `text-embedding-ada-002` schema. The embedding service zero-pads the 1024-d output to 1536-d before storage. A future migration to native 1024-d storage would reduce index size and query latency by ~33%.

---

## 4. Retrieval Pipeline

### Uploaded-Document Path

```
Query string
    │
    ▼
embed_query(query, prefix="Represent this sentence for searching…")
    │
    ▼
pgvector ANN (cosine distance, top-k=10 per selected doc_id)
    │
    ▼
Reranker: chunk_query_overlap_score (semantic + token overlap)
    │
    ▼
Rebalance across doc_ids (equitable distribution)
    │
    ▼
Build citation-grounded context string
    │
    ▼
GPT-4o-mini generation with system prompt
    │
    ▼
M/S/A confidence scoring → evidence panel
```

### Public Search Path

```
Query string
    │
    ▼
normalize_public_query() → keyword variants
    │
    ▼ (concurrent ThreadPoolExecutor)
┌───────────┬──────────┬────────────┬───────────┬──────────┬──────────┬──────┐
OpenAlex  arXiv    Crossref  SemanticScholar  Springer  Elsevier  IEEE
(15)      (15)     (10)      (10)              (10)      (10)      (10)
└───────────┴──────────┴────────────┴───────────┴──────────┴──────────┴──────┘
    │
    ▼
Deduplicate by DOI fingerprint + title normalization (~80 → ~40 candidates)
    │
    ▼
embed_documents(candidates) → VECTOR(1536)
    │
    ▼
Hybrid Score: (1-α) × cosine_sim(query_vec, candidate_vec) + α × sparse_overlap
    │
    ▼
Top-K selection, provider contribution tracking
    │
    ▼
GPT-4o-mini generation
    │
    ▼
M/S/A confidence scoring → evidence panel
```

### Fallback Cascade

If uploaded retrieval yields insufficient evidence (coverage < 20%), the system can fall back to:
1. Web search (`backend/public_web.py`) if `ENABLE_WEB_FALLBACK=true`
2. Templated response with scope explanation

---

## 5. Confidence Model

The `build_confidence()` function in `backend/confidence.py` operates in two stages:

**Stage 1 — Retrieval-based score:**
```
base = 0.35 × top_sim + 0.25 × top_rerank_norm + 0.25 × citation_coverage + 0.15 × evidence_margin
score = clamp(base − 0.45 × ambiguity_pen − 0.40 × insufficiency_pen − 0.50 × scope_pen)
```

**Stage 2 — M/S/A blend (when NLI + stability + agreement are available):**
```
msa_score = sigmoid(b + w1×M + w2×S + w3×A)
  default weights: b=0.0, w1=0.58, w2=0.22, w3=0.20

final_score = clamp(0.62 × stage1_score + 0.38 × msa_score)
```

Weights are loaded from the `confidence_calibration` Postgres table at request time, enabling online calibration without redeployment.

**Labels:** High (≥0.75), Med (≥0.50), Low (<0.50)

---

## 6. Scaling Considerations and Known Limitations

| Component | Current State | Production Path |
|-----------|--------------|-----------------|
| `lru_cache` in `nli.py` | In-process, 2000-entry cap; lost on restart | Redis-backed distributed cache |
| `_PUBLIC_SEARCH_CACHE` dict | In-process, TTL-evicted; not shared across workers | Postgres table or Redis |
| Dimension padding (1024→1536) | Zero-pad; ~33% wasted index space | Migrate pgvector column to VECTOR(1024) |
| NLI entailment | LLM API call per claim (expensive, ~300ms) | Fine-tuned DeBERTa-NLI cross-encoder |
| Ollama embedding | Single host; no load balancing | Ollama cluster or replicated service |
| Retrieval stability score (S) | Proxy via repeated retrieval consistency check | True bootstrap re-run across query perturbations |
| Auth | Supabase JWT validation | Compatible with standard OAuth2 / OIDC providers |

---

## 7. External API Dependencies

| Provider | Module | Notes |
|----------|--------|-------|
| OpenAlex | `utils/openalex_utils.py` | Free, no key required; 15 results/query |
| arXiv | `utils/arxiv_utils.py` | Free, no key; 15 results/query |
| Crossref | `utils/crossref_utils.py` | Free; DOI-centric, best for citation resolution |
| Semantic Scholar | `utils/semanticscholar_utils.py` | Key optional; 10 results/query |
| Springer | `utils/springer_utils.py` | Key required; 10 results/query |
| Elsevier | `utils/elsevier_utils.py` | Institutional key required; 10 results/query |
| IEEE Xplore | `utils/ieee_utils.py` | Key required; 10 results/query |
| OpenAI | `openai` SDK | GPT-4o-mini for generation, judging, NLI |
| Ollama | HTTP (`services/embeddings.py`) | `mxbai-embed-large` for all embeddings |
