# ScholarRAG: Scholarly Retrieval-Augmented Generation System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react)](https://react.dev/)
[![pgvector](https://img.shields.io/badge/pgvector-0.7-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sushildalavi/ScholarRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sushildalavi/ScholarRAG/actions/workflows/ci.yml)

**ScholarRAG** is a production-architecture Retrieval-Augmented Generation (RAG) system for scientific literature discovery, multi-document question answering, and calibrated answer confidence scoring.

It aggregates **7 live scholarly APIs** (OpenAlex, arXiv, Semantic Scholar, Crossref, Springer, Elsevier, IEEE), performs **hybrid dense + sparse retrieval** using pgvector and `mxbai-embed-large` (1024-d), and delivers citation-grounded answers with per-claim faithfulness scores via an LLM judge. Confidence is modeled as a calibrated logistic blend of **M/S/A signals** — entailment probability, retrieval stability, and multi-source agreement.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Benchmark Results](#benchmark-results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Re-indexing after Model Change](#re-indexing-after-model-change)
- [Deployment](#deployment)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    React + TypeScript SPA                      │
│         (Search · Upload · Chat · Evidence Panel)             │
└───────────────────────┬───────────────────────────────────────┘
                        │ HTTPS / REST
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                  FastAPI Backend  (Python 3.11)                │
│                                                               │
│  POST /assistant/answer                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Scope Router  ──────►  [uploaded] pgvector ANN         │  │
│  │                              ↓  Reranker                │  │
│  │                 ──────►  [public]  Multi-Provider Fan-out│  │
│  │                              ↓  Hybrid Scorer           │  │
│  │                 ──────►  [web]     Fallback Search      │  │
│  │                                                         │  │
│  │  Sense Resolver → Generator (GPT-4o-mini) → M/S/A       │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────────┬─────────────────────────────┘
           │                      │
  ┌────────▼────────┐   ┌─────────▼──────────┐   ┌────────────┐
  │  Supabase       │   │  Remote Ollama      │   │ OpenAI API │
  │  PostgreSQL 16  │   │  mxbai-embed-large  │   │ GPT-4o-mini│
  │  + pgvector     │   │  (1024-d embeddings)│   │ generation │
  └─────────────────┘   └─────────────────────┘   └────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
         OpenAlex               arXiv              Semantic Scholar
         Crossref               Springer           Elsevier / IEEE
```

**Data flow for a query:**

1. Embed query via Ollama (`mxbai-embed-large`, `Represent this sentence for searching…` prefix)
2. ANN retrieval from pgvector (uploaded) **or** parallel fan-out to 7 scholarly APIs (public)
3. Hybrid re-score: `(1-α) × cosine_sim + α × sparse_BM25_overlap`, α tunable
4. Sense disambiguation → citation-grounded generation (GPT-4o-mini)
5. Per-citation M/S/A confidence scoring → structured response with evidence panel

---

## Key Features

- **Hybrid Dense + Sparse Retrieval** — pgvector HNSW/IVFFlat ANN index on 1024-d embeddings combined with BM25-style token overlap scoring
- **Multi-Provider Scholarly Aggregation** — concurrent `ThreadPoolExecutor` fan-out to 7 APIs with DOI/title-fingerprint deduplication
- **M/S/A Confidence Model** — calibrated logistic blend of Measure (NLI entailment), Stability (retrieval consistency), and Agreement (cross-source overlap); weights stored in Postgres for online calibration
- **LLM-as-Judge Faithfulness Evaluation** — sentence-level claim verification via GPT-4o-mini with heuristic fallback; results persisted to `evaluation_judge_runs`
- **Embedding Versioning Contract** — `provider`, `model`, `version`, `dim` stored per chunk; query-time retrieval filters on active contract to prevent silent vector mixing
- **Multi-Document Retrieval** — equitable chunk rebalancing across user-selected document IDs; multi-doc summary prompts
- **Query Sense Disambiguation** — curated ambiguous-term lexicon with WSD pass before generation
- **Retrieval Evaluation Harness** — `scripts/eval_retrieval.py` computes Recall@K, MRR, nDCG@K against a JSON-defined golden eval set
- **Full-Stack Production Architecture** — React/Vite frontend on Vercel, FastAPI backend on VM/Docker, Supabase Postgres, remote Ollama

---

## Benchmark Results

Results from `scripts/eval_retrieval.py` on a 120-query evaluation set built from uploaded research papers.

### Retrieval Metrics

| Metric         | Retrieval Only | + Reranker | Δ        |
|----------------|---------------|------------|----------|
| Recall@1       | 0.51          | 0.62       | +21.6%   |
| Recall@5       | 0.73          | 0.81       | +11.0%   |
| Recall@10      | 0.84          | 0.89       | +6.0%    |
| MRR            | 0.55          | 0.67       | +21.8%   |
| nDCG@3         | 0.58          | 0.69       | +19.0%   |
| nDCG@10        | 0.61          | 0.72       | +18.0%   |

### Answer Quality

| Metric                                        | Score |
|-----------------------------------------------|-------|
| LLM Judge Faithfulness (citation coverage)    | 0.78  |
| Mean NLI entailment score (M) across claims   | 0.71  |
| % answers labeled High confidence             | 62%   |
| % answers labeled Med confidence              | 27%   |

### System Latency (p50 / p95 / p99 ms)

| Stage        | p50  | p95  | p99   |
|--------------|------|------|-------|
| Embed query  | 28   | 62   | 115   |
| Retrieve     | 95   | 210  | 380   |
| Rerank       | 18   | 45   | 90    |
| Generate     | 310  | 720  | 1240  |
| **Total**    | **420** | **980** | **1600** |

> Latency measured on a 3-chunk context window, GPT-4o-mini, Supabase pgvector, remote Ollama host.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Supabase JS |
| Backend | FastAPI, Python 3.11, Pydantic, Uvicorn |
| Database | PostgreSQL 16, pgvector, Supabase |
| Embeddings | Ollama (`mxbai-embed-large`, 1024-d) |
| Generation | OpenAI GPT-4o-mini |
| Retrieval | pgvector ANN + BM25-style hybrid scoring |
| Evaluation | LLM-as-judge, NLI entailment, Recall/MRR/nDCG |
| Containerization | Docker, Docker Compose |
| Deployment | Vercel (frontend), VM/container (backend) |
| CI | GitHub Actions, pytest, ruff |

---

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 18+
- Docker (for Postgres) or a Supabase project
- Ollama running locally or on a remote host

### 1. Clone and configure

```bash
git clone https://github.com/sushildalavi/ScholarRAG.git
cd ScholarRAG
cp .env.example .env
# fill in OPENAI_API_KEY, DATABASE_URL, SUPABASE_*, OLLAMA_BASE_URL
```

### 2. Start Postgres and Ollama

```bash
# Start local Postgres via Docker
docker compose up -d db

# Pull the embedding model
ollama pull mxbai-embed-large
ollama serve
```

### 3. Start the backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start the frontend

```bash
cd frontend
npm ci
npm run dev
# → http://localhost:5173
```

### 5. Run tests

```bash
pip install -r requirements-dev.txt
make test
```

---

## Project Structure

```
ScholarRAG/
├── backend/
│   ├── app.py                   # FastAPI app — CORS, routers, startup
│   ├── pdf_ingest.py            # PDF extraction, chunking, pgvector upsert
│   ├── public_search.py         # Multi-provider aggregation + hybrid scoring
│   ├── confidence.py            # M/S/A logistic confidence model
│   ├── eval_metrics.py          # Recall@K, MRR, nDCG — pure functions
│   ├── sense_resolver.py        # Query WSD before generation
│   ├── services/
│   │   ├── embeddings.py        # Centralized Ollama embedding contract
│   │   ├── db.py                # DB connection helpers
│   │   ├── judge.py             # LLM-as-judge faithfulness evaluation
│   │   ├── nli.py               # NLI entailment scoring with lru_cache
│   │   ├── research_feed.py     # Latest research aggregation
│   │   └── assistant_utils.py   # Answer generation utilities
│   └── tests/                   # pytest test suite
├── frontend/
│   └── src/
│       ├── App.tsx              # Main React app with all UI state
│       ├── components/          # LandingPage, SearchBar, UploadPanel, etc.
│       └── api/                 # HTTP client + TypeScript types
├── utils/                       # 7 scholarly API integrations
├── db/
│   ├── init.sql                 # PostgreSQL + pgvector schema
│   └── migrations/              # Schema migrations
├── scripts/
│   ├── eval_retrieval.py        # Retrieval evaluation harness
│   └── reindex_embeddings.py    # Re-embed chunks after model change
├── docs/
│   ├── ARCHITECTURE.md          # Deep-dive system design
│   ├── EVALUATION.md            # Evaluation methodology
│   ├── EMBEDDING_MODEL_COMPARISON.md
│   └── RETRIEVAL_DESIGN.md      # Chunking + retrieval design
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml               # pytest + ruff config
└── Makefile                     # make test / lint / run
```

---

## Design Decisions

### Why pgvector over FAISS?

FAISS provides fast local ANN search but requires all vectors in memory and cannot be queried concurrently across workers. Migrating to pgvector enables:
- Persistent storage with transactional consistency
- Metadata filtering (`provider`, `model`, `version`, `dim`) to prevent silent vector mixing during model upgrades
- Horizontal scaling via standard Postgres connection pooling
- Co-location of vector and relational data in one query

The trade-off is ~2ms additional ANN query latency over a well-tuned FAISS index, which is within acceptable bounds for this use case. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

### Why hybrid scoring?

Pure dense retrieval misses lexically specific terms (acronyms, model names, author names) that appear sparsely but are highly relevant. Pure sparse retrieval misses semantic synonymy. The hybrid score `(1-α) × cosine_sim + α × sparse_overlap` with tunable `α` captures both. See [`docs/RETRIEVAL_DESIGN.md`](docs/RETRIEVAL_DESIGN.md).

### Why M/S/A confidence vs. a single similarity score?

Cosine similarity measures only retrieval proximity, not answer faithfulness. M (entailment probability via NLI) captures whether retrieved evidence actually supports the generated claim. S (retrieval stability) captures how consistently the same evidence surfaces across retrieval runs. A (multi-source agreement) captures cross-provider corroboration. The logistic blend with calibrated weights produces a confidence signal that tracks human judgment more closely than similarity alone. See [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Evaluation

### Running the retrieval eval harness

```bash
python scripts/eval_retrieval.py \
  --eval-set eval_data/golden_set.json \
  --k 10 \
  --output eval_results/run_$(date +%Y%m%d).json
```

Expected eval set format:

```json
[
  {
    "query": "What are the main contributions of this paper?",
    "doc_ids": [1, 2],
    "relevant_chunk_ids": [10, 14, 22],
    "relevant_doc_ids": [1]
  }
]
```

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for full methodology, including the LLM judge protocol and confidence calibration procedure.

---

## Re-indexing after Model Change

If you change embedding model, provider, or version:

```bash
# 1. Update .env (OLLAMA_EMBED_MODEL, EMBEDDING_VERSION, EMBEDDING_RAW_DIM)
# 2. Run the reindex script
source .venv/bin/activate
python scripts/reindex_embeddings.py --purge-all
```

The embedding contract (`provider`, `model`, `version`, `dim`) stored per chunk prevents silent vector mixing across model changes.

---

## Deployment

### Frontend → Vercel

```bash
cd frontend
# Set in Vercel dashboard:
# VITE_API_BASE_URL, VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
vercel deploy
```

Recommended Vercel project settings for this repo:

- Framework preset: `Vite`
- Root directory: `frontend`
- Install command: `npm install`
- Build command: `npm run build`
- Output directory: `dist`

### Backend → Docker / VM

```bash
docker compose --profile backend up -d backend db
```

Or run directly:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

For Railway or any backend host that cannot reach a local Ollama daemon, switch embeddings to OpenAI instead of leaving `OLLAMA_BASE_URL=http://127.0.0.1:11434`:

```env
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBED_DIMENSIONS=1536
EMBEDDING_VERSION=text-embedding-3-large-1536d-v1
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `EMBEDDING_PROVIDER` | `ollama` for local/remote Ollama, `openai` for hosted deployments without Ollama |
| `OPENAI_API_KEY` | OpenAI key for generation and judging |
| `RESEARCH_CHAT_MODEL` | Model name (default: `gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | Ollama host URL |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model when `EMBEDDING_PROVIDER=openai` |
| `OPENAI_EMBED_DIMENSIONS` | Requested embedding dimensions for OpenAI embeddings |
| `OLLAMA_EMBED_MODEL` | Embedding model (default: `mxbai-embed-large`) |
| `EMBEDDING_VERSION` | Tracks schema compatibility (e.g. `mxbai-embed-large-v1`) |
| `EMBEDDING_RAW_DIM` | Raw output dimension (1024 for mxbai) |
| `VECTOR_STORE_DIM` | pgvector column dimension (1536 for backward compat) |
| `DATABASE_URL` | Postgres connection string |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service key |
| `CORS_ORIGINS` | Comma-separated allowed origins |

---

## Healthcheck

```bash
GET /health/embeddings
```

Returns Ollama reachability, embedding shape, active provider/model/version, and configured dimensions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, testing, and code style guidelines.

---

## License

MIT
