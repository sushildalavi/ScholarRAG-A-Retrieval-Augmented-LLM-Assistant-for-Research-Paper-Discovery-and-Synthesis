# ScholarRAG

ScholarRAG is a retrieval-augmented research assistant for:
- uploaded PDF question answering
- public scholarly search aggregation
- citation-grounded answer generation
- confidence-aware and judge-aware responses

Current production direction:
- frontend on `Vercel`
- backend as hosted `FastAPI`
- database / pgvector / storage via `Supabase`
- embeddings via remote or local `Ollama`
- generation and judging via `OpenAI`

## Architecture summary

### Runtime architecture

- `frontend/`
  - React + Vite + TypeScript SPA
  - upload, multi-doc selection, chat, evidence panel, evaluation views

- `backend/app.py`
  - main API orchestration
  - routes uploaded-doc and public-search questions
  - builds grounded answer responses
  - exposes retrieval, confidence, judge, and health outputs

- `backend/pdf_ingest.py`
  - PDF extraction
  - chunking
  - chunk embedding insertion
  - uploaded-document vector retrieval from Postgres / pgvector

- `backend/public_search.py`
  - multi-provider scholarly aggregation
  - provider contribution tracking
  - hybrid scoring over public candidates

- `backend/services/embeddings.py`
  - centralized embedding contract
  - query and document embedding via Ollama HTTP
  - retries, timeouts, schema validation, healthcheck

- `backend/services/db.py`
  - database connection helpers

- `db/init.sql`
  - Postgres / pgvector schema
  - documents, chunks, chunk embeddings, chat, eval, confidence tables

## Embedding architecture

ScholarRAG now uses one centralized embedding service:

- provider: `ollama`
- model: `mxbai-embed-large`
- query prefix: configurable
- document prefix: configurable
- batching: configurable
- remote host: configurable

Important:
- embeddings are generated through an Ollama-compatible HTTP endpoint
- the backend never assumes Ollama runs inside Vercel
- production should point `OLLAMA_BASE_URL` at a separate host or VM

### Current vector storage contract

The active embedding model is `mxbai-embed-large`.

Its raw output dimension is tracked separately from the vector-store dimension:
- raw embedding dimension: `1024`
- current pgvector storage dimension: `1536`

The embedding service pads vectors to the configured store dimension so the current schema remains compatible while avoiding a dangerous full vector-column rewrite during this refactor.

To avoid accidental vector mixing, chunk embeddings now store:
- `provider`
- `model`
- `embedding_version`
- `dim`

Uploaded retrieval filters on the current embedding contract, so query-time retrieval only uses vectors from the active provider/model/version.

## Public scholarly aggregation flow

When a public query is given:

1. normalize query
2. query enabled providers
3. deduplicate candidates
4. embed candidates and compute hybrid score
5. rerank merged results
6. build one final result set
7. generate grounded answer from those results

Current providers wired:
- OpenAlex
- arXiv
- Crossref
- Semantic Scholar
- Springer
- Elsevier
- IEEE

Provider contribution status is tracked and returned in response metadata so you can see which APIs contributed to the final public answer.

If a provider is misconfigured or unauthorized, the aggregator continues using the providers that succeed.

## Uploaded-document flow

1. upload PDF
2. extract page-aware text
3. chunk text into retrieval-friendly segments
4. embed chunks with the centralized embedding service
5. store document, chunks, and chunk embeddings in Postgres
6. retrieve relevant chunks for selected docs at query time
7. rerank and build grounded answer context
8. generate answer with citations
9. attach confidence and optional judge output

For multi-document retrieval:
- selected document ids are passed explicitly
- retrieval is rebalanced across selected docs
- multi-doc summary prompts are phrased differently from single-doc prompts

## Confidence / judge / evaluation

ScholarRAG includes:
- citation-grounded answers
- confidence objects
- M/S/A-style evidence score tables
- judge run storage
- calibration storage

### Evaluation hooks

Added scripts:
- `scripts/reindex_embeddings.py`
  - rebuild chunk embeddings for the active provider/model/version
- `scripts/eval_retrieval.py`
  - evaluate uploaded-doc retrieval with:
    - `Recall@5`
    - `Recall@10`
    - `MRR`
    - `nDCG@10`

Expected eval set format:

```json
[
  {
    "query": "What skills are listed in the resume?",
    "doc_ids": [1, 2],
    "relevant_chunk_ids": [10, 14],
    "relevant_doc_ids": [1]
  }
]
```

This is intentionally lightweight so you can compare embedding models later, including `mxbai-embed-large` vs `text-embedding-3-large`.

## Environment configuration

Copy:

```bash
cp .env.example .env
```

Key variables:

```env
OPENAI_API_KEY=
RESEARCH_CHAT_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=mxbai-embed-large
EMBEDDING_QUERY_PREFIX=Represent this sentence for searching relevant passages:
EMBEDDING_DOC_PREFIX=Represent this document for retrieval:
EMBEDDING_BATCH_SIZE=16
EMBEDDING_VERSION=mxbai-embed-large-v1
EMBEDDING_RAW_DIM=1024
VECTOR_STORE_DIM=1536

DATABASE_URL=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_JWT_SECRET=
```

## Local development

### 1. Start Ollama

```bash
ollama serve
ollama pull mxbai-embed-large
```

### 2. Start Postgres locally or use Supabase

Local Docker option:

```bash
docker compose up -d db adminer
```

### 3. Start backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start frontend

```bash
cd frontend
npm ci
npm run dev
```

## Re-indexing after model change

If you change embedding model, provider, or version:

1. update `.env`
2. run schema migration if needed
3. rebuild embeddings

```bash
source .venv/bin/activate
python scripts/reindex_embeddings.py --purge-all
```

This prevents silent mixing of embeddings produced by different models.

## Healthcheck

Backend endpoint:

```text
GET /health/embeddings
```

It verifies:
- Ollama endpoint reachability
- embedding response shape
- active provider/model/version
- configured vector dimensions

## Production deployment notes

### Frontend

- deploy `frontend/` to `Vercel`
- set:
  - `VITE_API_BASE_URL`
  - `VITE_SUPABASE_URL`
  - `VITE_SUPABASE_ANON_KEY`

### Backend

Deploy FastAPI separately from Vercel functions for production safety.

Reason:
- Vercel Functions are not a safe place to assume local Ollama access
- the backend needs stable network access to:
  - remote Ollama
  - Supabase Postgres
  - public scholarly APIs
  - OpenAI

Recommended:
- host backend on a VM / container platform
- point `OLLAMA_BASE_URL` to a remote Ollama host

### Supabase

Use Supabase for:
- Postgres
- pgvector storage
- auth
- storage integration

### Remote Ollama

Production must use a separate Ollama-compatible host.

Example:

```env
OLLAMA_BASE_URL=https://your-ollama-host.example.com
OLLAMA_EMBED_MODEL=mxbai-embed-large
```

Do not deploy Ollama inside Vercel.

## Docker usage notes

Docker is kept for:
- local Postgres consistency
- optional local/backend container runtime
- optional self-hosted backend deployment
- optional Ollama co-hosting on a VM

Docker is not used for:
- Vercel frontend runtime
- pretending Ollama exists inside Vercel Functions

Useful commands:

```bash
docker compose --profile ollama up -d ollama
docker compose --profile backend up -d backend db
```

## Removed legacy pieces

The active runtime no longer depends on FAISS.

Removed from the repository:
- legacy FAISS retriever runtime path
- legacy per-user FAISS store helpers
- legacy FAISS index builder scripts

This refactor keeps the project aligned with a Supabase/Postgres + remote-Ollama production architecture.
