# ScholarRAG

ScholarRAG is a Retrieval-Augmented Generation (RAG) assistant for research paper discovery and synthesis.

It combines:
- FastAPI backend for retrieval, ingestion, and answer generation
- React (Vite + TypeScript) frontend for document upload and conversational Q&A
- PostgreSQL + pgvector for chunk/document storage and vector search
- FAISS + metadata bootstrap for global paper retrieval

## Architecture

### Backend (`backend/`)
- `backend/app.py`: main API entrypoint and route wiring
- `backend/pdf_ingest.py`: upload, parse, chunk, embed, and document status tracking
- `backend/chat.py`: chat sessions and chat upload routes
- `backend/retriever.py`: retrieval and reranking pipeline
- `backend/public_search.py`: live public source retrieval
- `backend/services/db.py`: DB access layer
- `backend/services/embeddings.py`: embedding generation + caching

### Frontend (`frontend/`)
- React app with document upload, doc-scoped QA, and assistant panel
- API base URL via `VITE_API_BASE` (defaults to `http://127.0.0.1:8000`)

### Data and Storage
- `data/scholar_index.faiss`: global FAISS index
- `data/metadata.json`: metadata aligned to FAISS entries
- `data/user_indices/`: per-user indices (runtime)
- `storage/`: uploaded files (runtime)
- `logs/`: request/retrieval logs (runtime)

## Requirements

- Python 3.10+
- Node.js 18+
- Docker Desktop (for local PostgreSQL)

## Environment Configuration

Create local env file:

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY`
- `RESEARCH_CHAT_MODEL` (default: `gpt-4o-mini`)

Optional provider keys:
- `OPENALEX_API_KEY`
- `SPRINGER_API_KEY`
- `IEEE_API_KEY`
- `ELSEVIER_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`

Common optional tuning:
- `PUBLIC_OPENALEX_LIMIT`, `PUBLIC_ARXIV_LIMIT`, `PUBLIC_CROSSREF_LIMIT`, `PUBLIC_S2_LIMIT`
- `PUBLIC_SPRINGER_LIMIT`, `PUBLIC_ELSEVIER_LIMIT`, `PUBLIC_IEEE_LIMIT`
- `ARXIV_API_URL` (defaults to `https://export.arxiv.org/api/query`)
- `ENABLE_WEB_FALLBACK` (default: `false`, keeps research retrieval on standard scholarly sources)
- `SPRINGER_META_VERSION` (`v2` default; falls back to legacy metadata endpoint)

For full supported environment variables, see `.env.example`.

## Local Development

### 1) Start PostgreSQL + Adminer

```bash
docker compose up -d db adminer
```

- PostgreSQL: `127.0.0.1:5432`
- Adminer: `http://127.0.0.1:8080`

### 2) Start Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary pgvector python-dotenv requests openai pydantic PyPDF2 python-multipart faiss-cpu numpy google-auth
uvicorn backend.app:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
```

Backend docs:
- `http://127.0.0.1:8000/docs`

### 3) Start Frontend

```bash
cd frontend
cp .env.example .env 2>/dev/null || true
npm ci
npm run dev
```

Frontend:
- `http://127.0.0.1:5173`

## Production Notes

- Keep `.env` out of git.
- Do not commit runtime/generated artifacts:
  - `.venv/`, `node_modules/`, `frontend/dist/`, `frontend/.vite/`, `__pycache__/`, `storage/`, `logs/`
- Rotate API keys immediately if exposed.
- Keep CI checks under `.github/workflows/` for quality gates.

## Useful Commands

From repo root:

```bash
# Python syntax sanity
python -m py_compile $(find . -name '*.py' -not -path './.venv/*' -not -path './frontend/node_modules/*')

# Frontend lint/build
cd frontend && npm run lint && npm run build
```

## API Surface (high-level)

- `GET /` health message
- `POST /assistant/answer` unified assistant endpoint
- `POST /assistant/chat` chat endpoint
- `POST /documents/upload` upload and ingest document
- `POST /documents/search/chunks` chunk retrieval
- `GET /documents/latest` latest uploaded docs
- `DELETE /documents/{doc_id}` delete uploaded doc
- `POST /eval/run` run retrieval-only vs retrieval+rerank evaluation
- `GET /eval/runs` list stored eval runs

## Confidence Definition

ScholarRAG now returns a structured confidence object for each answer and source card:

```json
{
  "score": 0.0,
  "label": "Low|Med|High",
  "needs_clarification": false,
  "factors": {
    "top_sim": 0.0,
    "top_rerank_norm": 0.0,
    "citation_coverage": 0.0,
    "evidence_margin": 0.0,
    "ambiguity_penalty": 0.0,
    "insufficiency_penalty": 0.0
  },
  "explanation": "..."
}
```

Interpretation:
- `top_sim`: top chunk relevance signal (normalized)
- `top_rerank_norm`: top reranker normalized score
- `citation_coverage`: fraction of answer paragraphs that include citations
- `evidence_margin`: separation between top evidence and runner-up evidence
- `ambiguity_penalty`: reduction due to unresolved or compare-mode ambiguity
- `insufficiency_penalty`: reduction when evidence is missing/unsupported

UI behavior:
- Confidence badge shows `High/Med/Low + %`
- Hover tooltip explains factor breakdown and scoring meaning

## “Why This Answer?” Trace

Each assistant answer includes a trace payload (`why_answer`) and the UI renders it below the message:
- top retrieved chunks (title, chunk id, page, snippet preview)
- similarity and rerank score per chunk
  - `cosine` = similarity signal (raw)
  - `rerank_raw` = raw reranker score
  - `rerank_norm` = normalized reranker score for display
- whether the chunk was actually cited
- rank delta (`before` vs `after`) when reranking changed order

This keeps responses doc-grounded and auditable.

## Sense Resolver (Ambiguity Gate)

Before generating an answer, ScholarRAG runs a sense-resolution step:
- detects curated ambiguous terms (e.g., `transformer`, `python`, `java`, `rust`, `react`, etc.)
- inspects top evidence for mixed-sense signals
- if ambiguous and user has not selected a sense, returns clarification instead of an answer

Example clarification:
- “Do you mean ML Transformer models, or Electrical power transformers?”

The frontend renders clickable options (chips). Once selected, retrieval is constrained to the chosen sense.

Compare mode:
- If explicitly enabled, the model may answer in separate sense sections, still citation-grounded.

Endpoint:
- `POST /assistant/resolve_sense` (query + optional sense)

## Grounding Policy (Strict)

- Default mode is doc-grounded (`Docs only`).
- General background is disabled unless user enables `Allow general background`.
- Every claim paragraph/bullet must have citation support.
- If unsupported paragraphs are detected, answer is blocked and the assistant asks for clarification or more evidence.
- For ambiguous terms, answer generation is blocked until sense is clarified (unless compare mode is explicitly enabled).

## Entity-Scope Aware Retrieval Policy

To prevent resume-based hallucinated company summaries, ScholarRAG now enforces scope checks:

1. `doc_type` metadata on uploaded documents:
- `resume | research_paper | official_doc | assignment | notes | other`
- inferred from filename at upload time
- can be overridden via `PUT /documents/{doc_id}/type`

2. Entity-level query detection:
- patterns like `tell me about X`, `what is X`, `X company`, `X irvine`
- excludes personal-role intent (`worked`, `experience`, `intern`, etc.)

3. Scope guard before generation:
- if entity-level query + retrieved evidence is only personal/resume context + no official/company docs exist,
- assistant does **not** generate company-level overview,
- returns context-limited clarification prompt instead.

4. Confidence alignment:
- applies `scope_penalty` when entity-level answers rely on personal-only context
- may return `Context-limited` label / `Needs clarification`.

5. UI scope indicator:
- answer shows scope tags such as `personal_document_context` or `official_document_context`.

## Evaluation Methodology (`/eval`)

Open `http://127.0.0.1:5173/eval` to run retrieval evaluations with a test set:
- Input format: array of objects with `query` and `expected_doc_id`
- Pipelines compared:
1. retrieval-only (raw vector order)
2. retrieval + rerank

Computed metrics:
- Recall@K for `K = 1, 3, 5, 10`
- MRR
- nDCG@K for `K = 3, 5, 10`
- Latency breakdown: retrieve, rerank, generate

Runs are stored in DB table `eval_runs` and listed via `GET /eval/runs`.

## Safety / Grounding Policy

- Default behavior remains evidence-first.
- If evidence is insufficient, the assistant explicitly says so and avoids broad unsupported claims.
- The answer text is phrased naturally, but scope limitations are preserved and cited.

## License

Add your preferred license file (`LICENSE`) if this repository is intended for public reuse.
