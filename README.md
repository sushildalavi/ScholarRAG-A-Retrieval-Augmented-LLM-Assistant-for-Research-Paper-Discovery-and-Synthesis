# ScholarRAG

ScholarRAG is a retrieval-augmented research assistant for research paper discovery, uploaded-document question answering, citation-grounded responses, and confidence-aware answer generation.

It currently combines:
- `FastAPI` backend for ingestion, retrieval, reranking, grounding, confidence scoring, and judge evaluation
- `React + Vite + TypeScript` frontend for document upload, multi-document chat, evidence inspection, and evaluation views
- `Postgres + pgvector` for uploaded document chunks, embeddings, evaluation runs, and confidence artifacts
- `FAISS` bootstrap indexing for the current local paper-index path
- `Ollama` for embeddings
- `OpenAI` for answer generation and judge evaluation

## Current status

This repository is in a hybrid state:

- Implemented now:
  - uploaded PDF ingestion
  - chunking + embedding
  - Postgres-backed uploaded-document retrieval
  - citation-grounded answering
  - confidence metadata
  - judge / calibration tables
  - multi-provider public retrieval aggregation
  - local frontend/backend development flow

- Still present from the earlier local architecture:
  - FAISS-based paper index bootstrap
  - local runtime artifact folders such as `data/`, `storage/`, and `logs/`

- Deployment target being implemented:
  - frontend on `Vercel`
  - backend as hosted `FastAPI`
  - cloud database/storage via `Supabase`
  - `Ollama` embeddings + `OpenAI` generation/judge

Do not treat the project as fully migrated away from FAISS yet. The codebase is currently a mixed local/cloud architecture with a clear path toward a more production-oriented Supabase/Postgres deployment.

## Architecture

### Current implemented architecture

- `frontend/`
  - React SPA for upload, chat, evidence panel, and evaluation screens

- `backend/`
  - `backend/app.py`: primary API entrypoint and orchestration
  - `backend/pdf_ingest.py`: PDF ingestion, chunking, embedding storage, uploaded-doc search
  - `backend/public_search.py`: public scholarly source aggregation and reranking
  - `backend/chat.py`: chat session and upload routes
  - `backend/services/db.py`: DB connection and SQL helpers
  - `backend/services/embeddings.py`: OpenAI/Ollama embedding abstraction
  - `backend/services/judge.py`: answer faithfulness / judge logic
  - `backend/confidence.py`: confidence scoring utilities

- `db/init.sql`
  - base schema for:
    - `documents`
    - `chunks`
    - `chunk_embeddings`
    - `chat_sessions`
    - `chat_messages`
    - `eval_runs`
    - `confidence_calibration`
    - `evidence_scores`
    - `evaluation_judge_runs`

### Target deployment architecture

- Frontend: `Vercel`
- Backend: hosted `FastAPI`
- Database: `Supabase Postgres`
- Storage: `Supabase Storage`
- Embeddings: `Ollama`
- Generation: `OpenAI`
- Judge evaluation: `OpenAI`
- Retrieval: uploaded-doc Postgres retrieval + public multi-provider aggregation

## Public retrieval aggregation flow

When a public query is given, ScholarRAG is designed to aggregate results across multiple scholarly APIs and return one merged final result set.

Current aggregation path in `backend/public_search.py`:

- `OpenAlex`
- `arXiv`
- `Crossref`
- `Semantic Scholar`
- `Springer`
- `Elsevier`
- `IEEE`

Flow:

1. Normalize the user query into a search-friendly form.
2. Fetch candidate records from all enabled providers.
3. Deduplicate records by DOI / provider ID / title.
4. Compute:
   - dense similarity using embeddings
   - sparse lexical overlap score
5. Build a hybrid score.
6. Return one reranked merged list as the final public retrieval result.

This is the correct high-level flow for your project.

Important implementation note:

- Not every configured provider is currently operational in your environment.
- `Elsevier` and `IEEE` are currently failing authorization in your setup.
- The aggregator still works because the final result is built from whichever providers successfully return candidates.

So yes, keeping the “aggregate across APIs, merge, rerank, and return one final result” flow is the right design.

## Uploaded-document flow

1. User uploads one or more PDFs.
2. Backend extracts text and splits it into chunks.
3. Chunks are embedded using the configured embedding provider.
4. Chunks and vectors are stored in Postgres.
5. User asks a question over selected documents.
6. Backend retrieves the most relevant chunk evidence from the selected docs.
7. Evidence is reranked and passed into answer generation.
8. Answer is returned with citations, confidence metadata, and optional judge results.

## Multi-document QA flow

For multi-document questions, the intended behavior is:

1. User selects multiple uploaded docs.
2. Retrieval pulls evidence from all selected docs.
3. Evidence is rebalanced so one file does not dominate the pool.
4. Answer generation should organize output by document when the request is summary/comparison-oriented.
5. Final output includes:
   - answer text
   - citations
   - confidence
   - supporting evidence panel

This is also the right flow to keep.

## Confidence and trust layer

ScholarRAG currently includes:

- citation grounding
- answer confidence objects
- M/S/A-style support tracking in the schema
- judge evaluation support
- calibration storage tables

Current confidence objects expose:

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

## Sense resolver and scope guard

The backend includes guards for:

- ambiguous terms such as `transformer`, `python`, `java`, `react`
- company/entity questions that only have resume-style evidence
- evidence-insufficient answers that should be limited or clarified rather than hallucinated

This is aligned with the direction you want for a more rigorous ScholarRAG system.

## Data and storage

Current local/runtime artifacts:

- `data/scholar_index.faiss`
- `data/metadata.json`
- `storage/`
- `logs/`

Current database-backed uploaded-doc path:

- `documents`
- `chunks`
- `chunk_embeddings`
- `chat_sessions`
- `chat_messages`
- `eval_runs`
- `confidence_calibration`
- `evidence_scores`
- `evaluation_judge_runs`

## Environment configuration

Create local env file:

```bash
cp .env.example .env
```

Core required values:

- `OPENAI_API_KEY`
- `RESEARCH_CHAT_MODEL`
- `EMBEDDING_PROVIDER`

For Ollama embeddings:

- `EMBEDDING_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_EMBED_MODEL=nomic-embed-text`

For DB / cloud path:

- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWT_SECRET`
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

Optional public provider keys:

- `OPENALEX_API_KEY`
- `SPRINGER_API_KEY`
- `IEEE_API_KEY`
- `ELSEVIER_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`

Important note:

- `OpenAI` is still required for answer generation and judge evaluation in the current architecture.

## Local development

### 1. Start Ollama

```bash
ollama serve
ollama pull nomic-embed-text
```

### 2. Start backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary pgvector python-dotenv requests openai pydantic PyPDF2 python-multipart faiss-cpu numpy google-auth
uvicorn backend.app:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
```

Backend docs:

- `http://127.0.0.1:8000/docs`

### 3. Start frontend

```bash
cd frontend
npm ci
npm run dev
```

Frontend:

- `http://127.0.0.1:5173`

## Deployment target for today

The deployment direction you should keep is:

1. Frontend deploy to `Vercel`
2. Backend deploy separately as `FastAPI`
3. Use `Supabase` for hosted Postgres and storage
4. Keep `Ollama` for embeddings where available
5. Keep `OpenAI` for answer generation and judge

Be explicit during deployment that:

- the current codebase still contains local FAISS/local-storage pieces
- the cloud path is the target production architecture
- some public providers may need credential fixes before being enabled in production

## API surface

- `GET /`
- `POST /assistant/answer`
- `POST /assistant/chat`
- `POST /documents/upload`
- `POST /documents/search/chunks`
- `GET /documents/latest`
- `DELETE /documents/{doc_id}`
- `POST /eval/run`
- `GET /eval/runs`

## Known limitations right now

- FAISS is still part of the local architecture
- UI polish is still in progress
- multi-document summarization behavior is being actively refined
- Elsevier and IEEE credentials/endpoints are not currently working in this environment
- cloud deployment architecture is not yet fully completed end to end

## Recommendation

The flow you want to keep is sound:

- aggregate all public scholarly APIs
- merge and rerank into one final evidence set
- answer only from evidence
- attach citations and confidence
- use judge/calibration as the trust layer

That is the right system design for ScholarRAG.
