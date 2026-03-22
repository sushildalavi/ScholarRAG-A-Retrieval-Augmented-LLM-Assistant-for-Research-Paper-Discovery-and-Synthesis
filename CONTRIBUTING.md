# Contributing to ScholarRAG

## Local Development Setup

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for local Postgres)
- Ollama (for local embeddings)

### 2. Install dependencies

```bash
# Python runtime + dev dependencies
python3 -m venv .venv
source .venv/bin/activate
make install-dev

# Frontend
cd frontend && npm ci
```

### 3. Configure environment

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, DATABASE_URL, SUPABASE_*, OLLAMA_BASE_URL
```

### 4. Start backing services

```bash
# Postgres
make db-up

# Ollama
ollama pull mxbai-embed-large
ollama serve
```

### 5. Start the services

```bash
# Backend (terminal 1)
make run

# Frontend (terminal 2)
make run-frontend
```

---

## Running Tests

```bash
make test           # full test suite
make test-fast      # stop on first failure
make test-coverage  # with HTML coverage report
```

Tests are located in `backend/tests/`. They test only pure-function logic and require no external API calls or database connections.

---

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting.

```bash
make lint       # check
make lint-fix   # auto-fix
```

Key conventions:
- Python 3.11+ type hints on all new public functions
- No bare `except:` — always `except SomeError:` or `except Exception:`
- Pure functions should be prefixed with `_` if internal, unprefixed if part of public API
- Module-level docstrings required on all new service modules

---

## Project Layout

```
backend/          Core FastAPI backend
  services/       Stateless service modules (embeddings, db, judge, nli, etc.)
  tests/          pytest test suite (no external deps required)
utils/            Scholarly API integrations (one file per provider)
frontend/src/     React + TypeScript SPA
db/               PostgreSQL schema and migrations
docs/             Architecture, evaluation, and design documentation
scripts/          One-off operational scripts (eval, reindex)
```

---

## Adding a New Scholarly API Provider

1. Create `utils/<provider>_utils.py` following the pattern in `utils/openalex_utils.py`
2. Add the provider function to `backend/public_search.py` in the `_PROVIDER_FUNS` dict
3. Add the API key env var to `.env.example`
4. Update the provider count in `README.md`

---

## Adding a New Embedding Provider

See [`docs/EMBEDDING_MODEL_COMPARISON.md`](docs/EMBEDDING_MODEL_COMPARISON.md) for the full migration guide.

---

## Evaluation

Before merging changes that affect retrieval or generation:

```bash
make eval   # runs scripts/eval_retrieval.py against eval_data/golden_set.json
```

Report Recall@5, MRR, and nDCG@10 in the PR description if they change.

---

## Pull Request Guidelines

- Keep PRs focused: one feature or fix per PR
- Include test coverage for new pure functions
- If retrieval metrics change, include before/after numbers
- Link relevant issues in the PR description
