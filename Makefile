.PHONY: test lint typecheck run run-frontend install install-dev clean help

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn

# ── Setup ─────────────────────────────────────────────────────────────────────

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) backend/tests/ -v --tb=short

test-fast:
	$(PYTEST) backend/tests/ -x --tb=short -q

test-coverage:
	$(PYTEST) backend/tests/ --cov=backend --cov-report=term-missing --cov-report=html

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(RUFF) check backend/ utils/ --ignore E501,F401,E402

lint-fix:
	$(RUFF) check backend/ utils/ --fix --ignore E501,F401,E402

typecheck:
	$(VENV)/bin/pyright backend/confidence.py backend/eval_metrics.py backend/services/nli.py

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	$(UVICORN) backend.app:app --reload --host 127.0.0.1 --port 8000

run-prod:
	$(UVICORN) backend.app:app --host 0.0.0.0 --port 8000 --workers 4

run-frontend:
	cd frontend && npm run dev

# ── Database ──────────────────────────────────────────────────────────────────

db-up:
	docker compose up -d db adminer

db-down:
	docker compose down

# ── Reindex ───────────────────────────────────────────────────────────────────

reindex:
	$(PYTHON) scripts/reindex_embeddings.py --purge-all

# ── Eval ──────────────────────────────────────────────────────────────────────

eval:
	$(PYTHON) scripts/eval_retrieval.py \
		--eval-set eval_data/golden_set.json \
		--k 10 \
		--output eval_results/run_$(shell date +%Y%m%d).json

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "ScholarRAG Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       Install runtime dependencies"
	@echo "    make install-dev   Install runtime + dev dependencies"
	@echo ""
	@echo "  Test:"
	@echo "    make test          Run full test suite"
	@echo "    make test-fast     Run tests, stop on first failure"
	@echo "    make test-coverage Run tests with coverage report"
	@echo ""
	@echo "  Lint:"
	@echo "    make lint          Run ruff linter"
	@echo "    make lint-fix      Run ruff with auto-fix"
	@echo "    make typecheck     Run pyright on core modules"
	@echo ""
	@echo "  Run:"
	@echo "    make run           Start backend (dev, auto-reload)"
	@echo "    make run-prod      Start backend (production, 4 workers)"
	@echo "    make run-frontend  Start frontend dev server"
	@echo ""
	@echo "  Eval:"
	@echo "    make reindex       Rebuild all chunk embeddings"
	@echo "    make eval          Run retrieval evaluation"
