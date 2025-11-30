# Status Report

Completed
- Project structure reorganized into docs/, data/, backend/, frontend/, utils/.
- Centralized configuration helper for secrets and URLs.
- FastAPI endpoints `/search` and `/summarize` working with FAISS/metadata.
- Streamlit UI for search with backend URL override and error handling.
- Added OpenAlex fallback retrieval with usage telemetry surfaced in the UI.
- Introduced `evaluation/llm_judge.py` to score answers via GPT-based rubric.
- Added `evaluation/run_batch.py` to generate reusable evaluation datasets and compute retrieval metrics.
- Streamlit UI now displays latency, similarity stats, and token usage with a similarity bar chart per query.
- Implemented persistent embedding cache + retry backoff for OpenAI/OpenAlex calls and structured JSON logging.

Next Steps
- Add `/ask` endpoint to synthesize answers with citations from retrieved papers.
- Add a chat-focused Streamlit app (e.g., `frontend/scholarrag_chat.py`).
- Integrate OpenAlex API as a fallback/expansion to FAISS hits.
- Plan AWS Athena schema and implement large-scale retrieval.
- Caching/rate limiting for OpenAI calls to improve performance.
- Use the LLM judge alongside retrieval metrics to set a regression bar.
- Extend notebook metrics with graded relevance labels and nDCG once a gold set is curated.

Operational
- Use `.env` (local) or environment variables for secrets; keep `.env` out of git.
- Optionally wire GitHub Actions/Secrets for CI and deployments.
