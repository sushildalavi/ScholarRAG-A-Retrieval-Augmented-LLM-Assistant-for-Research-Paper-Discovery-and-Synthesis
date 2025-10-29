# Status Report

Completed
- Project structure reorganized into docs/, data/, backend/, frontend/, utils/.
- Centralized configuration helper for secrets and URLs.
- FastAPI endpoints `/search` and `/summarize` working with FAISS/metadata.
- Streamlit UI for search with backend URL override and error handling.

Next Steps
- Add `/ask` endpoint to synthesize answers with citations from retrieved papers.
- Add a chat-focused Streamlit app (e.g., `frontend/scholarrag_chat.py`).
- Integrate OpenAlex API as a fallback/expansion to FAISS hits.
- Plan AWS Athena schema and implement large-scale retrieval.
- Caching/rate limiting for OpenAI calls to improve performance.

Operational
- Use `.env` (local) or environment variables for secrets; keep `.env` out of git.
- Optionally wire GitHub Actions/Secrets for CI and deployments.

