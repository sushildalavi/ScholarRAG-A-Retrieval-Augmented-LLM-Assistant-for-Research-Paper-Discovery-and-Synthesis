# ScholarRAG – AI-Powered Academic Research Assistant

## Objective
A ChatGPT-style assistant that answers academic questions with proper citations, powered by OpenAlex + OpenAI + FAISS + (planned) AWS Athena.

## Architecture
User → Streamlit Frontend → FastAPI Backend → Retriever (FAISS + OpenAlex/Athena planned) → LLM Synthesizer (GPT-4-Turbo)

## Components
- backend/app.py — FastAPI API (`/search`, `/summarize`; `/ask` planned)
- backend/scholar_index_builder.py — builds embeddings and FAISS index
- frontend/streamlit_app.py — simple search UI; chat UI planned
- utils/config.py — environment, .env, Streamlit, AWS Secrets Manager support
- data/ — FAISS index and metadata

## Data Source
Primary: local FAISS index with metadata. Planned: OpenAlex API and AWS Athena for large-scale retrieval.

## Stack
Python · FastAPI · Streamlit · FAISS · OpenAI API · (planned) AWS Athena · OpenAlex API

## Status (~65%)
- ✅ Data ingestion, embeddings, FAISS indexing
- ⚙️ Backend partially built (`/search`, `/summarize`)
- 💬 Frontend prototype working (integration pending)
- 🧠 Citation & summarization with `/ask` next
- ☁️ AWS Athena integration planned

