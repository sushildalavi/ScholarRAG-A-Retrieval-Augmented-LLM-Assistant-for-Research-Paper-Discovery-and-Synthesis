# ScholarRAG Positioning — Why It Beats NotebookLM, Zotero, and AlphaXiv

## Downsides of NotebookLM
- Closed ecosystem: no API, no custom pipelines/vector DBs/CLI integration.
- Opaque RAG: no visibility into embeddings, chunking, retrieval logs, or ranking scores; cannot tune for scientific accuracy.
- Weak on research-scale docs: large PDFs/books/multi-GB corpora hit limits and truncation.
- No versioning of notes/citations: no diffs, history, or collaborative verification.
- Not citation-grade: no enforced formats (APA/MLA/IEEE) or grounding validation.

## Downsides of Zotero
- No AI-native search or Q&A: stores refs only; no PDF interpretation or summarization.
- Manual organization: folders/tags/annotations are manual; no auto clustering/topic modeling.
- No semantic search: keyword only; cannot handle paraphrases/latent connections.
- Struggles with large libraries: slow around 5k–10k refs; sync pain.
- No multimodal RAG: cannot ingest PDFs → vector store → answer with citations.

## Downsides of AlphaXiv
- Narrow scope: mostly AI/ML; weak for interdisciplinary work.
- No personal corpora: cannot upload your PDFs/datasets.
- Shallow reasoning: surface summaries; no multi-step chains across papers.
- Limited metadata extraction: misses equations, benchmarks, methods, tables.
- No true RAG grounding: search isn’t anchored in your corpus.

## ScholarRAG — Superior Approach
- Multi-source ingestion: PDFs, HTML, transcripts, datasets, notes.
- Personalized RAG: per-user vector stores tuned to domain; semantic search across papers/books/slides/reports/wiki/datasets.
- Research-grade grounding: citations with links/page numbers/snippet highlights/confidence scores.
- Automated literature mapping: topic clusters, embedding visuals, citation graphs.
- Intermediate reasoning: multi-hop chains across documents (without leaking private chains).
- Extensible and API-first: plugins for Firebase/Postgres/Pinecone/LlamaIndex/LangChain.
- Domain-agnostic: works beyond AI/ML across all scholarly fields.
