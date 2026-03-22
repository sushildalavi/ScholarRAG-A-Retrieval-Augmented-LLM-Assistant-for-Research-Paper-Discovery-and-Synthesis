# Retrieval Design

This document covers the chunking strategy, hybrid scoring, and reranking logic in ScholarRAG's retrieval pipeline.

---

## 1. Chunking Strategy

Chunking is implemented in `backend/pdf_ingest.py`. The design goal is to produce chunks that:

1. Fit within the embedding model's context window
2. Preserve enough local context for an answer to be generated from a single chunk
3. Preserve document structure (headings) to support heading-aware citation display
4. Overlap with neighbors to reduce inter-chunk boundary effects

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target chunk size | 420–620 tokens | Balances context density vs. retrieval precision; longer chunks are harder to rank precisely |
| Overlap | 100 tokens (~15–20%) | Ensures sentences at chunk boundaries appear in at least two chunks, reducing information loss |
| Min chunk size | 80 tokens | Discards very short fragments that add noise without semantic value |

### Structure-Aware Chunking

The chunker tracks heading context from the PDF extraction pass. Each chunk stores a `heading_path` field (e.g., `"2 > Background > 2.1 Related Work"`) that appears in the citation display in the frontend evidence panel. This helps users immediately locate the source passage without scanning the full document.

### Why Not Semantic Chunking?

Semantic chunking (splitting at semantic boundaries detected by an embedding model) produces more coherent chunks but:
- Requires an inference pass over the entire document before indexing
- Is sensitive to the quality of the boundary detector
- Adds 5–10× latency to the indexing pipeline

The fixed-size overlap strategy is a pragmatic default that performs well for academic PDFs, which have predictable section/paragraph structure. Semantic chunking is the natural next step for noisy or multi-column PDF layouts.

---

## 2. ANN Index Configuration

Embeddings are stored in PostgreSQL with `pgvector`. The index type is determined by data scale:

| Scale | Recommended Index | Config |
|-------|------------------|--------|
| < 100K vectors | **Exact** (flat scan) | Default; no index needed |
| 100K–1M vectors | **HNSW** | `CREATE INDEX ON chunk_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200)` |
| > 1M vectors | **IVFFlat** | `CREATE INDEX ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists=200)` |

The current dev/test deployment uses exact scan. For production with large document corpora, the HNSW index provides sub-millisecond ANN query latency with recall > 0.99 at ef_search=100.

---

## 3. Hybrid Scoring

Pure dense retrieval misses lexically specific queries (e.g., searching for a specific model name, author, or acronym not seen during embedding model training). Pure sparse retrieval misses semantic synonymy.

The hybrid score combines both:

```
hybrid_score = (1 - α) × cosine_sim(query_vec, candidate_vec) + α × sparse_overlap(query_tokens, candidate_tokens)
```

Where:
- `cosine_sim` is the pgvector cosine similarity between the query embedding and chunk embedding
- `sparse_overlap` is a normalized BM25-style token overlap score: `|query_tokens ∩ candidate_tokens| / sqrt(|query_tokens| × |candidate_tokens|)`
- `α` is configurable via `PUBLIC_SPARSE_WEIGHT` (default 0.25)

**Why α = 0.25?** Empirically, most research queries are semantic in nature (asking about concepts, methods, findings). Dense retrieval dominates. Sparse overlap is a correction signal for the minority of queries that are lexically specific. Increasing α helps for named-entity-heavy queries at the cost of generic semantic queries.

---

## 4. Reranker

After initial retrieval, chunks are reranked using `_chunk_query_overlap` in `backend/services/assistant_utils.py`:

```
rerank_score = semantic_similarity(chunk_embedding, query_embedding)
             + token_overlap_bonus(chunk_text, query_tokens)
             + position_bonus(chunk_page_rank)
```

The reranker operates on the top-20 retrieved chunks and returns the top-10 for generation. The benchmarks show it improves MRR by +21.8% and nDCG@10 by +18.0% over retrieval-only.

**Why not a cross-encoder reranker?** Cross-encoders (e.g., `ms-marco-MiniLM-L-6-v2`) provide higher-quality reranking scores but require a full forward pass per (query, chunk) pair. At batch size 20, this adds ~200ms per query. The current lightweight reranker adds ~18ms (p50) while recovering most of the quality gap. A cross-encoder is the natural upgrade for production workloads where latency budget allows.

---

## 5. Multi-Document Retrieval

When a user selects multiple documents for retrieval:

1. For each selected `doc_id`, retrieve the top-`k/n` chunks (where `n` = number of selected documents)
2. Merge all retrieved chunks across documents into a single candidate pool
3. Rerank the full pool
4. Build context with document-attributed citations (`[doc_id, page]`)

This **equitable rebalancing** ensures that no single document dominates the context window simply because it has more indexed chunks. Each document gets approximately equal representation before the reranker can promote the best evidence regardless of source.

Multi-document generation uses a different system prompt that explicitly instructs the model to synthesize across sources and attribute claims to specific documents.

---

## 6. Retrieval Fallback Cascade

```
User query
    │
    ▼
[uploaded scope selected?]
    │ Yes → pgvector ANN retrieval
    │         → if coverage < 20% AND ENABLE_WEB_FALLBACK=true → web search
    │
    ▼
[public scope selected?]
    │ Yes → multi-provider fan-out (7 APIs)
    │         → if all providers fail → web search fallback
    │
    ▼
[general knowledge query detected?]
    │ Yes → no retrieval; direct GPT-4o-mini generation with disclaimer
```

The fallback thresholds:
- `_FALLBACK_MIN_PARAGRAPHS = 3`: only enforce coverage check on multi-paragraph answers
- `_FALLBACK_MIN_COVERAGE = 0.20`: less than 20% of paragraphs cited → critically uncited → trigger fallback template

---

## 7. Future Retrieval Improvements

| Improvement | Estimated Impact | Complexity |
|-------------|-----------------|------------|
| Cross-encoder reranker (MiniLM) | +5–8% nDCG@10 | Medium |
| Semantic chunking for noisy PDFs | +3–5% Recall@5 | High |
| Query expansion (HyDE) | +4–7% Recall@10 | Medium |
| Learned sparse retrieval (SPLADE) | +2–4% MRR | High |
| Native VECTOR(1024) column | 33% smaller index, ~5ms faster ANN | Low |
| HNSW index for production scale | Sub-ms ANN at 100K+ vectors | Low |
