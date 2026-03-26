# Embedding Model Comparison

ScholarRAG uses `mxbai-embed-large` via Ollama as its production embedding model. This document explains the selection rationale and the migration path to alternative models.

---

## Model Comparison

| Property | `mxbai-embed-large` | `text-embedding-3-large` | `text-embedding-3-small` | `nomic-embed-text` |
|----------|--------------------|--------------------------|--------------------------|--------------------|
| **Provider** | Ollama (local/remote) | OpenAI API | OpenAI API | Ollama (local/remote) |
| **Output dimension** | 1024-d | 3072-d (truncatable to any) | 1536-d | 768-d |
| **MTEB overall score** | ~64.7 | ~64.6 | ~62.3 | ~62.4 |
| **Cost** | Free (compute only) | ~$0.13 / 1M tokens | ~$0.02 / 1M tokens | Free (compute only) |
| **Latency (local)** | ~25ms / batch-16 | N/A (API) | N/A (API) | ~15ms / batch-16 |
| **Data privacy** | Full (no external calls) | Sends text to OpenAI | Sends text to OpenAI | Full |
| **Requires API key** | No | Yes | Yes | No |

**MTEB**: Massive Text Embedding Benchmark (higher = better retrieval quality across 56 tasks)

---

## Why `mxbai-embed-large`?

1. **Competitive quality**: MTEB score (~64.7) matches `text-embedding-3-large` (~64.6) at zero per-query cost
2. **No data egress**: Research documents can be sensitive; local Ollama keeps all text on-premises
3. **No API dependency**: Embedding availability is not tied to OpenAI uptime or rate limits
4. **Consistent latency**: No network round-trip to an external API; latency is predictable and infrastructure-controlled
5. **Offline-capable**: Works in air-gapped environments once the model is pulled

---

## Current Storage Strategy

`mxbai-embed-large` outputs **1024-d** vectors. The pgvector column is `VECTOR(1536)` for backward compatibility with an earlier `text-embedding-ada-002` schema (which outputs 1536-d).

The embedding service zero-pads the 1024-d output before storage:

```python
# In backend/services/embeddings.py
def _trim_or_pad(vec: list[float], target_dim: int) -> list[float]:
    if len(vec) >= target_dim:
        return vec[:target_dim]
    return vec + [0.0] * (target_dim - len(vec))
```

**Trade-off**: Storing 1024-d vectors in 1536-d columns wastes ~33% of index space and marginally increases ANN query time. This is acceptable during the transition period.

---

## Migration Path to Native Dimensions

When the schema can be migrated, the preferred path is:

### Option A: Migrate to VECTOR(1024) for `mxbai-embed-large`

```sql
-- Run in a maintenance window or via zero-downtime blue/green migration
ALTER TABLE chunk_embeddings
  ALTER COLUMN embedding TYPE VECTOR(1024)
  USING embedding::text::VECTOR(1024);

-- Update VECTOR_STORE_DIM in .env
VECTOR_STORE_DIM=1024
```

**Benefit**: ~33% smaller index, faster ANN queries.

### Option B: Migrate to `text-embedding-3-large` at VECTOR(3072)

```sql
ALTER TABLE chunk_embeddings
  ALTER COLUMN embedding TYPE VECTOR(3072);
```

Then update `.env`:
```env
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBED_DIMENSIONS=3072
VECTOR_STORE_DIM=3072
EMBEDDING_VERSION=text-embedding-3-large-3072d-v1
```

Then reindex all chunks:
```bash
python scripts/reindex_embeddings.py --purge-all
```

**Note**: After reindexing, query-time retrieval will automatically filter on the new `embedding_version` contract, preventing any mixing with old vectors.

---

## Adding a New Embedding Provider

The embedding service is designed for provider-agnosticism. To add a new provider:

1. Implement the HTTP request logic in `backend/services/embeddings.py` inside a new `_embed_*` function
2. Add a branch in `embed_query()` and `embed_documents()` conditioned on `EMBEDDING_PROVIDER`
3. Update `.env.example` with the new provider's variables
4. Update `EMBEDDING_VERSION` to trigger a new contract (prevents silent mixing)
5. Run `scripts/reindex_embeddings.py --purge-all`

---

## Embedding Versioning Contract

Every embedding stored in `chunk_embeddings` carries:

| Column | Example Value | Purpose |
|--------|--------------|---------|
| `provider` | `ollama` | Embedding service provider |
| `model` | `mxbai-embed-large` | Model name |
| `embedding_version` | `mxbai-embed-large-v1` | Version tag (increment on model change) |
| `dim` | `1024` | Raw output dimension (before padding) |

At query time, `search_chunks()` filters `WHERE embedding_version = $EMBEDDING_VERSION`, ensuring the query vector and stored vectors are always from the same model. This eliminates the class of bugs where a model upgrade silently degrades retrieval quality because old and new embeddings are mixed in the same index.
