ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'unknown';
ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_version TEXT DEFAULT 'v1';

ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'unknown';
ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'unknown';
ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS embedding_version TEXT DEFAULT 'v1';

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_version
ON chunk_embeddings(provider, model, embedding_version);
