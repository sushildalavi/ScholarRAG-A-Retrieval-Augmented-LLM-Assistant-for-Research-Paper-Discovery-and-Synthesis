-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    paper_id   TEXT UNIQUE,
    title      TEXT,
    abstract   TEXT,
    authors    TEXT,
    year       INT,
    source     TEXT,
    source_url TEXT,
    embedding  vector(3072),
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

-- Simple updated_at trigger
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_papers_updated_at ON papers;
CREATE TRIGGER trg_papers_updated_at
BEFORE UPDATE ON papers
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
-- Optional ANN index (requires data loaded): uncomment if desired
-- CREATE INDEX IF NOT EXISTS idx_papers_embedding_ivfflat ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Embedding cache
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash  TEXT PRIMARY KEY,
    dim        INT,
    embedding  vector(3072),
    created_at TIMESTAMP DEFAULT now()
);
