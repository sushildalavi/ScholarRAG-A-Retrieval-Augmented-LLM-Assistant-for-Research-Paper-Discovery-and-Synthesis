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
    -- Using 1536-dim to match OpenAI text-embedding-3-small (fast/accurate balance)
    embedding  vector(1536),
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
    provider   TEXT DEFAULT 'unknown',
    model      TEXT DEFAULT 'unknown',
    embedding_version TEXT DEFAULT 'v1',
    dim        INT,
    embedding  vector(1536),
    created_at TIMESTAMP DEFAULT now()
);

-- --- Document ingestion (chunk-level RAG) ---
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    doc_type TEXT DEFAULT 'other',
    source_path TEXT,
    mime_type TEXT,
    pages INT,
    bytes BIGINT,
    hash_sha256 TEXT,
    status TEXT DEFAULT 'ready',
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    page_no INT,
    chunk_index INT,
    text TEXT,
    tokens INT,
    modality TEXT DEFAULT 'text',
    heading_path TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INT REFERENCES chunks(id) ON DELETE CASCADE,
    provider TEXT DEFAULT 'unknown',
    model TEXT NOT NULL,
    embedding_version TEXT DEFAULT 'v1',
    dim INT NOT NULL,
    vector vector(1536),
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk ON chunk_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_version ON chunk_embeddings(provider, model, embedding_version);
-- Optional ANN index (build after data load)
-- CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_ivfflat ON chunk_embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- --- Chat sessions/messages/uploads ---
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- 'user' | 'assistant'
    content TEXT,
    citations JSONB,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_uploads (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES chat_sessions(id) ON DELETE CASCADE,
    doc_id INT REFERENCES documents(id) ON DELETE SET NULL,
    file_path TEXT,
    mime_type TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_uploads_session ON chat_uploads(session_id);

-- --- Evaluation runs ---
CREATE TABLE IF NOT EXISTS eval_runs (
    id SERIAL PRIMARY KEY,
    name TEXT,
    scope TEXT DEFAULT 'uploaded',
    k INT DEFAULT 10,
    case_count INT DEFAULT 0,
    metrics_retrieval_only JSONB,
    metrics_retrieval_rerank JSONB,
    latency_breakdown JSONB,
    details JSONB,
    created_at TIMESTAMP DEFAULT now()
);

-- ----- M/S/A confidence scoring artifacts -----
CREATE TABLE IF NOT EXISTS confidence_calibration (
    id SERIAL PRIMARY KEY,
    model_name TEXT DEFAULT 'msa_logistic_v1',
    label TEXT DEFAULT 'default',
    weights JSONB NOT NULL,
    metrics JSONB,
    dataset_size INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence_scores (
    id SERIAL PRIMARY KEY,
    request_id TEXT,
    sentence_id INT,
    citation_id INT,
    evidence_id TEXT,
    m_score REAL,
    s_score REAL,
    a_score REAL,
    score REAL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_judge_runs (
    id SERIAL PRIMARY KEY,
    scope TEXT DEFAULT 'uploaded',
    query_count INT DEFAULT 0,
    metrics JSONB,
    details JSONB,
    created_at TIMESTAMP DEFAULT now()
);
