-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    filing_type TEXT NOT NULL,
    filing_date DATE,
    quarter TEXT,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    chunk_type TEXT,
    section_name TEXT,
    table_id TEXT,
    row_range TEXT,
    page_estimate INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_ticker ON document_chunks(ticker);
CREATE INDEX IF NOT EXISTS idx_filing_type ON document_chunks(filing_type);
CREATE INDEX IF NOT EXISTS idx_filing_date ON document_chunks(filing_date);
CREATE INDEX IF NOT EXISTS idx_doc_id ON document_chunks(doc_id);

-- Create composite index for common filtered searches
CREATE INDEX IF NOT EXISTS idx_ticker_filing ON document_chunks(ticker, filing_type);

-- Create indexes for new metadata columns
CREATE INDEX IF NOT EXISTS idx_chunk_type ON document_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_section_name ON document_chunks(section_name);
CREATE INDEX IF NOT EXISTS idx_table_id ON document_chunks(table_id);

-- Create vector similarity search index using HNSW
CREATE INDEX IF NOT EXISTS idx_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Analyze table for query optimizer
ANALYZE document_chunks;
