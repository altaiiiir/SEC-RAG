-- Migration script to add new metadata columns to existing document_chunks table
-- Run this if you already have a database with the old schema

-- Add new columns
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section_name TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS table_id TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS row_range TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_estimate INT;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_chunk_type ON document_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_section_name ON document_chunks(section_name);
CREATE INDEX IF NOT EXISTS idx_table_id ON document_chunks(table_id);

-- Analyze table for query optimizer
ANALYZE document_chunks;

-- Note: After running this migration, you should re-index your documents
-- to populate the new metadata columns with adaptive chunking data.
