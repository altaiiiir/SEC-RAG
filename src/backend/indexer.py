from pathlib import Path
from typing import List, Dict, Optional
import re
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
import tiktoken
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os

from src.backend.db_config import get_db_config, get_embedding_model_name, get_chunk_config

# Global connection pool
_connection_pool = None

def get_connection_pool():
    """Get or create connection pool."""
    global _connection_pool
    if _connection_pool is None:
        db_config = get_db_config()
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **db_config
        )
    return _connection_pool

@lru_cache(maxsize=1)
def get_embedding_model():
    """Get cached embedding model."""
    model_name = get_embedding_model_name()
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

class DocumentIndexer:
    """Indexes SEC EDGAR documents into pgvector database."""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.embedding_model_name = get_embedding_model_name()
        chunk_config = get_chunk_config()
        self.chunk_size = chunk_config["size"]
        self.chunk_overlap = chunk_config["overlap"]
        
        self.model = get_embedding_model()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.pool = get_connection_pool()
        
    def _parse_filename(self, filename: str) -> Dict[str, Optional[str]]:
        """Extract metadata from filename like AAPL_10K_2022Q3_2022-10-28_full.txt"""
        parts = filename.replace("_full.txt", "").split("_")
        
        metadata = {
            "ticker": parts[0] if len(parts) > 0 else None,
            "filing_type": parts[1] if len(parts) > 1 else None,
            "quarter": None,
            "filing_date": None,
        }
        
        # Extract quarter and date
        for part in parts[2:]:
            if re.match(r"\d{4}Q\d", part):
                metadata["quarter"] = part
            elif re.match(r"\d{4}-\d{2}-\d{2}", part):
                metadata["filing_date"] = part
                
        return metadata
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def _get_db_connection(self):
        """Get database connection from pool."""
        return self.pool.getconn()
    
    def _return_db_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def index_document(self, filepath: Path) -> int:
        """Index a single document file with batch inserts."""
        metadata = self._parse_filename(filepath.name)
        doc_id = filepath.stem
        
        print(f"Indexing {filepath.name}...")
        
        # Read document
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Chunk document
        chunks = self._chunk_text(content)
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings in batch
        embeddings = self.model.encode(chunks, show_progress_bar=False, batch_size=32)
        
        # Store in database with batch insert
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Prepare batch data
        batch_data = [
            (
                doc_id,
                metadata["ticker"],
                metadata["filing_type"],
                metadata["filing_date"],
                metadata["quarter"],
                idx,
                chunk,
                embedding.tolist(),
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Batch insert
        try:
            execute_batch(
                cursor,
                """
                INSERT INTO document_chunks 
                (doc_id, ticker, filing_type, filing_date, quarter, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                batch_data,
                page_size=100
            )
            conn.commit()
            inserted = len(batch_data)
        except Exception as e:
            conn.rollback()
            print(f"  Error inserting chunks: {e}")
            inserted = 0
        finally:
            cursor.close()
            self._return_db_connection(conn)
        
        print(f"  Inserted {inserted} chunks")
        return inserted
    
    def index_corpus(self, corpus_dir: str = "edgar_corpus", max_docs: int = None, parallel: bool = True) -> Dict[str, int]:
        """Index all documents with optional parallel processing.
        
        Args:
            corpus_dir: Directory containing documents
            max_docs: Maximum number of documents to index (None = all)
            parallel: Use parallel processing (default: True)
        """
        corpus_path = Path(corpus_dir)
        manifest_path = corpus_path / "manifest.json"
        
        # Load manifest if it exists
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            txt_files = [corpus_path / filename for filename in manifest.get('files', [])]
            print(f"Loaded {len(txt_files)} files from manifest.json")
        else:
            # Fallback to glob if no manifest
            txt_files = list(corpus_path.glob("*.txt"))
            print(f"No manifest.json found, using glob: {len(txt_files)} files")
        
        # Limit number of files if specified
        if max_docs is not None:
            txt_files = txt_files[:max_docs]
            print(f"Limited to {max_docs} documents")
        
        print(f"Indexing {len(txt_files)} documents (parallel={parallel})...")
        
        stats = {
            "total_docs": len(txt_files),
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
        }
        
        if parallel and len(txt_files) > 1:
            # Parallel processing
            max_workers = min(os.cpu_count() or 4, len(txt_files))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._index_document_worker, str(fp)): fp for fp in txt_files}
                
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        chunks_inserted = future.result()
                        stats["total_chunks"] += chunks_inserted
                        stats["successful"] += 1
                    except Exception as e:
                        print(f"Failed to index {filepath.name}: {e}")
                        stats["failed"] += 1
        else:
            # Sequential processing
            for filepath in txt_files:
                try:
                    chunks_inserted = self.index_document(filepath)
                    stats["total_chunks"] += chunks_inserted
                    stats["successful"] += 1
                except Exception as e:
                    print(f"Failed to index {filepath.name}: {e}")
                    stats["failed"] += 1
                    
        return stats
    
    @staticmethod
    def _index_document_worker(filepath_str: str) -> int:
        """Worker function for parallel processing."""
        # Each worker needs its own indexer instance
        indexer = DocumentIndexer()
        return indexer.index_document(Path(filepath_str))
    
    def clear_index(self):
        """Clear all data from the index."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE document_chunks")
        conn.commit()
        cursor.close()
        self._return_db_connection(conn)
        print("Index cleared")


if __name__ == "__main__":
    import sys
    
    indexer = DocumentIndexer()
    
    # Get max_docs from command line if provided
    max_docs = None
    if len(sys.argv) > 1:
        try:
            max_docs = int(sys.argv[1])
            print(f"Limiting to {max_docs} documents")
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, indexing all documents")
    
    # Index documents
    stats = indexer.index_corpus(max_docs=max_docs)
    
    print("\n" + "="*50)
    print("INDEXING COMPLETE")
    print("="*50)
    print(f"Total documents: {stats['total_docs']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks: {stats['total_chunks']}")
