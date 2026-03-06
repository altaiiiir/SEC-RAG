from pathlib import Path
from typing import List, Dict, Optional
import re
import json
import psycopg2
from sentence_transformers import SentenceTransformer
import tiktoken

from src.db_config import get_db_config, get_embedding_model_name, get_chunk_config

class DocumentIndexer:
    """Indexes SEC EDGAR documents into pgvector database."""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.embedding_model_name = get_embedding_model_name()
        chunk_config = get_chunk_config()
        self.chunk_size = chunk_config["size"]
        self.chunk_overlap = chunk_config["overlap"]
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
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
        """Create database connection."""
        return psycopg2.connect(**self.db_config)
    
    def index_document(self, filepath: Path) -> int:
        """Index a single document file."""
        metadata = self._parse_filename(filepath.name)
        doc_id = filepath.stem
        
        print(f"Indexing {filepath.name}...")
        
        # Read document
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Chunk document
        chunks = self._chunk_text(content)
        print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Store in database
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                cursor.execute(
                    """
                    INSERT INTO document_chunks 
                    (doc_id, ticker, filing_type, filing_date, quarter, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
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
                )
                inserted += 1
            except Exception as e:
                print(f"  Error inserting chunk {idx}: {e}")
                
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"  Inserted {inserted} chunks")
        return inserted
    
    def index_corpus(self, corpus_dir: str = "edgar_corpus", max_docs: int = None) -> Dict[str, int]:
        """Index all documents in the corpus directory using manifest.json.
        
        Args:
            corpus_dir: Directory containing documents
            max_docs: Maximum number of documents to index (None = all)
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
        
        print(f"Indexing {len(txt_files)} documents...")
        
        stats = {
            "total_docs": len(txt_files),
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
        }
        
        for filepath in txt_files:
            try:
                chunks_inserted = self.index_document(filepath)
                stats["total_chunks"] += chunks_inserted
                stats["successful"] += 1
            except Exception as e:
                print(f"Failed to index {filepath.name}: {e}")
                stats["failed"] += 1
                
        return stats
    
    def clear_index(self):
        """Clear all data from the index."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        conn.commit()
        cursor.close()
        conn.close()
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
