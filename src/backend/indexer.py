from pathlib import Path
from typing import List, Dict, Optional
import re
import json
from psycopg2 import pool
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os

from src.backend.db_config import get_db_config, get_embedding_model_name
from src.backend.content_detector import SECFilingParser
from src.backend.adaptive_chunker import SECChunker

_connection_pool = None


def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        db_config = get_db_config()
        _connection_pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, **db_config)
    return _connection_pool


@lru_cache(maxsize=1)
def get_embedding_model():
    model_name = get_embedding_model_name()
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


class DocumentIndexer:
    """Indexes SEC EDGAR documents into pgvector database."""

    def __init__(self):
        self.db_config = get_db_config()
        self.model = get_embedding_model()
        self.pool = get_connection_pool()
        self.parser = SECFilingParser()
        self.chunker = SECChunker()

    def _parse_filename(self, filename: str) -> Dict[str, Optional[str]]:
        """Extract metadata from filename like AAPL_10K_2022Q3_2022-10-28_full.txt"""
        parts = filename.replace("_full.txt", "").split("_")
        metadata = {
            "ticker": parts[0] if len(parts) > 0 else None,
            "filing_type": parts[1] if len(parts) > 1 else None,
            "quarter": None,
            "filing_date": None,
        }
        for part in parts[2:]:
            if re.match(r"\d{4}Q\d", part):
                metadata["quarter"] = part
            elif re.match(r"\d{4}-\d{2}-\d{2}", part):
                metadata["filing_date"] = part
        return metadata

    def _chunk_document(self, text: str) -> List[Dict]:
        """Parse and chunk an SEC filing."""
        sections = self.parser.parse(text)
        return self.chunker.chunk_document(sections)

    def index_document(self, filepath: Path) -> int:
        """Index a single document file with batch inserts."""
        metadata = self._parse_filename(filepath.name)
        doc_id = filepath.stem
        print(f"Indexing {filepath.name}...")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunk_dicts = self._chunk_document(content)
        print(f"  Created {len(chunk_dicts)} chunks")

        if not chunk_dicts:
            print("  No chunks created, skipping")
            return 0

        chunk_texts = [chunk['text'] for chunk in chunk_dicts]
        embeddings = self.model.encode(chunk_texts, show_progress_bar=False, batch_size=32)

        conn = self.pool.getconn()
        cursor = conn.cursor()

        batch_data = []
        for idx, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
            batch_data.append((
                doc_id,
                metadata["ticker"],
                metadata["filing_type"],
                metadata["filing_date"],
                metadata["quarter"],
                idx,
                chunk_dict['text'],
                embedding.tolist(),
                chunk_dict.get('chunk_type', 'narrative'),
                chunk_dict.get('section_name', ''),
                None,
                None,
                None,
            ))

        try:
            execute_batch(
                cursor,
                """
                INSERT INTO document_chunks 
                (doc_id, ticker, filing_type, filing_date, quarter, chunk_index, content, embedding,
                 chunk_type, section_name, table_id, row_range, page_estimate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            self.pool.putconn(conn)

        print(f"  Inserted {inserted} chunks")
        return inserted

    def index_corpus(self, corpus_dir: str = "edgar_corpus", max_docs: int = None, parallel: bool = True) -> Dict[str, int]:
        """Index all documents with optional parallel processing."""
        corpus_path = Path(corpus_dir)
        manifest_path = corpus_path / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            txt_files = [corpus_path / filename for filename in manifest.get('files', [])]
            print(f"Loaded {len(txt_files)} files from manifest.json")
        else:
            txt_files = list(corpus_path.glob("*.txt"))
            print(f"No manifest.json found, using glob: {len(txt_files)} files")

        if max_docs is not None:
            txt_files = txt_files[:max_docs]
            print(f"Limited to {max_docs} documents")

        print(f"Indexing {len(txt_files)} documents (parallel={parallel})...")

        stats = {"total_docs": len(txt_files), "total_chunks": 0, "successful": 0, "failed": 0}

        if parallel and len(txt_files) > 1:
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
        indexer = DocumentIndexer()
        return indexer.index_document(Path(filepath_str))

    def clear_index(self):
        conn = self.pool.getconn()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE document_chunks")
        conn.commit()
        cursor.close()
        self.pool.putconn(conn)
        print("Index cleared")


if __name__ == "__main__":
    import sys

    indexer = DocumentIndexer()

    max_docs = None
    if len(sys.argv) > 1:
        try:
            max_docs = int(sys.argv[1])
            print(f"Limiting to {max_docs} documents")
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, indexing all documents")

    stats = indexer.index_corpus(max_docs=max_docs)

    print("\n" + "=" * 50)
    print("INDEXING COMPLETE")
    print("=" * 50)
    print(f"Total documents: {stats['total_docs']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks: {stats['total_chunks']}")
