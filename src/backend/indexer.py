from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import os

from src.backend.db_config import get_db_config, get_embedding_model_name
from src.backend.chunking_config import get_chunking_config
from src.backend.content_detector import SECFilingParser
from src.backend.adaptive_chunker import SECChunker

_connection_pool = None


def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        db_config = get_db_config()
        _connection_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=20, **db_config)
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
        config = get_chunking_config()
        self.embed_batch_size = config["embed_batch_size"]
        self.embed_batch_files = config["embed_batch_files"]

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

    def _already_indexed(self, doc_ids: List[str]) -> set:
        """Single DB query returning the set of doc_ids already in the table."""
        if not doc_ids:
            return set()
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id FROM document_chunks WHERE doc_id = ANY(%s)",
                    (doc_ids,),
                )
                return {row[0] for row in cur.fetchall()}
        except Exception as e:
            print(f"Warning: duplicate check failed: {e}")
            return set()
        finally:
            self.pool.putconn(conn)

    def _parse_chunk_file(self, filepath: Path) -> Tuple[Path, List[Dict]]:
        """Read, parse, and chunk one file. Safe to call from a thread."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = self._chunk_document(content)
        return filepath, chunks

    def _insert_batch(self, records: List[tuple]) -> int:
        """Bulk-insert a list of row tuples. Returns number inserted."""
        if not records:
            return 0
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """
                    INSERT INTO document_chunks
                    (doc_id, ticker, filing_type, filing_date, quarter,
                     chunk_index, content, embedding, chunk_type, section_name,
                     table_id, row_range)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    records,
                    page_size=500,
                )
            conn.commit()
            return len(records)
        except Exception as e:
            conn.rollback()
            print(f"  DB insert error: {e}")
            return 0
        finally:
            self.pool.putconn(conn)

    # ------------------------------------------------------------------
    # Single-document path (kept for backwards compatibility / small runs)
    # ------------------------------------------------------------------
    def index_document(self, filepath: Path) -> int:
        """Index a single document (used by test helpers and small runs)."""
        doc_id = filepath.stem
        already = self._already_indexed([doc_id])
        if doc_id in already:
            print(f"Skipping {filepath.name} (already indexed)")
            return 0

        print(f"Indexing {filepath.name}...")
        _, chunk_dicts = self._parse_chunk_file(filepath)
        if not chunk_dicts:
            print("  No chunks created, skipping")
            return 0

        print(f"  Created {len(chunk_dicts)} chunks")
        metadata = self._parse_filename(filepath.name)
        texts = [c["text"] for c in chunk_dicts]
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=self.embed_batch_size)

        records = [
            (
                doc_id,
                metadata["ticker"],
                metadata["filing_type"],
                metadata["filing_date"],
                metadata["quarter"],
                idx,
                c["text"],
                emb.tolist(),
                c.get("chunk_type", "narrative"),
                c.get("section_name", ""),
                c.get("table_id"),
                c.get("row_range"),
            )
            for idx, (c, emb) in enumerate(zip(chunk_dicts, embeddings))
        ]

        inserted = self._insert_batch(records)
        print(f"  Inserted {inserted} chunks")
        return inserted

    # ------------------------------------------------------------------
    # Full corpus path: thread-parallel parse+chunk, batched embed+insert
    # ------------------------------------------------------------------
    def index_corpus(self, corpus_dir: str = "edgar_corpus", max_docs: int = None, parallel: bool = True) -> Dict[str, int]:
        """Index all documents: thread-parallel parse/chunk, batched embedding, bulk DB insert."""
        corpus_path = Path(corpus_dir)
        manifest_path = corpus_path / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            txt_files = [corpus_path / fn for fn in manifest.get("files", [])]
            print(f"Loaded {len(txt_files)} files from manifest.json")
        else:
            txt_files = list(corpus_path.glob("*.txt"))
            print(f"No manifest.json found, using glob: {len(txt_files)} files")

        if max_docs is not None:
            txt_files = txt_files[:max_docs]
            print(f"Limited to {max_docs} documents")

        # --- Bulk duplicate check (one query, not N) ---
        all_doc_ids = [f.stem for f in txt_files]
        already_indexed = self._already_indexed(all_doc_ids)
        new_files = [f for f in txt_files if f.stem not in already_indexed]
        skipped = len(txt_files) - len(new_files)
        if skipped:
            print(f"Skipping {skipped} already-indexed documents")
        print(f"Indexing {len(new_files)} new documents...")

        stats = {
            "total_docs": len(txt_files),
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
            "skipped": skipped,
        }

        if not new_files:
            return stats

        n_workers = min(os.cpu_count() or 4, len(new_files), 8)

        def process_batch(batch: List[Path]):
            """Parse+chunk a batch of files in threads, embed all at once, insert."""
            # --- Thread-parallel parse + chunk ---
            file_chunks: Dict[Path, List[Dict]] = {}
            errors: Dict[Path, Exception] = {}

            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                futures = {exe.submit(self._parse_chunk_file, fp): fp for fp in batch}
                for fut in as_completed(futures):
                    fp = futures[fut]
                    try:
                        _, chunks = fut.result()
                        file_chunks[fp] = chunks
                    except Exception as e:
                        errors[fp] = e
                        print(f"  Parse error {fp.name}: {e}")

            # --- Collect all texts for a single model.encode() call ---
            ordered_files = [fp for fp in batch if fp in file_chunks and file_chunks[fp]]
            if not ordered_files:
                return 0, 0, len(errors)

            all_texts = []
            offsets: List[Tuple[Path, int, int]] = []  # (filepath, start_idx, end_idx)
            for fp in ordered_files:
                start = len(all_texts)
                all_texts.extend(c["text"] for c in file_chunks[fp])
                offsets.append((fp, start, len(all_texts)))

            print(f"  Embedding {len(all_texts)} chunks from {len(ordered_files)} files...")
            all_embeddings = self.model.encode(
                all_texts,
                show_progress_bar=False,
                batch_size=self.embed_batch_size,
            )

            # --- Build DB records and bulk insert ---
            records = []
            for fp, start, end in offsets:
                metadata = self._parse_filename(fp.name)
                doc_id = fp.stem
                for idx, (chunk, emb) in enumerate(zip(file_chunks[fp], all_embeddings[start:end])):
                    records.append((
                        doc_id,
                        metadata["ticker"],
                        metadata["filing_type"],
                        metadata["filing_date"],
                        metadata["quarter"],
                        idx,
                        chunk["text"],
                        emb.tolist(),
                        chunk.get("chunk_type", "narrative"),
                        chunk.get("section_name", ""),
                        chunk.get("table_id"),
                        chunk.get("row_range"),
                    ))

            inserted = self._insert_batch(records)
            successful = len(ordered_files)
            failed = len(errors)
            print(f"  Batch done: {successful} files, {inserted} chunks inserted")
            return inserted, successful, failed

        # Process in batches of embed_batch_files
        batch_size = self.embed_batch_files
        for i in range(0, len(new_files), batch_size):
            batch = new_files[i: i + batch_size]
            print(f"\n[{i + 1}-{min(i + batch_size, len(new_files))}/{len(new_files)}] Processing batch...")
            inserted, ok, fail = process_batch(batch)
            stats["total_chunks"] += inserted
            stats["successful"] += ok
            stats["failed"] += fail

        return stats

    def clear_index(self):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE document_chunks")
            conn.commit()
        finally:
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
    print(f"Successful:      {stats['successful']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Total chunks:    {stats['total_chunks']}")
