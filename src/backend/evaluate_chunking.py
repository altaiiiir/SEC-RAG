"""Evaluation script for chunking strategies. Compares metrics before/after."""
import sys
from pathlib import Path
from typing import Dict, List
import statistics
import tiktoken
import psycopg2

from src.backend.db_config import get_db_config
from src.backend.content_detector import SECFilingParser
from src.backend.adaptive_chunker import SECChunker
from src.backend.chunking_config import get_chunking_config


class ChunkingEvaluator:
    """Evaluate chunking quality."""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.db_config = get_db_config()

    def evaluate_file(self, filepath: Path) -> Dict:
        """Evaluate chunking on a single file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fixed-size chunking (baseline)
        fixed_chunks = self._chunk_fixed(content)

        # SEC-specialized chunking
        parser = SECFilingParser()
        chunker = SECChunker()
        sections = parser.parse(content)
        sec_chunks = chunker.chunk_document(sections)

        sec_token_counts = [c['token_count'] for c in sec_chunks] if sec_chunks else [0]

        return {
            'filename': filepath.name,
            'doc_length_chars': len(content),
            'doc_length_tokens': len(self.tokenizer.encode(content)),

            'fixed_chunk_count': len(fixed_chunks),
            'fixed_avg_size': statistics.mean([len(self.tokenizer.encode(c)) for c in fixed_chunks]),
            'fixed_std_size': statistics.stdev([len(self.tokenizer.encode(c)) for c in fixed_chunks]) if len(fixed_chunks) > 1 else 0,

            'sec_chunk_count': len(sec_chunks),
            'sec_avg_size': statistics.mean(sec_token_counts),
            'sec_std_size': statistics.stdev(sec_token_counts) if len(sec_chunks) > 1 else 0,
            'sec_sections_found': len(sections),
            'sec_section_names': [s.name for s in sections],

            'mid_sentence_splits': self._count_mid_sentence_splits(fixed_chunks),
        }

    def _chunk_fixed(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Fixed-size chunking (baseline)."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunks.append(self.tokenizer.decode(tokens[start:end]))
            start += chunk_size - overlap
        return chunks

    def _count_mid_sentence_splits(self, chunks: List[str]) -> int:
        """Count how many chunks end mid-sentence."""
        import re
        sentence_end = re.compile(r'[.!?]\s*$')
        return sum(1 for chunk in chunks[:-1] if not sentence_end.search(chunk.strip()))

    def get_database_stats(self) -> Dict:
        """Get statistics from indexed chunks in database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*), COUNT(DISTINCT doc_id), COUNT(DISTINCT ticker)
                FROM document_chunks
            """)
            overall = cursor.fetchone()

            cursor.execute("""
                SELECT section_name, COUNT(*) FROM document_chunks
                WHERE section_name IS NOT NULL AND section_name != ''
                GROUP BY section_name ORDER BY COUNT(*) DESC
            """)
            sections = dict(cursor.fetchall())

            cursor.execute("""
                SELECT AVG(LENGTH(content))::int, MIN(LENGTH(content)), MAX(LENGTH(content))
                FROM document_chunks
            """)
            sizes = cursor.fetchone()

            cursor.close()
            conn.close()

            return {
                'total_chunks': overall[0], 'total_docs': overall[1], 'total_tickers': overall[2],
                'section_distribution': sections,
                'avg_size': sizes[0], 'min_size': sizes[1], 'max_size': sizes[2],
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

    def print_metrics(self, metrics: Dict):
        """Pretty print evaluation metrics."""
        print("\n" + "=" * 60)
        print(f"CHUNKING EVALUATION: {metrics['filename']}")
        print("=" * 60)

        print(f"\nDocument: {metrics['doc_length_chars']:,} chars, {metrics['doc_length_tokens']:,} tokens")

        print(f"\nFixed-Size Baseline:")
        print(f"  Chunks: {metrics['fixed_chunk_count']}")
        print(f"  Avg: {metrics['fixed_avg_size']:.0f} tokens (std: {metrics['fixed_std_size']:.0f})")
        print(f"  Mid-sentence splits: {metrics['mid_sentence_splits']}")

        print(f"\nSEC-Specialized:")
        print(f"  Chunks: {metrics['sec_chunk_count']}")
        print(f"  Avg: {metrics['sec_avg_size']:.0f} tokens (std: {metrics['sec_std_size']:.0f})")
        print(f"  Sections found: {metrics['sec_sections_found']}")
        for name in metrics['sec_section_names']:
            print(f"    - {name}")

        print("\n" + "=" * 60)

    def print_database_stats(self, stats: Dict):
        """Pretty print database statistics."""
        if not stats:
            print("\nNo database statistics available")
            return

        print("\n" + "=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"  Chunks: {stats.get('total_chunks', 0):,}  |  Docs: {stats.get('total_docs', 0):,}  |  Tickers: {stats.get('total_tickers', 0):,}")
        print(f"  Chunk sizes: avg={stats.get('avg_size', 0)} chars, range={stats.get('min_size', 0)}-{stats.get('max_size', 0)}")

        sections = stats.get('section_distribution', {})
        if sections:
            print(f"\n  Sections:")
            for name, count in sections.items():
                print(f"    {name}: {count}")

        print("=" * 60)


def main():
    evaluator = ChunkingEvaluator()

    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        if filepath.exists():
            metrics = evaluator.evaluate_file(filepath)
            evaluator.print_metrics(metrics)
        else:
            print(f"File not found: {filepath}")
            sys.exit(1)
    else:
        sample_dir = Path("edgar_corpus")
        if sample_dir.exists():
            txt_files = list(sample_dir.glob("*.txt"))
            if txt_files:
                metrics = evaluator.evaluate_file(txt_files[0])
                evaluator.print_metrics(metrics)

    db_stats = evaluator.get_database_stats()
    evaluator.print_database_stats(db_stats)


if __name__ == "__main__":
    main()
