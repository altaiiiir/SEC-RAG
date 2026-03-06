"""
Evaluation script for chunking strategies.
Compares different chunking approaches and measures quality metrics.
"""
import sys
from pathlib import Path
from typing import Dict, List
import statistics
import tiktoken

from src.backend.db_config import get_db_config
from src.backend.indexer import DocumentIndexer
from src.backend.content_detector import ContentDetector
from src.backend.adaptive_chunker import AdaptiveChunker, chunk_document_adaptive
from src.backend.chunking_config import get_chunking_config
import psycopg2

class ChunkingEvaluator:
    """Evaluate and compare chunking strategies."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.db_config = get_db_config()
    
    def evaluate_file(self, filepath: Path) -> Dict:
        """
        Evaluate chunking on a single file.
        
        Returns:
            Dictionary with metrics
        """
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fixed-size chunking (baseline)
        fixed_chunks = self._chunk_fixed(content)
        
        # Adaptive chunking
        config = get_chunking_config()
        detector = ContentDetector(
            min_table_rows=config['min_table_rows'],
            min_list_items=config['min_list_items']
        )
        chunker = AdaptiveChunker()
        adaptive_chunks = chunk_document_adaptive(content, chunker, detector)
        
        # Calculate metrics
        metrics = {
            'filename': filepath.name,
            'doc_length_chars': len(content),
            'doc_length_tokens': len(self.tokenizer.encode(content)),
            
            # Fixed chunking metrics
            'fixed_chunk_count': len(fixed_chunks),
            'fixed_avg_size': statistics.mean([len(self.tokenizer.encode(c)) for c in fixed_chunks]),
            'fixed_std_size': statistics.stdev([len(self.tokenizer.encode(c)) for c in fixed_chunks]) if len(fixed_chunks) > 1 else 0,
            
            # Adaptive chunking metrics
            'adaptive_chunk_count': len(adaptive_chunks),
            'adaptive_avg_size': statistics.mean([c['token_count'] for c in adaptive_chunks]),
            'adaptive_std_size': statistics.stdev([c['token_count'] for c in adaptive_chunks]) if len(adaptive_chunks) > 1 else 0,
            
            # Content type distribution
            'chunk_types': self._count_chunk_types(adaptive_chunks),
            
            # Quality metrics
            'mid_sentence_splits': self._count_mid_sentence_splits(fixed_chunks),
            'adaptive_sentence_aware': self._check_sentence_boundaries(adaptive_chunks, content),
        }
        
        return metrics
    
    def _chunk_fixed(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Fixed-size chunking (baseline)."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += chunk_size - overlap
        
        return chunks
    
    def _count_chunk_types(self, chunks: List[Dict]) -> Dict[str, int]:
        """Count chunks by type."""
        types = {}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            types[chunk_type] = types.get(chunk_type, 0) + 1
        return types
    
    def _count_mid_sentence_splits(self, chunks: List[str]) -> int:
        """Count how many chunks end mid-sentence."""
        import re
        sentence_end = re.compile(r'[.!?]\s*$')
        
        mid_sentence = 0
        for chunk in chunks[:-1]:  # Exclude last chunk
            if not sentence_end.search(chunk.strip()):
                mid_sentence += 1
        
        return mid_sentence
    
    def _check_sentence_boundaries(self, chunks: List[Dict], original_text: str) -> bool:
        """Check if adaptive chunks respect sentence boundaries."""
        import re
        sentence_end = re.compile(r'[.!?]\s*$')
        
        narrative_chunks = [c for c in chunks if c.get('chunk_type') == 'narrative']
        if not narrative_chunks:
            return True
        
        # Check if most narrative chunks end at sentence boundaries
        boundary_count = sum(1 for c in narrative_chunks[:-1] if sentence_end.search(c['text'].strip()))
        
        return boundary_count / len(narrative_chunks) > 0.8 if narrative_chunks else False
    
    def get_database_stats(self) -> Dict:
        """Get statistics from indexed chunks in database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT doc_id) as total_docs,
                    COUNT(DISTINCT ticker) as total_tickers
                FROM document_chunks
            """)
            overall = cursor.fetchone()
            
            # Chunk type distribution
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count
                FROM document_chunks
                WHERE chunk_type IS NOT NULL
                GROUP BY chunk_type
                ORDER BY count DESC
            """)
            chunk_types = dict(cursor.fetchall())
            
            # Average chunk size by type (approximate from content length)
            cursor.execute("""
                SELECT 
                    chunk_type,
                    AVG(LENGTH(content)) as avg_chars,
                    MIN(LENGTH(content)) as min_chars,
                    MAX(LENGTH(content)) as max_chars
                FROM document_chunks
                WHERE chunk_type IS NOT NULL
                GROUP BY chunk_type
            """)
            size_stats = {row[0]: {'avg': row[1], 'min': row[2], 'max': row[3]} 
                         for row in cursor.fetchall()}
            
            cursor.close()
            conn.close()
            
            return {
                'total_chunks': overall[0] if overall else 0,
                'total_docs': overall[1] if overall else 0,
                'total_tickers': overall[2] if overall else 0,
                'chunk_type_distribution': chunk_types,
                'size_stats_by_type': size_stats
            }
        
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def print_metrics(self, metrics: Dict):
        """Pretty print evaluation metrics."""
        print("\n" + "="*60)
        print(f"CHUNKING EVALUATION: {metrics['filename']}")
        print("="*60)
        
        print(f"\nDocument Size:")
        print(f"  Characters: {metrics['doc_length_chars']:,}")
        print(f"  Tokens: {metrics['doc_length_tokens']:,}")
        
        print(f"\nFixed-Size Chunking (Baseline):")
        print(f"  Total chunks: {metrics['fixed_chunk_count']}")
        print(f"  Avg size: {metrics['fixed_avg_size']:.1f} tokens")
        print(f"  Std dev: {metrics['fixed_std_size']:.1f} tokens")
        print(f"  Mid-sentence splits: {metrics['mid_sentence_splits']} ({metrics['mid_sentence_splits']/max(metrics['fixed_chunk_count'],1)*100:.1f}%)")
        
        print(f"\nAdaptive Chunking:")
        print(f"  Total chunks: {metrics['adaptive_chunk_count']}")
        print(f"  Avg size: {metrics['adaptive_avg_size']:.1f} tokens")
        print(f"  Std dev: {metrics['adaptive_std_size']:.1f} tokens")
        print(f"  Sentence-aware: {'Yes' if metrics['adaptive_sentence_aware'] else 'No'}")
        
        print(f"\nChunk Type Distribution:")
        for chunk_type, count in sorted(metrics['chunk_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / metrics['adaptive_chunk_count'] * 100
            print(f"  {chunk_type}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*60)
    
    def print_database_stats(self, stats: Dict):
        """Pretty print database statistics."""
        if not stats:
            print("\nNo database statistics available")
            return
        
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        
        print(f"\nOverall:")
        print(f"  Total chunks: {stats.get('total_chunks', 0):,}")
        print(f"  Total documents: {stats.get('total_docs', 0):,}")
        print(f"  Total tickers: {stats.get('total_tickers', 0):,}")
        
        chunk_types = stats.get('chunk_type_distribution', {})
        if chunk_types:
            print(f"\nChunk Type Distribution:")
            total = sum(chunk_types.values())
            for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total * 100 if total > 0 else 0
                print(f"  {chunk_type or 'NULL'}: {count:,} ({percentage:.1f}%)")
        
        size_stats = stats.get('size_stats_by_type', {})
        if size_stats:
            print(f"\nAverage Size by Type (characters):")
            for chunk_type, stats_dict in sorted(size_stats.items()):
                print(f"  {chunk_type}:")
                print(f"    Avg: {stats_dict['avg']:.0f} chars")
                print(f"    Range: {stats_dict['min']:.0f} - {stats_dict['max']:.0f} chars")
        
        print("\n" + "="*60)


def main():
    """Main evaluation entry point."""
    evaluator = ChunkingEvaluator()
    
    # Check if a specific file was provided
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        if filepath.exists():
            print(f"Evaluating chunking on: {filepath}")
            metrics = evaluator.evaluate_file(filepath)
            evaluator.print_metrics(metrics)
        else:
            print(f"File not found: {filepath}")
            sys.exit(1)
    else:
        # Evaluate on sample file
        sample_dir = Path("edgar_corpus")
        if sample_dir.exists():
            txt_files = list(sample_dir.glob("*.txt"))
            if txt_files:
                sample_file = txt_files[0]
                print(f"Evaluating chunking on sample: {sample_file}")
                metrics = evaluator.evaluate_file(sample_file)
                evaluator.print_metrics(metrics)
            else:
                print("No .txt files found in edgar_corpus/")
        else:
            print("edgar_corpus/ directory not found")
    
    # Always show database stats if available
    print("\n")
    db_stats = evaluator.get_database_stats()
    evaluator.print_database_stats(db_stats)


if __name__ == "__main__":
    main()
