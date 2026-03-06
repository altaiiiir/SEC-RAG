from typing import List, Dict, Optional
import psycopg2
from psycopg2 import pool
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import time

from src.backend.db_config import get_db_config, get_embedding_model_name

# Global connection pool
_connection_pool = None

def get_connection_pool():
    """Get or create connection pool."""
    global _connection_pool
    if _connection_pool is None:
        db_config = get_db_config()
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            **db_config
        )
    return _connection_pool

@lru_cache(maxsize=1)
def get_embedding_model():
    """Get cached embedding model."""
    model_name = get_embedding_model_name()
    return SentenceTransformer(model_name)

# Simple query cache
_query_cache = {}
_cache_ttl = 300  # 5 minutes

def get_cached_query(cache_key: str):
    """Get cached query result if not expired."""
    if cache_key in _query_cache:
        result, timestamp = _query_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            return result
        else:
            del _query_cache[cache_key]
    return None

def cache_query(cache_key: str, result):
    """Cache query result."""
    _query_cache[cache_key] = (result, time.time())
    # Limit cache size
    if len(_query_cache) > 100:
        oldest = min(_query_cache.keys(), key=lambda k: _query_cache[k][1])
        del _query_cache[oldest]

class SearchResult:
    """A single search result with metadata."""
    
    def __init__(self, chunk_id: int, doc_id: str, ticker: str, filing_type: str,
                 filing_date: str, quarter: str, content: str, similarity: float,
                 chunk_type: str = None, section_name: str = None,
                 table_id: str = None, row_range: str = None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.ticker = ticker
        self.filing_type = filing_type
        self.filing_date = filing_date
        self.quarter = quarter
        self.content = content
        self.similarity = similarity
        self.chunk_type = chunk_type
        self.section_name = section_name
        self.table_id = table_id
        self.row_range = row_range
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "quarter": self.quarter,
            "content": self.content,
            "similarity": self.similarity,
            "chunk_type": self.chunk_type,
            "section_name": self.section_name,
            "table_id": self.table_id,
            "row_range": self.row_range,
        }


class DocumentRetriever:
    """Retrieves relevant document chunks using vector similarity search."""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.embedding_model_name = get_embedding_model_name()
        self.model = get_embedding_model()
        self.pool = get_connection_pool()
    
    def _get_db_connection(self):
        """Get database connection from pool."""
        return self.pool.getconn()
    
    def _return_db_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks with caching."""
        
        # Check cache
        cache_key = f"{query}:{top_k}:{ticker}:{filing_type}:{chunk_type}"
        cached = get_cached_query(cache_key)
        if cached is not None:
            return cached
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Build SQL query with optional filters
        sql = """
            SELECT 
                id, doc_id, ticker, filing_type, filing_date, quarter, content,
                1 - (embedding <=> %s::vector) as similarity,
                chunk_type, section_name, table_id, row_range
            FROM document_chunks
            WHERE 1=1
        """
        params = [query_embedding.tolist()]
        
        if ticker:
            sql += " AND ticker = %s"
            params.append(ticker)
            
        if filing_type:
            sql += " AND filing_type = %s"
            params.append(filing_type)
        
        if chunk_type:
            sql += " AND chunk_type = %s"
            params.append(chunk_type)
        
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding.tolist(), top_k])
        
        # Execute query
        conn = self._get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()
            self._return_db_connection(conn)
        
        # Convert to SearchResult objects
        results = [
            SearchResult(
                chunk_id=row[0],
                doc_id=row[1],
                ticker=row[2],
                filing_type=row[3],
                filing_date=str(row[4]) if row[4] else None,
                quarter=row[5],
                content=row[6],
                similarity=float(row[7]),
                chunk_type=row[8],
                section_name=row[9],
                table_id=row[10],
                row_range=row[11],
            )
            for row in rows
        ]
        
        # Cache results
        cache_query(cache_key, results)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics with optimized query."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Optimized query combining multiple aggregations
            cursor.execute("""
                WITH doc_stats AS (
                    SELECT 
                        filing_type,
                        COUNT(DISTINCT doc_id) as doc_count
                    FROM document_chunks
                    GROUP BY filing_type
                )
                SELECT 
                    (SELECT COUNT(*) FROM document_chunks) as total_chunks,
                    (SELECT COUNT(DISTINCT doc_id) FROM document_chunks) as total_docs,
                    (SELECT COUNT(DISTINCT ticker) FROM document_chunks) as total_tickers,
                    (SELECT json_object_agg(filing_type, doc_count) FROM doc_stats) as by_filing_type
            """)
            
            row = cursor.fetchone()
            
            return {
                "total_chunks": row[0] or 0,
                "total_documents": row[1] or 0,
                "total_tickers": row[2] or 0,
                "by_filing_type": row[3] or {},
            }
        finally:
            cursor.close()
            self._return_db_connection(conn)


if __name__ == "__main__":
    retriever = DocumentRetriever()
    
    # Test query
    query = "What was Apple's revenue in 2024?"
    results = retriever.search(query, top_k=3, ticker="AAPL")
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.ticker} - {result.filing_type} ({result.filing_date})")
        print(f"   Similarity: {result.similarity:.4f}")
        print(f"   {result.content[:200]}...")
        print()
