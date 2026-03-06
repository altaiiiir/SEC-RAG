from typing import List, Dict, Optional
import psycopg2
from sentence_transformers import SentenceTransformer

from src.backend.db_config import get_db_config, get_embedding_model_name

class SearchResult:
    """A single search result with metadata."""
    
    def __init__(self, chunk_id: int, doc_id: str, ticker: str, filing_type: str,
                 filing_date: str, quarter: str, content: str, similarity: float):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.ticker = ticker
        self.filing_type = filing_type
        self.filing_date = filing_date
        self.quarter = quarter
        self.content = content
        self.similarity = similarity
    
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
        }


class DocumentRetriever:
    """Retrieves relevant document chunks using vector similarity search."""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.embedding_model_name = get_embedding_model_name()
        self.model = SentenceTransformer(self.embedding_model_name)
    
    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity."""
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Build SQL query with optional filters
        sql = """
            SELECT 
                id, doc_id, ticker, filing_type, filing_date, quarter, content,
                1 - (embedding <=> %s::vector) as similarity
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
        
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding.tolist(), top_k])
        
        # Execute query
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert to SearchResult objects
        results = []
        for row in rows:
            result = SearchResult(
                chunk_id=row[0],
                doc_id=row[1],
                ticker=row[2],
                filing_type=row[3],
                filing_date=str(row[4]) if row[4] else None,
                quarter=row[5],
                content=row[6],
                similarity=float(row[7]),
            )
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Total chunks
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cursor.fetchone()[0]
        
        # Unique documents
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM document_chunks")
        total_docs = cursor.fetchone()[0]
        
        # Unique tickers
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM document_chunks")
        total_tickers = cursor.fetchone()[0]
        
        # Documents by filing type
        cursor.execute("""
            SELECT filing_type, COUNT(DISTINCT doc_id) 
            FROM document_chunks 
            GROUP BY filing_type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        
        return {
            "total_chunks": total_chunks,
            "total_documents": total_docs,
            "total_tickers": total_tickers,
            "by_filing_type": by_type,
        }


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
