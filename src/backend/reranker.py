"""Reranker for 2-stage retrieval to improve answer quality."""
from typing import List
from functools import lru_cache
from sentence_transformers import CrossEncoder
import os


@lru_cache(maxsize=1)
def get_reranker_model():
    """Get cached reranker model."""
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name)


class Reranker:
    """
    Cross-encoder reranker for improving retrieval quality.
    
    Usage:
        # Stage 1: Fast embedding search (top 20)
        candidates = retriever.search(query, top_k=20)
        
        # Stage 2: Precise reranking (top 5)
        reranker = Reranker()
        final_results = reranker.rerank(query, candidates, top_k=5)
    """
    
    def __init__(self):
        """Initialize reranker with cross-encoder model."""
        self.model = get_reranker_model()
    
    def rerank(self, query: str, results: List, top_k: int = 5) -> List:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: User query string
            results: List of SearchResult objects from initial retrieval
            top_k: Number of top results to return
            
        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []
        
        # If we have fewer results than top_k, just return them
        if len(results) <= top_k:
            return results
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, r.content) for r in results]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Sort by reranker score (descending)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return [result for result, score in ranked[:top_k]]
    
    def rerank_with_scores(self, query: str, results: List, top_k: int = 5) -> List[tuple]:
        """
        Rerank and return both results and their reranking scores.
        
        Args:
            query: User query string
            results: List of SearchResult objects
            top_k: Number of top results to return
            
        Returns:
            List of (SearchResult, rerank_score) tuples
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            # Still get scores for debugging
            pairs = [(query, r.content) for r in results]
            scores = self.model.predict(pairs)
            return list(zip(results, scores))
        
        pairs = [(query, r.content) for r in results]
        scores = self.model.predict(pairs)
        
        # Sort by reranker score (descending)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]


if __name__ == "__main__":
    # Test reranker
    from src.backend.retriever import DocumentRetriever, SearchResult
    
    # Mock search results for testing
    class MockResult:
        def __init__(self, content, similarity):
            self.content = content
            self.similarity = similarity
            self.ticker = "AAPL"
            self.filing_type = "10-K"
    
    query = "What are the main business risks?"
    
    # Simulate results where the most relevant is not at the top
    results = [
        MockResult("The company faces competition in the market.", 0.75),
        MockResult("Risk factors include regulatory changes, cybersecurity threats, and supply chain disruptions.", 0.73),
        MockResult("Our business operates globally in multiple markets.", 0.72),
        MockResult("Major risks to our operations include economic downturns and changing consumer preferences.", 0.70),
    ]
    
    reranker = Reranker()
    reranked = reranker.rerank_with_scores(query, results, top_k=2)
    
    print(f"Query: {query}\n")
    print("Reranked results:")
    for i, (result, score) in enumerate(reranked, 1):
        print(f"{i}. Score: {score:.4f}, Content: {result.content[:80]}...")
