from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
from contextlib import asynccontextmanager

from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever


# Shared instances
indexer = None
retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global indexer, retriever
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    yield


app = FastAPI(
    title="SEC EDGAR RAG API",
    description="Query SEC filings using vector search",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    ticker: Optional[str] = Field(None, description="Filter by ticker symbol")
    filing_type: Optional[str] = Field(None, description="Filter by filing type (10-K, 10-Q)")


class SearchResultResponse(BaseModel):
    chunk_id: int
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    quarter: Optional[str]
    content: str
    similarity: float


class QueryResponse(BaseModel):
    query: str
    results: List[SearchResultResponse]
    took_ms: float
    total_results: int


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    total_tickers: int
    by_filing_type: Dict[str, int]


class IndexResponse(BaseModel):
    status: str
    stats: Dict[str, int]


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SEC EDGAR RAG API",
        "version": "0.1.0",
        "endpoints": ["/health", "/stats", "/query", "/index"],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        stats = retriever.get_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "indexed_documents": stats["total_documents"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    try:
        stats = retriever.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Search for relevant document chunks."""
    start_time = time.time()
    
    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            ticker=request.ticker,
            filing_type=request.filing_type,
        )
        
        took_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            results=[SearchResultResponse(**r.to_dict()) for r in results],
            took_ms=took_ms,
            total_results=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/index", response_model=IndexResponse)
async def run_indexing():
    """Trigger indexing of all documents in the corpus."""
    try:
        stats = indexer.index_corpus()
        return IndexResponse(status="completed", stats=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.delete("/index")
async def clear_index():
    """Clear all indexed data."""
    try:
        indexer.clear_index()
        return {"status": "cleared", "message": "All indexed data removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
