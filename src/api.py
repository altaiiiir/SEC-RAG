from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import json
from contextlib import asynccontextmanager

from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.llm import OllamaClient


# Shared instances
indexer = None
retriever = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global indexer, retriever, llm_client
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    llm_client = OllamaClient()
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


class AskRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks")
    ticker: Optional[str] = Field(None, description="Filter by ticker symbol")
    filing_type: Optional[str] = Field(None, description="Filter by filing type (10-K, 10-Q)")


class Evidence(BaseModel):
    content: str
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    similarity: float


# Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SEC EDGAR RAG API",
        "version": "0.1.0",
        "endpoints": ["/health", "/stats", "/query", "/ask", "/index"],
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


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Ask a question and get an LLM-generated answer with evidence.
    Returns streaming response with answer followed by evidence metadata.
    """
    try:
        # Retrieve relevant context
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            ticker=request.ticker,
            filing_type=request.filing_type,
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Ticker: {result.ticker}\n"
                f"Filing: {result.filing_type}\n"
                f"Date: {result.filing_date}\n"
                f"Content: {result.content}\n"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = (
            f"Based on these SEC filings:\n\n{context}\n\n"
            f"Question: {request.query}\n\n"
            f"Answer:"
        )
        
        # Build evidence metadata
        evidence_list = [
            Evidence(
                content=r.content,
                ticker=r.ticker,
                filing_type=r.filing_type,
                filing_date=r.filing_date,
                similarity=r.similarity
            ).model_dump()
            for r in results
        ]
        
        async def generate_response():
            """Stream the LLM response followed by evidence."""
            # First, stream the LLM response
            async for chunk in llm_client.generate_stream(prompt):
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
            
            # Then send evidence
            yield f"data: {json.dumps({'type': 'evidence', 'data': evidence_list})}\n\n"
            
            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
