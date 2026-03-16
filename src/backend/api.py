from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import json
from contextlib import asynccontextmanager

from src.backend.indexer import DocumentIndexer
from src.backend.retriever import DocumentRetriever
from src.backend.llm import OllamaClient
from src.backend.query_parser import parse_query, suggest_top_k
from src.backend.reranker import Reranker

indexer = None
retriever = None
llm_client = None
reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global indexer, retriever, llm_client, reranker
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    llm_client = OllamaClient()
    reranker = Reranker()
    yield


app = FastAPI(title="SEC EDGAR RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    chunk_type: Optional[str] = None  # Filter by chunk type


class SearchResult(BaseModel):
    chunk_id: int
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    quarter: Optional[str]
    content: str
    similarity: float
    chunk_type: Optional[str] = None
    section_name: Optional[str] = None
    table_id: Optional[str] = None
    row_range: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    took_ms: float
    total_results: int


class Evidence(BaseModel):
    content: str
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    similarity: float
    chunk_type: Optional[str] = None
    section_name: Optional[str] = None
    table_id: Optional[str] = None
    row_range: Optional[str] = None


# Endpoints
@app.get("/")
async def root():
    return {"name": "SEC EDGAR RAG API", "version": "0.1.0", "endpoints": ["/health", "/stats", "/query", "/ask", "/index"]}


@app.get("/health")
async def health():
    try:
        stats = retriever.get_stats()
        return {"status": "healthy", "database": "connected", "indexed_documents": stats["total_documents"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


@app.get("/stats")
async def get_stats():
    try:
        return retriever.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/query")
async def query(request: QueryRequest):
    start_time = time.time()
    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            ticker=request.ticker,
            filing_type=request.filing_type,
            chunk_type=request.chunk_type
        )
        return QueryResponse(
            query=request.query,
            results=[SearchResult(**r.to_dict()) for r in results],
            took_ms=(time.time() - start_time) * 1000,
            total_results=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/index")
async def run_indexing():
    try:
        stats = indexer.index_corpus()
        return {"status": "completed", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.delete("/index")
async def clear_index():
    try:
        indexer.clear_index()
        return {"status": "cleared", "message": "All indexed data removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


@app.post("/ask")
async def ask(request: QueryRequest):
    """Ask a question and get an LLM-generated answer with evidence."""
    try:
        # Parse query to extract tickers and section hints
        parsed = parse_query(request.query)
        
        # Determine retrieval parameters
        tickers = parsed.get('tickers') if parsed.get('tickers') else None
        section_hint = parsed.get('section_hint')
        
        # Override with user-provided filters if specified
        if request.ticker:
            tickers = [request.ticker]
        
        # Adjust top_k for multi-company queries (retrieve more for reranking)
        initial_top_k = suggest_top_k(parsed, default=request.top_k) * 4  # Get 4x for reranking
        initial_top_k = min(initial_top_k, 50)  # Cap at 50
        
        # Stage 1: Initial retrieval with embedding search
        results = retriever.search(
            query=request.query,
            top_k=initial_top_k,
            tickers=tickers,
            filing_type=request.filing_type,
            chunk_type=request.chunk_type,
            section_name=section_hint
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Stage 2: Rerank to get best results
        final_results = reranker.rerank(request.query, results, top_k=request.top_k)
        
        # Build context with enhanced metadata
        context_parts = []
        for r in final_results:
            meta = f"Company: {r.ticker}, Filing: {r.filing_type}, Date: {r.filing_date}"
            if r.section_name:
                meta += f", Section: {r.section_name}"
            context_parts.append(f"{meta}\n{r.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = llm_client.format_prompt(context=context, query=request.query)
        
        # Include all metadata in evidence
        evidence_list = [Evidence(**{
            'content': r.content,
            'ticker': r.ticker,
            'filing_type': r.filing_type,
            'filing_date': r.filing_date,
            'similarity': r.similarity,
            'chunk_type': r.chunk_type,
            'section_name': r.section_name,
            'table_id': r.table_id,
            'row_range': r.row_range
        }).model_dump() for r in final_results]
        
        async def generate_response():
            async for chunk in llm_client.generate_stream(prompt):
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'evidence', 'data': evidence_list})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(generate_response(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
