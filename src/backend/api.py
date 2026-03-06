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


app = FastAPI(title="SEC EDGAR RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    ticker: Optional[str] = None
    filing_type: Optional[str] = None


class SearchResult(BaseModel):
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
    results: List[SearchResult]
    took_ms: float
    total_results: int


class Evidence(BaseModel):
    content: str
    ticker: str
    filing_type: str
    filing_date: Optional[str]
    similarity: float


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
        results = retriever.search(query=request.query, top_k=request.top_k, ticker=request.ticker, filing_type=request.filing_type)
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
        results = retriever.search(query=request.query, top_k=request.top_k, ticker=request.ticker, filing_type=request.filing_type)
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        context = "\n\n---\n\n".join([
            f"Company: {r.ticker}, Filing: {r.filing_type}, Date: {r.filing_date}\n{r.content}"
            for r in results
        ])
        
        prompt = (
            f"You are a financial analyst assistant. Answer the question based on the SEC filing excerpts below. "
            f"Provide a direct, concise answer without referencing document numbers or sources - the evidence will be shown separately.\n\n"
            f"SEC Filing Excerpts:\n{context}\n\n"
            f"Question: {request.query}\n\n"
            f"Answer:"
        )
        
        evidence_list = [Evidence(**{k: v for k, v in r.to_dict().items() if k in ['content', 'ticker', 'filing_type', 'filing_date', 'similarity']}).model_dump() for r in results]
        
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
