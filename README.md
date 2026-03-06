# SEC EDGAR RAG System

Chat with SEC EDGAR filings using AI-powered semantic search and LLM-generated answers.

## Quick Start

```bash
make setup
make start
make index
```

**Access:**

- **Chat UI**: http://localhost:8501 (Ask questions, get AI answers)
- **Debug View**: http://localhost:8501/RAG_Debug (Raw search results)
- **API Docs**: http://localhost:8000/docs

## What It Does

- Loads 246 SEC 10-K and 10-Q reports
- Chunks documents for search
- Creates embeddings for semantic matching
- Uses Ollama LLM to generate answers with evidence
- Streams responses in real-time

## Architecture

```
User Question
    ↓
Vector Search (pgvector)
    ↓
Relevant Document Chunks
    ↓
Ollama LLM (qwen3.5:4b)
    ↓
AI-Generated Answer + Evidence
```

**Components:**
- PostgreSQL + pgvector: Vector database
- FastAPI: Backend API
- Streamlit: Web interface (2 pages)
- Ollama: Local LLM hosting
- sentence-transformers: Embeddings

## Requirements

- Docker and Docker Compose
- 8GB+ RAM
- 15GB+ disk space (includes ~2.7GB Ollama model)

## Setup

### 1. Install Docker

Download from [docker.com](https://www.docker.com/get-started)

### 2. Start Services

```bash
make start
```

Services:
- PostgreSQL with pgvector (port 5432)
- FastAPI backend (port 8000)
- Streamlit UI (port 8501)
- Ollama LLM service (port 11434)

### 3. Index Documents

First time only (takes 10-20 minutes):

```bash
make index
```

### 4. Ollama Model

The Ollama model (qwen3.5:4b) downloads automatically on first use.

To manually pull the model:

```bash
make pull-model
```

Check installed models:

```bash
make check-ollama
```

**Note:** Model download is ~2.7GB and happens once. It persists across container restarts.

## Commands

```bash
make help         # Show all commands
make start        # Start services
make stop         # Stop services
make restart      # Restart services
make logs         # View logs
make health       # Check API health
make stats        # Database statistics
make test         # Run unit tests
make clean        # Remove everything
make pull-model   # Pull Ollama model
make check-ollama # Check Ollama models
```

## Usage

### Chat Interface (Recommended)

1. Go to http://localhost:8501
2. Ask a question (e.g., "What was Apple's revenue in 2024?")
3. Watch the AI generate an answer in real-time
4. Click the evidence dropdown to see source documents
5. Optional: filter by ticker or filing type in sidebar

### Debug View (Raw Search)

1. Go to http://localhost:8501/RAG_Debug
2. Enter your question
3. See raw vector search results with similarity scores
4. Useful for testing and debugging retrieval quality

### API (FastAPI)

Chat with LLM:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple'\''s revenue in 2024?",
    "top_k": 5,
    "ticker": "AAPL"
  }'
```

Raw vector search:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple'\''s revenue in 2024?",
    "top_k": 5,
    "ticker": "AAPL"
  }'
```

Stats:

```bash
curl http://localhost:8000/stats
```

API docs: http://localhost:8000/docs

## Project Structure

```
SEC-RAG/
├── edgar_corpus/       # 246 SEC filing txt files
├── src/
│   ├── indexer.py     # Document processing
│   ├── retriever.py   # Vector search
│   ├── llm.py         # Ollama LLM client
│   ├── api.py         # FastAPI backend
│   ├── app.py         # Streamlit chat UI
│   └── pages/
│       └── 2_RAG_Debug.py  # Debug view
├── tests/
│   └── test_rag.py    # Unit tests
├── docker-compose.yml
├── Dockerfile
├── init.sql
├── Makefile
└── pyproject.toml     # UV dependencies
```

## Configuration

The `.env` file controls all settings. Makefile commands automatically use these values.

Key settings:

- `API_PORT`: API port (default: 8000)
- `STREAMLIT_PORT`: Streamlit port (default: 8501)
- `POSTGRES_PASSWORD`: Database password
- `OLLAMA_MODEL`: LLM model (default: qwen3.5:4b)
- `OLLAMA_HOST`: Ollama host (default: ollama)
- `OLLAMA_PORT`: Ollama port (default: 11434)
- `CHUNK_SIZE`: Tokens per chunk (default: 512)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `EMBEDDING_MODEL`: Model (default: all-MiniLM-L6-v2)

## Testing

```bash
make test       # Run tests
make test-cov   # With coverage
```

## Troubleshooting

**Services won't start:**
- Check Docker is running
- Check ports 5432, 8000, 8501, 11434 are free

**Ollama model not downloading:**
- Check Ollama container: `docker logs edgar_ollama`
- Manually pull: `make pull-model`
- Check disk space (need 3GB+ free)

**Indexing fails:**
- Check postgres health: `docker-compose ps`
- View logs: `docker-compose logs api`

**No search results:**
- Verify indexing completed: `make stats`
- Check API: http://localhost:8000/health

**Out of memory:**
- Increase Docker memory to 8GB+

## Local Development

Without Docker:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start postgres only
docker-compose up postgres -d

# Run API
uv run uvicorn src.api:app --reload

# Run Streamlit
uv run streamlit run src/app.py
```

## Tech Stack

- **Vector DB**: PostgreSQL + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (qwen3.5:4b)
- **Backend**: FastAPI
- **Frontend**: Streamlit (multi-page)
- **Package Manager**: UV
- **Deployment**: Docker Compose
