# SEC EDGAR RAG System

Query SEC EDGAR filings using semantic search.

## Quick Start

```bash
make setup
make start
make index
```

**Access:**

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## What It Does

- Loads 246 SEC 10-K and 10-Q reports
- Chunks documents for search
- Creates embeddings for semantic matching
- Answers questions about filings

## Requirements

- Docker and Docker Compose
- 8GB+ RAM
- 10GB+ disk space

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

### 3. Index Documents

First time only (takes 10-20 minutes):

```bash
make index
```

## Commands

```bash
make help      # Show all commands
make start     # Start services
make stop      # Stop services
make restart   # Restart services
make logs      # View logs
make health    # Check API health
make stats     # Database statistics
make test      # Run unit tests
make clean     # Remove everything
```

## Usage

### Web UI (Streamlit)

1. Go to http://localhost:8501
2. Enter your question
3. Optional: filter by ticker or filing type
4. Click Search

### API (FastAPI)

Query:

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
│   ├── api.py         # FastAPI backend
│   └── app.py         # Streamlit UI
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
- Check ports 5432, 8000, 8501 are free

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
- **Embeddings**: sentence-transformers
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Package Manager**: UV
- **Deployment**: Docker Compose
