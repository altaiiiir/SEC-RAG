.PHONY: help setup start stop restart logs clean index health stats shell build reset test test-cov ensure-env pull-model check-ollama clear-db snapshot restore

# Load environment variables from .env if it exists
ifneq (,$(wildcard .env))
	include .env
	export
endif

# Set defaults if not in .env
API_PORT ?= 8000
STREAMLIT_PORT ?= 8501
API_HOST ?= localhost
POSTGRES_USER ?= postgres
POSTGRES_DB ?= edgar_rag

# Default target
help:
	@echo "SEC EDGAR RAG - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup & Start:"
	@echo "  make setup     - Create .env file (auto-detects OS)"
	@echo "  make start     - Start all services"
	@echo "  make stop      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo ""
	@echo "Indexing:"
	@echo "  make index        - Index all documents"
	@echo "  make index n=5    - Index only 5 documents (for testing)"
	@echo "  make clear        - Clear all indexed data"
	@echo "  make clear-db     - Drop all data and schema from database"
	@echo ""
	@echo "Snapshots (share a pre-indexed corpus without re-indexing):"
	@echo "  make snapshot     - Dump indexed DB to snapshots/ directory"
	@echo "  make restore f=snapshots/corpus.dump - Restore from a dump file"
	@echo ""
	@echo "Status:"
	@echo "  make health       - Check API health"
	@echo "  make stats        - Show database statistics"
	@echo "  make logs         - View logs from all services"
	@echo "  make check-ollama - Check Ollama models"
	@echo ""
	@echo "Testing (Docker only; no venv or running API required):"
	@echo "  make test      - Run tests in Docker (starts postgres + test container)"
	@echo "  make test-cov  - Run tests with coverage"
	@echo ""
	@echo "Development:"
	@echo "  make shell       - Open shell in API container"
	@echo "  make build       - Build containers"
	@echo "  make clean       - Stop and remove containers"
	@echo "  make reset       - Full reset (clean + start services)"
	@echo "  make pull-model  - Pull Ollama model ($(OLLAMA_MODEL))"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup"
	@echo "  2. make start"
	@echo "  3. make index"

# Initial setup
setup:
	@echo "Setting up SEC EDGAR RAG..."
	@python -c "import shutil; from pathlib import Path; dst=Path('.env'); (shutil.copy('.env.example',dst) and print('Created .env file')) if not dst.exists() else print('.env already exists')"
	@echo "Setup complete! Run: make start"

# Start services
start:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "  - Streamlit UI: http://$(API_HOST):$(STREAMLIT_PORT)"
	@echo "  - API: http://$(API_HOST):$(API_PORT)"
	@echo "  - API Docs: http://$(API_HOST):$(API_PORT)/docs"

# Stop services
stop:
	@echo "Stopping services..."
	docker-compose down

# Restart services
restart: stop start

# View logs
logs:
	docker-compose logs -f

# Index documents
index:
	@echo "Indexing documents..."
	@docker-compose exec api python -m src.backend.indexer $(if $(n),$(n),)
	@echo "Indexing complete!"

# Clear index
clear:
	@echo "Clearing index..."
	docker-compose exec api python -c "from src.backend.indexer import DocumentIndexer; DocumentIndexer().clear_index()"
	@echo "Index cleared!"

# Clear database (drop all data and schema)
clear-db:
	@echo "WARNING: This will drop all data and schema from the database!"
	@echo "Dropping database schema and data..."
	@docker-compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "DROP TABLE IF EXISTS document_chunks CASCADE; DROP EXTENSION IF EXISTS vector CASCADE;"
	@echo "Database cleared!"
	@echo "Reinitializing database schema..."
	@docker-compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f /docker-entrypoint-initdb.d/init.sql
	@echo "Database reinitialized successfully!"

# Check health
health:
	@curl -s http://$(API_HOST):$(API_PORT)/health | python -m json.tool

# Get stats
stats:
	@curl -s http://$(API_HOST):$(API_PORT)/stats | python -m json.tool

# Ensure .env exists (no Python required; uses Docker). Safe to run every time.
ensure-env:
	@docker run --rm -v "$(CURDIR):/w" -w /w alpine sh -c "cp -n .env.example .env 2>/dev/null || true"

# Run tests in Docker: starts postgres + test container (no venv or running API needed)
test: ensure-env
	@echo "Running tests (Docker: postgres + test runner)..."
	docker-compose --profile test run --rm test

# Run tests with coverage
test-cov: ensure-env
	@echo "Running tests with coverage..."
	docker-compose --profile test run --rm test pytest tests/ -v --cov=src --cov-report=term-missing

# Open shell
shell:
	docker-compose exec api /bin/bash

# Clean everything
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "All containers and volumes removed!"

# Build containers
build:
	@echo "Building containers..."
	docker-compose build

# Full reset
reset: clean setup
	@echo "Starting services..."
	@docker-compose up -d
	@echo "Services started. To index: make index"

# Pull Ollama model
pull-model:
	@echo "Pulling Ollama model..."
	@echo "This may take a few minutes..."
	docker exec edgar_ollama ollama pull $(OLLAMA_MODEL)
	@echo "Model pulled successfully!"

# Check Ollama models
check-ollama:
	@echo "Installed Ollama models:"
	@docker exec edgar_ollama ollama list

# Snapshot: dump the fully-indexed postgres DB to a portable binary file.
# Share the .dump file so others can run make restore instead of make index.
snapshot:
	@docker run --rm -v "$(CURDIR):/w" -w /w alpine mkdir -p snapshots
	@docker-compose exec -T postgres pg_dump -U $(POSTGRES_USER) -Fc $(POSTGRES_DB) > snapshots/corpus.dump
	@echo "Snapshot saved to snapshots/corpus.dump"

# Restore: load a snapshot into the running postgres container.
# Usage: make restore f=snapshots/corpus.dump
restore:
	@docker run --rm -v "$(CURDIR):/w" alpine sh -c 'test -n "$(f)" || (echo "Usage: make restore f=snapshots/corpus.dump" && exit 1)'
	@docker-compose exec -T postgres pg_restore -U $(POSTGRES_USER) -d $(POSTGRES_DB) --clean --if-exists < $(f)
	@echo "Restored from $(f)"

