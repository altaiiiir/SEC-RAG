FROM python:3.11-slim

WORKDIR /app

# Install UV for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using UV
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src/ ./src/
COPY edgar_corpus/ ./edgar_corpus/

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
