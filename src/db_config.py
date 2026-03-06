"""Shared database and model configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_config():
    """Get database configuration."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "edgar_rag"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }

def get_embedding_model_name():
    """Get embedding model name."""
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_chunk_config():
    """Get chunking configuration."""
    return {
        "size": int(os.getenv("CHUNK_SIZE", "512")),
        "overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
    }
