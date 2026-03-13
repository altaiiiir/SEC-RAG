"""Chunking configuration. Centralized hyperparameters for SEC filing chunking."""
import os
from typing import Dict, Any


def get_chunking_config() -> Dict[str, Any]:
    """Chunking hyperparameters from environment variables with defaults."""
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "min_chunk_size": int(os.getenv("MIN_CHUNK_SIZE", "100")),
        "enable_sentence_boundaries": os.getenv("ENABLE_SENTENCE_BOUNDARIES", "true").lower() == "true",
        "table_row_chunk_size": int(os.getenv("TABLE_ROW_CHUNK_SIZE", "15")),
    }
