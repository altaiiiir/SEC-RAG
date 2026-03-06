"""
Chunking configuration with hyperparameters.
Centralized configuration for adaptive chunking strategies.
"""
import os
from typing import Dict, Any

def get_chunking_config() -> Dict[str, Any]:
    """
    Get chunking configuration from environment variables with defaults.
    
    Returns:
        Dictionary with all chunking hyperparameters
    """
    return {
        # Basic chunking parameters
        "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "min_chunk_size": int(os.getenv("MIN_CHUNK_SIZE", "100")),
        
        # Table chunking parameters
        "table_row_chunk_size": int(os.getenv("TABLE_ROW_CHUNK_SIZE", "5")),
        "table_preserve_header": os.getenv("TABLE_PRESERVE_HEADER", "true").lower() == "true",
        
        # Content detection parameters
        "min_table_rows": int(os.getenv("MIN_TABLE_ROWS", "3")),
        "min_list_items": int(os.getenv("MIN_LIST_ITEMS", "2")),
        
        # Feature flags
        "enable_adaptive_chunking": os.getenv("ENABLE_ADAPTIVE_CHUNKING", "true").lower() == "true",
        "enable_sentence_boundaries": os.getenv("ENABLE_SENTENCE_BOUNDARIES", "true").lower() == "true",
        "enable_table_detection": os.getenv("ENABLE_TABLE_DETECTION", "true").lower() == "true",
        "enable_list_detection": os.getenv("ENABLE_LIST_DETECTION", "true").lower() == "true",
        "enable_semantic_overlap": os.getenv("ENABLE_SEMANTIC_OVERLAP", "false").lower() == "true",
        "enable_nlp_detection": os.getenv("ENABLE_NLP_DETECTION", "false").lower() == "true",
        
        # Overlap strategy
        "overlap_strategy": os.getenv("OVERLAP_STRATEGY", "fixed"),  # fixed, semantic, sentence
    }

def get_table_config() -> Dict[str, Any]:
    """Get table-specific chunking configuration."""
    config = get_chunking_config()
    return {
        "row_chunk_size": config["table_row_chunk_size"],
        "preserve_header": config["table_preserve_header"],
        "min_rows": config["min_table_rows"],
    }

def get_list_config() -> Dict[str, Any]:
    """Get list-specific chunking configuration."""
    config = get_chunking_config()
    return {
        "min_items": config["min_list_items"],
        "preserve_context": True,
    }

def get_narrative_config() -> Dict[str, Any]:
    """Get narrative text chunking configuration."""
    config = get_chunking_config()
    return {
        "chunk_size": config["chunk_size"],
        "overlap": config["chunk_overlap"],
        "min_size": config["min_chunk_size"],
        "sentence_boundaries": config["enable_sentence_boundaries"],
        "semantic_overlap": config["enable_semantic_overlap"],
    }

# Expose main config function for easy import
__all__ = [
    "get_chunking_config",
    "get_table_config",
    "get_list_config",
    "get_narrative_config",
]
