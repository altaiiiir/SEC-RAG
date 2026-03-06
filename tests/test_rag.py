"""
Unit tests for SEC EDGAR RAG system.

Run with: make test (inside Docker) or pytest tests/ (locally with dependencies)
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFilenameParser:
    """Tests for filename parsing logic."""
    
    def test_parse_filename_standard(self):
        """Test parsing standard filename format."""
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
            'tiktoken': Mock(),
        }):
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            metadata = indexer._parse_filename("AAPL_10K_2024Q3_2024-11-01_full.txt")
            
            assert metadata["ticker"] == "AAPL"
            assert metadata["filing_type"] == "10K"
            assert metadata["quarter"] == "2024Q3"
            assert metadata["filing_date"] == "2024-11-01"
    
    def test_parse_filename_no_quarter(self):
        """Test parsing filename without quarter."""
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
            'tiktoken': Mock(),
        }):
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            metadata = indexer._parse_filename("TSLA_10K_2026-01-29_full.txt")
            
            assert metadata["ticker"] == "TSLA"
            assert metadata["filing_type"] == "10K"
            assert metadata["filing_date"] == "2026-01-29"
            assert metadata["quarter"] is None
    
    def test_parse_filename_variations(self):
        """Test parsing various filename formats."""
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
            'tiktoken': Mock(),
        }):
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            
            # With quarter
            meta1 = indexer._parse_filename("MSFT_10Q_2024Q2_2024-07-30_full.txt")
            assert meta1["ticker"] == "MSFT"
            assert meta1["filing_type"] == "10Q"
            assert meta1["quarter"] == "2024Q2"
            
            # Without quarter  
            meta2 = indexer._parse_filename("NVDA_10K_2025-02-26_full.txt")
            assert meta2["ticker"] == "NVDA"
            assert meta2["filing_type"] == "10K"


class TestSearchResult:
    """Tests for SearchResult class."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
        }):
            from src.retriever import SearchResult
            
            result = SearchResult(
                chunk_id=1,
                doc_id="AAPL_10K_2024",
                ticker="AAPL",
                filing_type="10-K",
                filing_date="2024-11-01",
                quarter="2024Q3",
                content="Revenue was $100B",
                similarity=0.95
            )
            
            assert result.chunk_id == 1
            assert result.ticker == "AAPL"
            assert result.similarity == 0.95
    
    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
        }):
            from src.retriever import SearchResult
            
            result = SearchResult(
                chunk_id=1,
                doc_id="AAPL_10K_2024",
                ticker="AAPL",
                filing_type="10-K",
                filing_date="2024-11-01",
                quarter="2024Q3",
                content="Revenue was $100B",
                similarity=0.95
            )
            
            result_dict = result.to_dict()
            
            assert isinstance(result_dict, dict)
            assert result_dict["ticker"] == "AAPL"
            assert result_dict["similarity"] == 0.95
            assert "content" in result_dict
            assert result_dict["content"] == "Revenue was $100B"


class TestDatabaseOperations:
    """Tests for database-related logic."""
    
    def test_db_config_from_env(self):
        """Test database config reads from environment."""
        with patch.dict('os.environ', {
            'POSTGRES_HOST': 'testhost',
            'POSTGRES_PORT': '5433',
            'POSTGRES_DB': 'testdb',
        }):
            with patch.dict('sys.modules', {
                'psycopg2': Mock(),
                'sentence_transformers': Mock(),
                'tiktoken': Mock(),
            }):
                from src.indexer import DocumentIndexer
                indexer = DocumentIndexer()
                
                assert indexer.db_config['host'] == 'testhost'
                assert indexer.db_config['port'] == '5433'
                assert indexer.db_config['database'] == 'testdb'
    
    def test_db_config_defaults(self):
        """Test database config uses defaults when not in Docker."""
        # In Docker, host defaults to 'postgres', locally to 'localhost'
        with patch.dict('sys.modules', {
            'psycopg2': Mock(),
            'sentence_transformers': Mock(),
            'tiktoken': Mock(),
        }):
            from src.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            
            # Just verify defaults exist (could be localhost or postgres)
            assert indexer.db_config['host'] in ['localhost', 'postgres']
            assert indexer.db_config['database'] == 'edgar_rag'


class TestRegexPatterns:
    """Test regex patterns used in the system."""
    
    def test_quarter_pattern(self):
        """Test quarter pattern matching."""
        pattern = r"\d{4}Q\d"
        
        assert re.match(pattern, "2024Q3")
        assert re.match(pattern, "2022Q1")
        assert re.match(pattern, "2024Q5")  # Pattern matches Q5 (validation happens elsewhere)
        assert not re.match(pattern, "Q1")  # Missing year
    
    def test_date_pattern(self):
        """Test date pattern matching."""
        pattern = r"\d{4}-\d{2}-\d{2}"
        
        assert re.match(pattern, "2024-11-01")
        assert re.match(pattern, "2022-01-31")
        assert not re.match(pattern, "24-11-01")  # Short year
        assert not re.match(pattern, "2024-1-1")  # Missing zero padding


class TestConstants:
    """Test system constants and configurations."""
    
    def test_chunk_size_config(self):
        """Test chunk size configuration."""
        with patch.dict('os.environ', {'CHUNK_SIZE': '1024'}):
            with patch.dict('sys.modules', {
                'psycopg2': Mock(),
                'sentence_transformers': Mock(),
                'tiktoken': Mock(),
            }):
                from src.indexer import DocumentIndexer
                indexer = DocumentIndexer()
                assert indexer.chunk_size == 1024
    
    def test_chunk_overlap_config(self):
        """Test chunk overlap configuration."""
        with patch.dict('os.environ', {'CHUNK_OVERLAP': '100'}):
            with patch.dict('sys.modules', {
                'psycopg2': Mock(),
                'sentence_transformers': Mock(),
                'tiktoken': Mock(),
            }):
                from src.indexer import DocumentIndexer
                indexer = DocumentIndexer()
                assert indexer.chunk_overlap == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
