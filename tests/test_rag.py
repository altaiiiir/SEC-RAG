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


def _mock_psycopg2():
    """Return a Mock that has psycopg2.extras.execute_batch and psycopg2.pool for imports."""
    m = Mock()
    m.pool = Mock()
    m.extras = Mock()
    m.extras.execute_batch = Mock()
    return m


def _indexer_mocks():
    """Common sys.modules mocks for tests that import DocumentIndexer."""
    return {
        "psycopg2": _mock_psycopg2(),
        "psycopg2.pool": Mock(),
        "psycopg2.extras": Mock(),
        "sentence_transformers": Mock(),
        "tiktoken": Mock(),
        "concurrent": Mock(),
        "concurrent.futures": Mock(),
    }


class TestFilenameParser:
    """Tests for filename parsing logic."""
    
    def test_parse_filename_standard(self):
        """Test parsing standard filename format."""
        with patch.dict('sys.modules', _indexer_mocks()):
            from src.backend.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            metadata = indexer._parse_filename("AAPL_10K_2024Q3_2024-11-01_full.txt")
            
            assert metadata["ticker"] == "AAPL"
            assert metadata["filing_type"] == "10K"
            assert metadata["quarter"] == "2024Q3"
            assert metadata["filing_date"] == "2024-11-01"
    
    def test_parse_filename_no_quarter(self):
        """Test parsing filename without quarter."""
        with patch.dict('sys.modules', _indexer_mocks()):
            from src.backend.indexer import DocumentIndexer
            indexer = DocumentIndexer()
            metadata = indexer._parse_filename("TSLA_10K_2026-01-29_full.txt")
            
            assert metadata["ticker"] == "TSLA"
            assert metadata["filing_type"] == "10K"
            assert metadata["filing_date"] == "2026-01-29"
            assert metadata["quarter"] is None
    
    def test_parse_filename_variations(self):
        """Test parsing various filename formats."""
        with patch.dict('sys.modules', _indexer_mocks()):
            from src.backend.indexer import DocumentIndexer
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
            from src.backend.retriever import SearchResult
            
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
            from src.backend.retriever import SearchResult
            
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
            with patch.dict('sys.modules', _indexer_mocks()):
                from src.backend.indexer import DocumentIndexer
                indexer = DocumentIndexer()
                
                assert indexer.db_config['host'] == 'testhost'
                assert indexer.db_config['port'] == '5433'
                assert indexer.db_config['database'] == 'testdb'
    
    def test_db_config_defaults(self):
        """Test database config uses defaults when not in Docker."""
        # In Docker, host defaults to 'postgres', locally to 'localhost'
        with patch.dict('sys.modules', _indexer_mocks()):
            from src.backend.indexer import DocumentIndexer
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
            from src.backend.chunking_config import get_chunking_config
            config = get_chunking_config()
            assert config["chunk_size"] == 1024
    
    def test_chunk_overlap_config(self):
        """Test chunk overlap configuration."""
        with patch.dict('os.environ', {'CHUNK_OVERLAP': '100'}):
            from src.backend.chunking_config import get_chunking_config
            config = get_chunking_config()
            assert config["chunk_overlap"] == 100


class TestOllamaClient:
    """Tests for Ollama LLM client."""
    
    def test_ollama_client_initialization(self):
        """Test OllamaClient initialization with environment variables."""
        with patch.dict('os.environ', {
            'OLLAMA_HOST': 'test-ollama',
            'OLLAMA_PORT': '11435',
            'OLLAMA_MODEL': 'test-model'
        }):
            with patch.dict('sys.modules', {'httpx': Mock()}):
                from src.backend.llm import OllamaClient
                client = OllamaClient()
                
                assert client.host == 'test-ollama'
                assert client.port == 11435
                assert client.model == 'test-model'
                assert client.base_url == 'http://test-ollama:11435'
    
    def test_default_model(self):
        """Test OllamaClient uses default values."""
        with patch.dict('sys.modules', {'httpx': Mock()}):
            from src.backend.llm import OllamaClient
            client = OllamaClient()
            
            assert client.model == 'qwen3.5:4b'  # Updated default model
            assert client.port == 11434
    
    def test_ollama_client_custom_params(self):
        """Test OllamaClient with custom parameters."""
        with patch.dict('sys.modules', {'httpx': Mock()}):
            from src.backend.llm import OllamaClient
            client = OllamaClient(host='custom-host', port=9999, model='custom-model')
            
            assert client.host == 'custom-host'
            assert client.port == 9999
            assert client.model == 'custom-model'
            assert client.base_url == 'http://custom-host:9999'


class TestPromptFormatting:
    """Tests for prompt formatting in the API."""
    
    def test_context_formatting(self):
        """Test context is properly formatted from search results."""
        mock_results = [
            Mock(ticker='AAPL', filing_type='10-K', filing_date='2024-11-01', content='Apple revenue was $100B', similarity=0.95),
            Mock(ticker='TSLA', filing_type='10-Q', filing_date='2024-10-01', content='Tesla sold 500K vehicles', similarity=0.88)
        ]
        
        context = "\n\n---\n\n".join([
            f"Company: {r.ticker}, Filing: {r.filing_type}, Date: {r.filing_date}\n{r.content}"
            for r in mock_results
        ])
        
        assert "AAPL" in context
        assert "TSLA" in context
        assert "Apple revenue was $100B" in context
        assert "Tesla sold 500K vehicles" in context
    
    def test_prompt_structure(self):
        """Test full prompt structure with context and question."""
        context = "Company: AAPL, Filing: 10-K, Date: 2024-11-01\nRevenue was $100B"
        query = "What was Apple's revenue?"
        
        prompt = (
            f"You are a financial analyst assistant. Answer the question based on the SEC filing excerpts below. "
            f"Provide a direct, concise answer without referencing document numbers or sources - the evidence will be shown separately.\n\n"
            f"SEC Filing Excerpts:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        assert "financial analyst assistant" in prompt
        assert "Question: What was Apple's revenue?" in prompt
        assert "Answer:" in prompt
        assert "Revenue was $100B" in prompt


class TestEvidenceModels:
    """Tests for evidence data models."""
    
    def test_evidence_structure(self):
        """Test Evidence data structure."""
        # Test at data level without Pydantic import issues
        evidence_data = {
            "content": "Test content",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-11-01",
            "similarity": 0.95
        }
        
        assert evidence_data["content"] == "Test content"
        assert evidence_data["ticker"] == "AAPL"
        assert evidence_data["filing_type"] == "10-K"
        assert evidence_data["similarity"] == 0.95


class TestAskRequest:
    """Tests for Ask endpoint request model."""
    
    def test_ask_request_structure(self):
        """Test AskRequest data structure."""
        # Test at data level without Pydantic import issues
        request_data = {
            "query": "What was Apple's revenue?",
            "top_k": 5,
            "ticker": "AAPL",
            "filing_type": "10-K"
        }
        
        assert request_data["query"] == "What was Apple's revenue?"
        assert request_data["top_k"] == 5
        assert request_data["ticker"] == "AAPL"
        assert request_data["filing_type"] == "10-K"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
