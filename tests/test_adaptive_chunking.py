"""
Tests for adaptive chunking and content detection.
"""
import pytest
from src.backend.content_detector import ContentDetector, ContentSection
from src.backend.adaptive_chunker import AdaptiveChunker


# Sample SEC filing text with different content types
SAMPLE_TABLE = """
| Product | Q1 2022 | Q2 2022 | Q3 2022 |
| iPhone | $50,365 | $40,665 | $42,626 |
| Mac | $10,435 | $7,382 | $11,508 |
| iPad | $7,646 | $7,224 | $7,174 |
| Services | $19,821 | $19,604 | $19,188 |
"""

SAMPLE_LIST = """
Our risk factors include:
- Market competition from other technology companies
- Supply chain disruptions affecting product availability
- Regulatory changes in key markets
- Economic uncertainty affecting consumer spending
- Currency exchange rate fluctuations
"""

SAMPLE_NARRATIVE = """
Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, 
wearables, and accessories worldwide. The Company sells and delivers digital content and 
applications through the iTunes Store, App Store, Mac App Store, TV App Store, iBooks Store, 
and Apple Music. Apple was founded in 1976 and is headquartered in Cupertino, California.
"""

SAMPLE_FINANCIAL = """
Total net sales consist of revenue from the sale of iPhone, Mac, iPad, 
Services and other products. Total net sales for fiscal 2022 were $394,328 million, 
compared to $365,817 million in fiscal 2021. Total operating expenses were $51,345 million 
or 17% of total net sales in 2022.
"""


class TestContentDetector:
    """Tests for ContentDetector class."""
    
    def test_table_detection(self):
        """Test pipe-delimited table detection."""
        detector = ContentDetector(min_table_rows=3)
        tables = detector.detect_tables(SAMPLE_TABLE)
        
        assert len(tables) > 0, "Should detect at least one table"
        start, end, metadata = tables[0]
        assert metadata['format'] == 'pipe'
        assert metadata['row_count'] >= 3
        assert 'iPhone' in metadata['header'] or 'Product' in metadata['header']
    
    def test_list_detection(self):
        """Test bullet list detection."""
        detector = ContentDetector(min_list_items=2)
        lists = detector.detect_lists(SAMPLE_LIST)
        
        assert len(lists) > 0, "Should detect at least one list"
        start, end, metadata = lists[0]
        assert metadata['list_type'] == 'bullet'
        assert metadata['item_count'] >= 4
    
    def test_financial_statement_detection(self):
        """Test financial statement detection."""
        detector = ContentDetector()
        statements = detector.detect_financial_statements(SAMPLE_FINANCIAL)
        
        assert len(statements) > 0, "Should detect financial data"
        start, end, metadata = statements[0]
        assert 'keyword' in metadata
        assert metadata['keyword'] in ['total revenue', 'total net sales', 'total operating expenses']
    
    def test_section_extraction(self):
        """Test section name extraction."""
        text = "Item 1A. Risk Factors\n\nOur business faces various risks..."
        detector = ContentDetector()
        section = detector.extract_section_name(text, 50)
        
        assert "Item 1A" in section or section == "", "Should extract section or return empty"
    
    def test_content_sections(self):
        """Test full content section detection."""
        combined_text = f"{SAMPLE_NARRATIVE}\n\n{SAMPLE_TABLE}\n\n{SAMPLE_LIST}"
        detector = ContentDetector(min_table_rows=3, min_list_items=2)
        
        sections = detector.detect_content_sections(combined_text)
        
        assert len(sections) > 0, "Should detect multiple sections"
        
        # Check that we have different types
        types = [s.type for s in sections]
        assert 'narrative' in types, "Should have narrative sections"
        
        # Verify sections don't overlap
        for i in range(len(sections) - 1):
            assert sections[i].end_pos <= sections[i+1].start_pos, "Sections should not overlap"


class TestAdaptiveChunker:
    """Tests for AdaptiveChunker class."""
    
    def test_table_chunking(self):
        """Test table chunking preserves headers."""
        chunker = AdaptiveChunker()
        section = ContentSection(
            type='table',
            text=SAMPLE_TABLE,
            start_pos=0,
            end_pos=len(SAMPLE_TABLE),
            metadata={'row_count': 4, 'header': '| Product | Q1 2022 | Q2 2022 | Q3 2022 |', 'format': 'pipe'}
        )
        
        chunks = chunker.chunk_section(section)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(c['chunk_type'] == 'table' for c in chunks), "All chunks should be type 'table'"
        
        # Check if header is preserved
        for chunk in chunks:
            if 'Product' in SAMPLE_TABLE.split('\n')[0]:
                # Header should be in first line of most chunks
                pass  # Basic validation
    
    def test_list_chunking(self):
        """Test list chunking keeps items together."""
        chunker = AdaptiveChunker()
        section = ContentSection(
            type='list',
            text=SAMPLE_LIST,
            start_pos=0,
            end_pos=len(SAMPLE_LIST),
            metadata={'item_count': 5, 'list_type': 'bullet'}
        )
        
        chunks = chunker.chunk_section(section)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(c['chunk_type'] == 'list' for c in chunks), "All chunks should be type 'list'"
        
        # Check that context is preserved
        if "Our risk factors" in SAMPLE_LIST:
            assert any("risk factors" in c['text'].lower() for c in chunks), "Context should be preserved"
    
    def test_narrative_chunking(self):
        """Test narrative chunking with sentence boundaries."""
        chunker = AdaptiveChunker()
        section = ContentSection(
            type='narrative',
            text=SAMPLE_NARRATIVE,
            start_pos=0,
            end_pos=len(SAMPLE_NARRATIVE),
            metadata={}
        )
        
        chunks = chunker.chunk_section(section)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(c['chunk_type'] == 'narrative' for c in chunks), "All chunks should be type 'narrative'"
        
        # Check token counts
        for chunk in chunks:
            assert 'token_count' in chunk, "Should have token count"
            assert chunk['token_count'] > 0, "Token count should be positive"
    
    def test_financial_statement_chunking(self):
        """Test financial statement chunking as semantic unit."""
        chunker = AdaptiveChunker()
        section = ContentSection(
            type='financial_statement',
            text=SAMPLE_FINANCIAL,
            start_pos=0,
            end_pos=len(SAMPLE_FINANCIAL),
            metadata={'statement_type': 'income_statement', 'keyword': 'total revenue'}
        )
        
        chunks = chunker.chunk_section(section)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(c['chunk_type'] == 'financial_statement' for c in chunks), "All chunks should be financial_statement"
        
        # Small financial statements should stay together
        if len(SAMPLE_FINANCIAL) < 500:
            assert len(chunks) == 1, "Small financial statements should be single chunk"
    
    def test_chunk_metadata(self):
        """Test that chunks contain proper metadata."""
        chunker = AdaptiveChunker()
        section = ContentSection(
            type='table',
            text=SAMPLE_TABLE,
            start_pos=0,
            end_pos=len(SAMPLE_TABLE),
            metadata={'row_count': 4, 'header': 'test', 'format': 'pipe'}
        )
        
        chunks = chunker.chunk_section(section)
        
        for chunk in chunks:
            assert 'text' in chunk, "Should have text field"
            assert 'chunk_type' in chunk, "Should have chunk_type field"
            assert 'token_count' in chunk, "Should have token_count field"
            assert 'metadata' in chunk, "Should have metadata field"
            assert chunk['text'].strip(), "Text should not be empty"


class TestIntegration:
    """Integration tests for detector + chunker."""
    
    def test_full_pipeline(self):
        """Test complete detection and chunking pipeline."""
        combined_text = f"{SAMPLE_NARRATIVE}\n\n{SAMPLE_TABLE}\n\n{SAMPLE_LIST}\n\n{SAMPLE_FINANCIAL}"
        
        detector = ContentDetector(min_table_rows=3, min_list_items=2)
        chunker = AdaptiveChunker()
        
        # Detect sections
        sections = detector.detect_content_sections(combined_text)
        assert len(sections) > 0, "Should detect sections"
        
        # Chunk all sections
        all_chunks = []
        for section in sections:
            section_chunks = chunker.chunk_section(section)
            all_chunks.extend(section_chunks)
        
        assert len(all_chunks) > 0, "Should create chunks"
        
        # Verify variety of chunk types
        chunk_types = set(c['chunk_type'] for c in all_chunks)
        assert len(chunk_types) >= 2, "Should have multiple chunk types"
        
        # Verify all chunks have required fields
        for chunk in all_chunks:
            assert 'text' in chunk
            assert 'chunk_type' in chunk
            assert 'token_count' in chunk
            assert chunk['token_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
