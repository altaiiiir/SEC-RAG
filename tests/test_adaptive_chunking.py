"""Tests for SEC filing parser and adaptive chunker (narrative + table/list)."""
import pytest
from src.backend.content_detector import SECFilingParser, FilingSection, ContentBlock
from src.backend.adaptive_chunker import SECChunker


SAMPLE_FILING = """Company: Apple Inc
Ticker: AAPL
Filing Type: 10-K (Annual Report)
Filing Date: 2022-10-28
============================================================

aapl-20220924false2022FY0000320193us-gaap:CommonStockMember2021-09-262022-09-24UNITED STATESSECURITIES AND EXCHANGE COMMISSIONWashington, D.C. 20549FORM 10-K
Table of Contents
Item 1. | Business | 1
Item 1A. | Risk Factors | 5
Item 1.    BusinessCompany BackgroundThe Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.
ProductsiPhoneiPhone is the Company's line of smartphones based on its iOS operating system.
Apple Inc. | 2022 Form 10-K | 1
Item 1A.    Risk FactorsThe Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors.
Macroeconomic and Industry RisksThe Company's operations depend significantly on global and regional economic conditions.
Apple Inc. | 2022 Form 10-K | 5
Item 2.    PropertiesThe Company's headquarters are located in Cupertino, California. As of September 24, 2022, the Company owned or leased facilities.
Item 7.    Management's Discussion and AnalysisFiscal 2022 HighlightsTotal net sales increased 8% or $28.5 billion during 2022 compared to 2021.
"""


class TestSECFilingParser:
    def test_strips_metadata_header(self):
        parser = SECFilingParser()
        result = parser._strip_metadata_header(SAMPLE_FILING)
        assert "Company: Apple Inc" not in result
        assert "Ticker: AAPL" not in result

    def test_strips_xbrl(self):
        parser = SECFilingParser()
        text = parser._strip_metadata_header(SAMPLE_FILING)
        result = parser._strip_xbrl(text)
        assert "aapl-20220924false" not in result
        assert "UNITED STATES" in result

    def test_splits_into_sections(self):
        parser = SECFilingParser()
        sections = parser.parse(SAMPLE_FILING)
        names = [s.name for s in sections]

        assert any("Item 1." in n for n in names), f"Should find Item 1, got: {names}"
        assert any("Item 1A." in n for n in names), f"Should find Item 1A, got: {names}"
        assert any("Item 2." in n for n in names), f"Should find Item 2, got: {names}"

    def test_no_empty_sections(self):
        parser = SECFilingParser()
        sections = parser.parse(SAMPLE_FILING)
        for s in sections:
            assert s.text.strip(), f"Section '{s.name}' has empty text"

    def test_no_xbrl_in_sections(self):
        parser = SECFilingParser()
        sections = parser.parse(SAMPLE_FILING)
        for s in sections:
            assert "us-gaap:" not in s.text, f"XBRL data leaked into section '{s.name}'"


class TestBlockDetection:
    def test_table_block_detected(self):
        table_text = "Col A    Col B    Col C\n1        2        3\n4        5        6"
        blocks = SECFilingParser.split_into_blocks(table_text)
        assert len(blocks) == 1
        assert blocks[0].block_type == "table"
        assert blocks[0].rows is not None
        assert len(blocks[0].rows) == 3

    def test_list_block_detected(self):
        list_text = "- First item.\n- Second item.\n- Third item."
        blocks = SECFilingParser.split_into_blocks(list_text)
        assert len(blocks) == 1
        assert blocks[0].block_type == "list"

    def test_narrative_block_detected(self):
        narrative = "This is a paragraph. It has no columns or bullets."
        blocks = SECFilingParser.split_into_blocks(narrative)
        assert len(blocks) == 1
        assert blocks[0].block_type == "narrative"


class TestSECChunker:
    def test_small_section_stays_single_chunk(self):
        chunker = SECChunker()
        section = FilingSection("Item 2.", "Short section text.", 0, 20)
        chunks = chunker.chunk_document([section])
        assert len(chunks) == 1
        assert chunks[0]['section_name'] == "Item 2."

    def test_large_section_splits(self):
        chunker = SECChunker()
        long_text = "This is a sentence about financial risk. " * 200
        section = FilingSection("Item 1A.", long_text, 0, len(long_text))
        chunks = chunker.chunk_document([section])
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        chunker = SECChunker()
        section = FilingSection("Item 7.", "Revenue grew 8%.", 0, 17)
        chunks = chunker.chunk_document([section])
        chunk = chunks[0]
        assert chunk['section_name'] == "Item 7."
        assert chunk['section_chunk_index'] == 0
        assert chunk['total_section_chunks'] == 1
        assert chunk['chunk_type'] == 'narrative'
        assert chunk['token_count'] > 0

    def test_no_empty_chunks(self):
        chunker = SECChunker()
        text = "Sentence one. " * 300
        section = FilingSection("Test", text, 0, len(text))
        chunks = chunker.chunk_document([section])
        for c in chunks:
            assert c['text'].strip(), "Chunk text should not be empty"
            assert c['token_count'] > 0

    def test_table_chunk_has_type_table(self):
        chunker = SECChunker()
        table_text = "Col A    Col B    Col C\n1        2        3\n4        5        6"
        section = FilingSection("Item 7.", table_text, 0, len(table_text))
        chunks = chunker.chunk_document([section])
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "table"
        assert "Col A" in chunks[0]["text"]


class TestIntegration:
    def test_full_pipeline(self):
        parser = SECFilingParser()
        chunker = SECChunker()

        sections = parser.parse(SAMPLE_FILING)
        assert len(sections) > 0

        chunks = chunker.chunk_document(sections)
        assert len(chunks) > 0

        for chunk in chunks:
            assert 'text' in chunk
            assert 'section_name' in chunk
            assert 'token_count' in chunk
            assert chunk['text'].strip()

    def test_section_names_propagate(self):
        parser = SECFilingParser()
        chunker = SECChunker()
        chunks = chunker.chunk_document(parser.parse(SAMPLE_FILING))
        section_names = set(c['section_name'] for c in chunks)
        assert len(section_names) > 1, "Should have chunks from multiple sections"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
