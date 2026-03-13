# Chunking Experiments Log

This document tracks all chunking strategy experiments, their results, and learnings.

## Baseline: Fixed-Size Token Chunking

**Date**: 2026-03-05

**Approach**: Fixed 512-token chunks with 50-token overlap using tiktoken encoding.

**Implementation**:
- Simple sliding window over token sequence
- No awareness of sentence boundaries, paragraphs, or document structure
- Uniform chunk size regardless of content type

**Metrics**:
- Chunk size: 512 tokens (fixed)
- Overlap: 50 tokens (fixed)
- Avg chunks per document: ~126 (based on AAPL 10-K example)
- Indexing time: Fast, minimal preprocessing

**Qualitative**:
- Search quality: Baseline (no comparison yet)
- LLM answer quality: Baseline (no comparison yet)
- Issues observed:
  - Chunks split mid-sentence
  - Tables broken across chunks without preserving headers
  - No semantic boundaries respected
  - Financial data scattered across unrelated chunks

**Pros**:
- Simple, fast, predictable
- Consistent chunk sizes for embedding model
- Easy to implement and debug

**Cons**:
- Breaks semantic units (sentences, paragraphs, tables)
- Poor context preservation for tables and lists
- No document structure awareness
- Hard to trace answers back to specific sections

**Verdict**: ✅ Good starting point, but needs improvement for structured content

---

## Bug Fixes: List Detection & Token Counting

**Date**: 2026-03-05

**Issues Fixed**:
1. List detection regex was overly permissive and incorrectly structured
2. Token counts summed individual parts instead of encoding final joined text

**Changes**:
- Fixed regex pattern: `r'^\s*(?:[-•●○]|\d+[.)]|\([a-zA-Z0-9]+\)|[a-zA-Z][.)])\s'`
- Token counts now encode final chunk text (4 locations in `_chunk_list` and `_chunk_financial_statement`)

**Impact**:
- More accurate list item detection (numbered, bulleted, lettered lists)
- Correct token counts for downstream logic and cost tracking

**Verdict**: ✅ Critical fixes applied

---

## Adaptive Chunking: Bugs & Failure Analysis

**Date**: 2026-03-05

**Approach**: Multi-strategy chunking (`_chunk_table`, `_chunk_list`, `_chunk_financial_statement`, `_chunk_narrative`) with content type detection.

**Bugs Found**:
1. **Empty chunks**: `text.split('\n\n')` produced empty strings that became blank chunks in the DB
2. **XBRL pollution**: The ~200KB XBRL data blob on line 12 of each filing was classified as a "financial_statement" because it contains keywords like "net income", "total assets" — it's machine-readable garbage, not useful content
3. **Giant chunks**: Financial statement chunks were 60K+ tokens (242KB) because the XBRL blob has no `\n\n` paragraph breaks
4. **Broken section names**: `extract_section_name(content, idx * 100)` used chunk index × 100 as a character position — completely wrong
5. **Over-engineered detection**: Table/list/financial detection logic (335 lines) rarely triggered correctly on actual SEC filings

**Results** (5 docs):
- Before fixes: 15 chunks, 1 empty, 1 is 242KB
- After empty-chunk fix: 10 chunks, none empty, but avg ~101KB (still unusable)
- After fallback splitting: 449 chunks, avg ~1900 chars — better but the approach is fundamentally wrong

**Verdict**: ❌ Multi-strategy detection doesn't fit this corpus. Need specialized approach.

---

## SEC-Specialized Chunking (Current)

**Date**: 2026-03-05

**Approach**: Strip XBRL + split by Item headers + single token-based chunker with sentence snapping.

**What Changed**:
- Replaced `content_detector.py` (335 lines, generic) → `SECFilingParser` (~60 lines, SEC-specific)
  - Strips metadata header above `======` separator
  - Strips XBRL blob by finding "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" marker
  - Splits clean text at `Item N.` headers (regex: `^Item\s+\d+[A-Z]?\.`)
- Replaced `adaptive_chunker.py` (355 lines, 4 strategies) → `SECChunker` (~60 lines, 1 strategy)
  - Single token-based sliding window with sentence-boundary snapping
  - Each chunk tagged with `section_name`, `section_chunk_index`, `total_section_chunks`
- Simplified `chunking_config.py` (76 → 14 lines)
  - Removed: `table_row_chunk_size`, `table_preserve_header`, `min_table_rows`, `min_list_items`, `enable_table_detection`, `enable_list_detection`, `enable_nlp_detection`, `overlap_strategy`, `enable_adaptive_chunking`, `enable_semantic_overlap`
  - Kept: `chunk_size`, `chunk_overlap`, `min_chunk_size`, `enable_sentence_boundaries`
- Updated `indexer.py`: removed legacy `_chunk_text`, fixed section name extraction (now comes from parser, not broken position estimate)

**Expected Results** (5 docs):
- ~100-200 chunks per doc (properly sized ~512 tokens)
- Every chunk has a meaningful `section_name` (e.g., "Item 1A. Risk Factors")
- Zero empty chunks, zero XBRL noise
- Section metadata enables "find more from this section" queries

**Code Reduction**: ~770 lines → ~135 lines across the 3 files

**Verdict**: ✅ Specialized for the corpus, simpler, and more effective
