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

## Future Experiments

### Experiment Template

```markdown
### Experiment: [Name]
**Date**: [YYYY-MM-DD]
**Approach**: [One-line description]

**Implementation**:
- [Bullet point changes]

**Metrics**:
- Indexing time: [before → after]
- Chunk count: [before → after]
- Avg chunk size: [tokens]
- Chunk size variance: [std dev or range]

**Qualitative**:
- Search quality: [Better/Same/Worse + why]
- LLM answer quality: [Better/Same/Worse + why]
- Specific improvements: [What got better]
- Regressions: [What got worse]

**Pros**:
- [What worked well]

**Cons**:
- [What didn't work]

**Verdict**: ✅ Keep / ❌ Revert / 🔄 Iterate

**Next Steps**:
- [What to try next based on learnings]
```

---

## Planned Experiments

1. **Sentence-Aware Boundaries**: Snap chunk boundaries to sentence endings
2. **Table Detection & Preservation**: Detect tables and chunk by row groups with headers
3. **List-Aware Chunking**: Keep list items together, preserve parent context
4. **Semantic Overlap**: Use full sentences for overlap instead of fixed tokens
5. **Section-Based Chunking**: Detect Item sections in 10-K/10-Q filings
6. **Adaptive Strategy**: Different chunking for tables vs narrative vs lists
7. **Metadata Enrichment**: Add section names, content types, table context
