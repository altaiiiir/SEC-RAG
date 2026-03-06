# Prompt Experiments Log

This document tracks all system prompt iterations, changes, evaluations, and learnings for the SEC-RAG LLM responses.

---

## Baseline: Hardcoded Basic Prompt

**Date**: 2026-03-05 (Initial Implementation)

**Version**: 0.1.0 (Proof of Concept)

**Location**: Hardcoded in `src/backend/api.py` lines 137-143

**Prompt Text**:
```
You are a financial analyst assistant. Answer the question based on the SEC filing excerpts below. 
Provide a direct, concise answer without referencing document numbers or sources - the evidence will be shown separately.

SEC Filing Excerpts:
{context}

Question: {query}

Answer:
```

**Design Philosophy**:
- Minimal viable prompt to get the system working
- Focus was on optimizing chunking strategies and embedding quality first
- Intentionally kept simple as a baseline for future improvements

**Why Basic**:
The initial implementation prioritized infrastructure over prompt engineering:
1. **Foundation First**: Needed to validate the RAG pipeline (indexing → retrieval → generation) worked end-to-end
2. **Chunking Optimization**: Primary focus was on experimenting with chunking strategies (see `CHUNKING_EXPERIMENTS.md`) to ensure high-quality context retrieval
3. **Embedding Quality**: Wanted to optimize semantic search before tuning LLM output
4. **Avoiding Premature Optimization**: Prompt engineering is powerful but can mask underlying retrieval issues - by keeping it basic, we could identify real problems in the pipeline
5. **Strategic Sequencing**: Intentionally saved prompt optimization for last as it's the "easiest" lever to pull and most forgiving to iterate on

**Characteristics**:
- **Role**: Generic "financial analyst assistant"
- **Output Format**: No specific formatting guidance
- **Tone**: Not specified
- **Structure**: No bullet points, tables, or organizational requirements
- **Audience**: Not defined
- **Length**: No conciseness requirements

**Performance Observations**:
- ✅ Produces coherent answers based on retrieved context
- ✅ Doesn't hallucinate or reference non-existent sources
- ❌ Responses often verbose and paragraph-heavy
- ❌ No consistent formatting or structure
- ❌ Doesn't optimize for business user readability
- ❌ Missing guidance for comparative analysis
- ❌ No emphasis on key metrics or actionable insights

**Evaluation Criteria** (established for future comparisons):

1. **Conciseness**
   - Average response length (words/characters)
   - Percentage of responses under 200 words
   - Information density (key facts per 100 words)

2. **Structure & Readability**
   - Use of bullet points (% of responses)
   - Use of numbered lists for steps/rankings
   - Paragraph length (prefer 2-3 sentences max)
   - Flesch Reading Ease score (target: 60-70)
   - Flesch-Kincaid Grade Level (target: 8-10)

3. **Business Relevance**
   - Answers the "so what?" question
   - Includes specific numbers/metrics
   - Highlights business implications
   - Provides actionable insights

4. **Accuracy & Grounding**
   - Facts align with source documents
   - No hallucinations or made-up data
   - Appropriate use of qualifiers ("approximately", "reported", etc.)
   - Acknowledges data limitations when relevant

5. **User Experience**
   - Time to comprehension (subjective)
   - Ease of scanning for key points
   - Appropriate detail level for audience
   - Professional but accessible tone

**Test Questions for Evaluation**:
1. "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"
2. "How has NVIDIA's revenue and growth outlook changed over the last two years?"
3. "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?"
4. "What is Amazon's strategy for AWS growth?"
5. "Compare the R&D spending of major tech companies as a percentage of revenue."

**Verdict**: ✅ Successful PoC, ready for optimization

---

## Version 1.0: Business-Optimized Prompt

**Date**: 2026-03-05

**Version**: 1.0.0

**Location**: `prompts/system_prompt.json`

**What Changed**:

1. **Role Refinement**
   - Before: Generic "financial analyst assistant"
   - After: "Financial analyst assistant helping business executives"
   - Why: Defines target audience and sets expectations for accessibility

2. **Output Requirements** (NEW)
   - Explicit formatting rules: bullet points, tables, structured lists
   - Conciseness mandate: "brief and concise - get to the point quickly"
   - Readability: "educated but non-expert audience - avoid unnecessary jargon"
   - Number formatting: Clear guidelines (e.g., "$5.2 billion")

3. **Analysis Frameworks** (NEW)
   - Four specific question types with tailored approaches:
     - Risk questions: categorize, prioritize, highlight unique risks
     - Financial performance: lead with key metrics, show trends
     - Comparative questions: parallel structure, easy scanning
     - Regulatory questions: environment + impact + mitigation
   - Ensures consistent, structured responses for common business queries

4. **Evidence Handling** (ENHANCED)
   - Retained: Don't reference document numbers
   - Added: Acknowledge limitations when data is missing
   - Why: Builds trust and prevents overconfident answers

5. **Tone Guidelines** (NEW)
   - "Professional but accessible"
   - "Direct and confident"
   - "Objective and fact-based"
   - Why: Balances authority with approachability for executive audience

6. **Configuration Metadata** (NEW)
   - `max_tokens: 500` - encourages brevity
   - `temperature: 0.3` - reduces creativity, increases consistency
   - Formatting guidelines as structured config
   - Version tracking for experiment reproducibility

**Why These Changes**:

The baseline prompt was a blank canvas. Version 1.0 targets three specific user needs:
1. **Executive Time Constraints**: Busy users need information fast → bullet points, conciseness
2. **Non-Expert Audience**: Not everyone is a CFA → accessible language, clear numbers
3. **Diverse Question Types**: Different questions need different structures → analysis frameworks

**Implementation Details**:
- Externalized to JSON for easy iteration without code changes
- Added metadata for tracking and reproducibility
- Template variables: `{context}` and `{query}` for dynamic insertion
- Config parameters available for future LLM API integration

**Expected Improvements**:
- 📉 40-50% reduction in average response length
- 📊 90%+ responses using bullet points or structured lists
- 📈 Improved readability scores (target Flesch Reading Ease: 60-70)
- ✅ Consistent structure for comparative questions
- 💡 More actionable insights focused on business implications

**Evaluation Plan**:

1. **Quantitative Metrics** (compare to baseline):
   - Run same 5 test questions through both prompts
   - Measure: word count, bullet point usage, reading scores
   - Compare response times and user satisfaction

2. **Qualitative Assessment**:
   - Blind A/B testing with sample users
   - Rate on 5-point scale: clarity, usefulness, readability
   - Collect feedback on specific pain points

3. **Edge Case Testing**:
   - Questions with missing data
   - Multi-company comparisons (3+ companies)
   - Complex regulatory questions
   - Questions requiring numerical calculations

**Next Steps**:
- [ ] Integrate with backend (`llm.py` and `api.py`)
- [ ] Run baseline comparison tests
- [ ] Collect user feedback on v1.0 responses
- [ ] Iterate based on findings

**Status**: 🚧 Implementation in progress

---

## Future Experiments

### Ideas to Test

1. **Chain-of-Thought Prompting**: Add "Let's think step by step" for complex analysis
2. **Few-Shot Examples**: Include 1-2 example Q&A pairs for format consistency
3. **Dynamic Prompt Selection**: Use different prompts based on question type detection
4. **Iterative Refinement**: Multi-step prompting (analyze → summarize → format)
5. **Confidence Scoring**: Ask LLM to rate confidence and highlight uncertain areas
6. **Executive Summary Mode**: Ultra-brief mode for C-suite (50-word max)
7. **Comparative Tables**: Structured output format for side-by-side comparisons

### Experiment Template

```markdown
## Version X.X: [Experiment Name]

**Date**: [YYYY-MM-DD]
**Version**: [X.X.X]
**Location**: `prompts/system_prompt_vX.json`

**What Changed**:
- [Specific changes with before/after]

**Why These Changes**:
- [Hypothesis and reasoning]

**Implementation Details**:
- [Technical notes]

**Metrics** (vs. previous version):
- Response length: [before → after]
- Bullet point usage: [before → after]
- Reading ease score: [before → after]
- User satisfaction: [before → after]

**Qualitative Feedback**:
- What users liked: [...]
- What users disliked: [...]
- Surprising findings: [...]

**Verdict**: ✅ Keep / ❌ Revert / 🔄 Iterate

**Next Steps**:
- [What to try next]
```

---

## Evaluation Guidelines

### How to Evaluate Prompt Quality

1. **Run Test Suite**: Use the 5 standard test questions + 5 edge cases
2. **Measure Objectively**: Track word count, structure, readability scores
3. **User Feedback**: Minimum 3 users rate responses (blind testing when possible)
4. **Compare to Baseline**: Always benchmark against v0.1.0 and previous version
5. **Document Edge Cases**: Note any failure modes or unexpected behaviors
6. **Consider Trade-offs**: Sometimes verbosity aids clarity - balance is key

### Red Flags

- Hallucinations or made-up facts
- Inconsistent formatting across similar questions
- Jargon creep (returning to technical language)
- Loss of nuance or important context
- Over-simplification that misleads

### Success Indicators

- Users can scan and understand in < 30 seconds
- Key numbers and facts are immediately visible
- Comparative questions have parallel structure
- Professional tone maintained while being accessible
- Answers are grounded in provided evidence
