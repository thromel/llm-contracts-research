# Multi-Stage Quality Filtering System

## Overview
Implements a rigorous 3-stage filtering system to ensure only high-quality, code-containing posts with peer-validated solutions are collected.

## Stack Overflow Filtering Stages

### Stage 1: Executable Code Detection
**Criterion:** Contains executable code in the question body  
**Expected Posts:** ~51,674 (based on research data)

**Implementation:**
- Detects 25+ code indicators including:
  - Markdown code blocks (```)
  - Programming keywords (import, def, function, etc.)
  - LLM API calls (openai., client., etc.)
  - HTTP methods (POST, GET, curl)
  - Async patterns (await, async)
- Requires ≥2 indicators for high confidence
- Applied during question processing

### Stage 2: High-Quality Discussion
**Criterion:** Question score > 5  
**Expected Posts:** ~5,570 (significant reduction from Stage 1)

**Implementation:**
- Applied at API level using `min=6` parameter
- Ensures community has validated the question quality
- Filters out low-engagement or unclear questions
- Reduces API calls by filtering server-side

### Stage 3: Peer-Validated Solutions
**Criterion:** Accepted answer with score > 5  
**Expected Posts:** ~3,661 (final high-quality dataset)

**Implementation:**
- Requires both `is_answered=true` AND `accepted_answer_id` present
- Fetches accepted answer via separate API call
- Validates answer score > 5 before including question
- Ensures solution has community endorsement

## GitHub Filtering Stages

### Stage 1: Issue Quality
- **Closed state:** Only resolved issues
- **Code detection:** Same algorithm as Stack Overflow
- **Scope:** Focus on usage issues, not feature requests

### Stage 2: High Engagement  
- **Minimum 5 comments:** Ensures substantial discussion
- **Active resolution:** Evidence of problem-solving process

### Stage 3: Relevance Filtering
- **Exclude enhancement/bug labels:** Focus on usage problems
- **Include error discussions:** Target implementation issues

## Expected Data Volume

| Platform | Stage 1 | Stage 2 | Stage 3 | Final |
|----------|---------|---------|---------|-------|
| Stack Overflow | ~51,674 | ~5,570 | ~3,661 | ~3,661 |
| GitHub | ~15,000 | ~3,000 | ~2,000 | ~2,000 |
| **Total** | **~66,674** | **~8,570** | **~5,661** | **~5,661** |

## API Efficiency

**Optimizations:**
- Stage 2 filtering at API level reduces bandwidth
- Batch answer score validation
- Early termination on failed criteria
- Focused tag selection (4 high-signal tags)

**Rate Limiting:**
- 0.1s delay between API calls
- Graceful quota handling
- Automatic backoff on rate limits

## Quality Assurance

**Code Detection Accuracy:**
- 25+ programming indicators
- LLM-specific patterns (openai., anthropic.)
- Error handling patterns (traceback, exception)
- Multi-language support (Python, JavaScript, SQL, etc.)

**Peer Validation:**
- Community-scored questions (>5 votes)
- Accepted answers with peer endorsement (>5 votes)
- High-engagement discussions (≥5 comments for GitHub)

## Configuration

```yaml
stackoverflow:
  max_questions_per_tag: 10000  # Reduced due to strict filtering
  tags: [openai-api, langchain, gpt-4, chatgpt-api]
  
github:
  enabled: true  # Now enabled with quality filters
  max_issues_per_repo: 1000
```

## Expected Outcomes

**Signal-to-Noise Ratio:** ~95% (vs. ~5-10% with minimal filtering)  
**Data Quality:** Peer-validated, code-containing, solution-endorsed  
**Research Value:** High-confidence LLM usage patterns and contract violations  
**Processing Efficiency:** Reduced downstream filtering needs 