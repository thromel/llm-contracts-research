# LLM API Contract Classification System

## Overview

This document describes the enhanced contract classification system for LLM API research. The system is designed to both identify known contract patterns and discover novel ones, based on empirical research of 600+ real-world violations.

## Key Components

### 1. Contract Taxonomy (`contract_taxonomy.py`)

The taxonomy provides a structured classification of known LLM API contracts:

```python
from pipeline.llm_screening.contract_taxonomy import LLMContractTaxonomy, ContractCategory

# Access the taxonomy
taxonomy = LLMContractTaxonomy()

# Get contracts by category
behavioral_contracts = taxonomy.get_contracts_by_category(ContractCategory.SINGLE_API_METHOD)
llm_specific = taxonomy.get_contracts_by_category(ContractCategory.LLM_SPECIFIC)

# Identify violations from error text
violations = taxonomy.identify_violations("Error: maximum context length exceeded")
```

### 2. Discovery-Oriented Prompts (`discovery_prompts.py`)

These prompts prioritize finding new contract types:

```python
from pipeline.llm_screening.prompts.discovery_prompts import ContractDiscoveryPrompts

prompts = ContractDiscoveryPrompts()

# Open-ended discovery
discovery_prompt = prompts.get_open_discovery_prompt()

# Deep analysis of findings
deep_analysis_prompt = prompts.get_deep_analysis_prompt()
```

### 3. Contract Analysis Engine (`contract_analysis.py`)

Analyzes posts for both known and novel contract violations:

```python
from pipeline.llm_screening.contract_analysis import ContractAnalyzer

analyzer = ContractAnalyzer()
result = analyzer.analyze_post(filtered_post, llm_screening_result)

# Check for novel discoveries
if result.novel_violations > 0:
    print(f"Found {result.novel_violations} new contract patterns!")
```

## Contract Categories

### Traditional (from ML APIs)
- **Value Range**: Parameter limits (temperature, top_p)
- **Type Constraints**: Data type requirements
- **Method Order**: API call sequences

### LLM-Specific
- **Prompt Format**: Message structure requirements
- **Output Format**: JSON mode, function calling
- **Content Policy**: Safety and ethical constraints
- **Context Management**: Conversation state, memory limits
- **Rate Limiting**: Token and request limits
- **Streaming**: Real-time response handling

### Novel Categories (Discovered)
- **Cost Contracts**: Budget and pricing constraints
- **Latency Contracts**: Response time expectations
- **Consistency Contracts**: Deterministic behavior requirements
- **Tool Integration**: Agent and function calling patterns

## Examples

### Example 1: Rate Limit Violation
```json
{
  "title": "OpenAI API returning 429 error",
  "content": "Getting 'Rate limit reached for gpt-4' after only 3 requests",
  "classification": {
    "contract_type": "RATE_LIMIT",
    "category": "LLM_SPECIFIC",
    "severity": "high",
    "evidence": ["429 error", "Rate limit reached"],
    "api_provider": "openai"
  }
}
```

### Example 2: Novel Format Contract
```json
{
  "title": "LLM ignoring JSON output instructions",
  "content": "Despite clear instructions, model responds in plain text instead of JSON",
  "classification": {
    "is_novel": true,
    "novel_name": "Instruction_Adherence_Contract",
    "description": "Model must follow output format instructions in prompt",
    "category_suggestion": "NovelBehavior",
    "evidence": ["ignoring instructions", "responds in plain text"]
  }
}
```

### Example 3: Context Length with Cascade
```json
{
  "title": "Conversation history causing token overflow",
  "content": "After 10 messages, getting context length error, then parsing fails",
  "classification": {
    "violations": [
      {
        "contract_type": "CONTEXT_LENGTH",
        "severity": "high"
      },
      {
        "contract_type": "OUTPUT_FORMAT",
        "severity": "medium"
      }
    ],
    "pattern": "cascade",
    "primary_violation": "CONTEXT_LENGTH"
  }
}
```

## Usage in Pipeline

### 1. Screening Enhancement

Update screening to use classification:

```python
# In screening orchestrator
from pipeline.llm_screening.contract_analysis import ContractAnalyzer

class EnhancedScreeningOrchestrator:
    def __init__(self):
        self.analyzer = ContractAnalyzer()
    
    async def screen_with_classification(self, post: FilteredPost):
        # Regular screening
        llm_result = await self.bulk_screener.screen(post)
        
        # Contract classification
        analysis = self.analyzer.analyze_post(post, llm_result)
        
        # Enhance LLM result with classification
        llm_result.contract_types_identified = [
            v.contract_type for v in analysis.violations 
            if v.contract_type
        ]
        llm_result.violation_severity = analysis.primary_violation.severity.value
        
        return llm_result, analysis
```

### 2. Novel Discovery Pipeline

```python
# Discover novel contracts across dataset
novel_contracts = []

for post in filtered_posts:
    analysis = analyzer.analyze_post(post)
    if analysis.novel_violations > 0:
        novel_contracts.extend([
            v for v in analysis.violations if v.is_novel
        ])

# Synthesize findings
print(f"Discovered {len(novel_contracts)} potential new contract types")
```

### 3. Research Value Filtering

```python
# Filter posts by research value
high_value_posts = []

for post, analysis in zip(posts, analyses):
    if analysis.research_value_score > 0.7:
        high_value_posts.append({
            'post': post,
            'analysis': analysis,
            'reason': 'high_research_value'
        })
    elif analysis.novel_violations > 0:
        high_value_posts.append({
            'post': post,
            'analysis': analysis,
            'reason': 'novel_discovery'
        })
```

## Best Practices

### 1. Open-Minded Discovery
- Don't force violations into existing categories
- Look for patterns that span multiple posts
- Document edge cases and unusual behaviors

### 2. Evidence Quality
- Prefer specific error messages over vague descriptions
- Include code snippets when available
- Note reproduction steps

### 3. Novel Contract Validation
- Cross-reference with multiple examples
- Test reproducibility
- Consider broader implications

### 4. Continuous Taxonomy Evolution
- Regularly review novel discoveries
- Update taxonomy with validated new patterns
- Version taxonomy changes

## Configuration

### Environment Variables
```bash
# Enable novel discovery mode
CONTRACT_DISCOVERY_MODE=true

# Set confidence thresholds
MIN_VIOLATION_CONFIDENCE=0.3
NOVEL_PATTERN_THRESHOLD=0.5

# Enable detailed logging
CONTRACT_ANALYSIS_DEBUG=true
```

### Analysis Settings
```python
# In config.yaml
contract_analysis:
  enable_novel_discovery: true
  min_confidence: 0.3
  max_novel_per_post: 5
  evidence_context_lines: 3
  
  # Provider-specific settings
  providers:
    openai:
      known_rate_limits:
        gpt-4: 500
        gpt-3.5-turbo: 3500
    anthropic:
      known_rate_limits:
        claude-3-opus: 1000
```

## Metrics and Monitoring

### Key Metrics
- **Novel Discovery Rate**: Novel contracts / Total contracts
- **Classification Confidence**: Average confidence score
- **Category Distribution**: Breakdown by contract category
- **Severity Distribution**: Critical/High/Medium/Low breakdown

### Monitoring Dashboard
```python
# Generate analysis report
from pipeline.llm_screening.contract_analysis import analyze_post_batch

results = analyze_post_batch(filtered_posts)

# Summary statistics
total_violations = sum(r.total_violations for r in results)
novel_violations = sum(r.novel_violations for r in results)
discovery_rate = novel_violations / total_violations if total_violations > 0 else 0

print(f"Analyzed {len(results)} posts")
print(f"Total violations: {total_violations}")
print(f"Novel violations: {novel_violations} ({discovery_rate:.1%})")
```

## Future Enhancements

1. **Machine Learning Integration**
   - Train classifiers on discovered patterns
   - Predict contract types from error text
   - Cluster similar violations automatically

2. **Interactive Discovery**
   - Web interface for reviewing novel contracts
   - Collaborative annotation tools
   - Community validation system

3. **Automated Documentation**
   - Generate contract documentation from discoveries
   - Create test cases for each contract type
   - Build enforcement libraries

4. **Cross-Platform Analysis**
   - Compare contracts across API providers
   - Identify platform-specific vs universal patterns
   - Track contract evolution over time