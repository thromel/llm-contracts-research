# ML/LLM API Contract Violation Analysis System

## Overview

This system analyzes GitHub issues to identify and categorize API contract violations in both traditional ML and LLM systems. It uses an enhanced taxonomy based on research from "What Kinds of Contracts Do ML APIs Need?" and incorporates continuous learning to discover new contract types.

## Key Features

1. **Comprehensive Taxonomy**
   - Traditional ML API Contracts
     * Single API Method Contracts (data types, boolean expressions)
     * API Method Order Contracts (temporal requirements)
     * Hybrid Contracts (behavioral and temporal combinations)
   - LLM-Specific Contracts
     * Input Contracts (prompt formatting, context management)
     * Processing Contracts (resource management, state consistency)
     * Output Contracts (response format, quality assurance)
     * Error Handling (failure modes, error reporting)
     * Security & Ethical Contracts (access control, content guidelines)

2. **Contract Discovery**
   - Identifies emerging patterns in issues
   - Suggests new contract types
   - Provides rationale and examples for suggestions
   - Helps evolve the taxonomy

3. **Detailed Analysis**
   - Root cause identification
   - Impact assessment
   - Resolution guidance
   - Confidence scoring

## Usage

```python
from src.analysis.core.analyzer import GitHubIssuesAnalyzer

# Initialize analyzer
analyzer = GitHubIssuesAnalyzer("owner/repo")

# Analyze a single issue
result = analyzer.analyze_issue(
    title="Model crashes with wrong input",
    body="Error when passing string instead of tensor",
    comments="Similar issues reported"
)

# Save analysis results
analyzer.save_results([result])
```

## Analysis Output

The system provides analysis results in JSON format:

```json
{
    "has_violation": true,
    "violation_type": "Single_API_Method.Data_Type",
    "severity": "high",
    "description": "Incorrect tensor type passed to model",
    "confidence": "high",
    "root_cause": "Input type mismatch",
    "effects": ["Model crash", "Invalid results"],
    "resolution_status": "Open",
    "resolution_details": "Convert input to float32 tensor",
    "pipeline_stage": "preprocessing",
    "contract_category": "Traditional ML",
    "suggested_new_contracts": [
        {
            "name": "Input_Type_Conversion",
            "description": "Contracts for automatic type conversion handling",
            "rationale": "Recurring issues with type conversion failures",
            "examples": ["string to tensor conversion", "numpy to torch tensor"],
            "parent_category": "Single_API_Method.Data_Type"
        }
    ]
}
```

## Severity Guidelines

The system uses the following criteria for severity assessment:

### High Severity
- System crashes or becomes unusable
- Data loss or corruption
- Security vulnerabilities
- Significant financial impact
- Complete failure of core functionality

### Medium Severity
- Degraded performance
- Partial loss of functionality
- Workaround available
- Limited impact on users
- Non-critical feature affected

### Low Severity
- Minor inconvenience
- Cosmetic issues
- Edge cases only
- Easy workaround available
- Minimal user impact

## Contract Discovery Process

The system actively looks for new contract types by:

1. Analyzing patterns across multiple issues
2. Identifying gaps in existing categories
3. Evaluating emerging challenges in ML/LLM systems
4. Considering industry best practices

When suggesting new contracts, it provides:
- Clear name and description
- Rationale for creation
- Example scenarios
- Placement in taxonomy

## Integration

The system integrates with:
- GitHub API for issue fetching
- OpenAI API for analysis
- Local storage for results

## Configuration

Key settings can be configured in `settings.py`:
- OpenAI model selection
- Analysis parameters
- Output formats
- Storage locations

## Contributing

To contribute:
1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

## References

1. "What Kinds of Contracts Do ML APIs Need?" (2307.14465v1)
2. [AI21 Prompt Engineering Documentation](https://docs.ai21.com/docs/prompt-engineering)
3. GitHub API Documentation
4. OpenAI API Documentation 