# ML/LLM API Contract Violation Analysis System

## Overview

This system analyzes GitHub issues to identify and categorize API contract violations in both traditional ML and LLM systems. It uses an enhanced taxonomy based on research from "What Kinds of Contracts Do ML APIs Need?" (2307.14465v1) and incorporates continuous learning to discover new contract types.

## Research Insights

The system's design is heavily influenced by key findings from the research paper:

1. **Contract Type Distribution**:
   - 28.4% of violations involve unacceptable input values
   - 56.93% of violations lead to system crashes
   - Early pipeline stage violations are most critical

2. **Contract Categories**:
   - Most frequent: Single API Method contracts (argument constraints)
   - Second most common: API Method Order contracts (temporal requirements)
   - Unique to ML: Hybrid contracts combining behavioral and temporal aspects

3. **Pipeline Stage Impact**:
   - Violations in early stages (preprocessing, model construction) are most critical
   - Effects often propagate through multiple pipeline stages
   - Stage-specific contract requirements are essential

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

2. **Pipeline Stage Analysis**
   - Stage identification
   - Cross-stage impact assessment
   - Error propagation tracking
   - Stage-specific requirements

3. **Contract Discovery**
   - Identifies emerging patterns
   - Suggests new contract types
   - Provides rationale and examples
   - Maps to pipeline stages

4. **Detailed Analysis**
   - Root cause identification
   - Impact assessment
   - Resolution guidance
   - Error propagation tracking

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
    "violation_type": "Single_API_Method.Data_Type.ML_Type.Tensor_Type",
    "severity": "high",
    "description": "Incorrect tensor type passed to model",
    "confidence": "high",
    "root_cause": "Input type mismatch",
    "effects": ["Model crash", "Invalid results"],
    "resolution_status": "Open",
    "resolution_details": "Convert input to float32 tensor",
    "pipeline_stage": "preprocessing",
    "contract_category": "Traditional ML",
    "error_propagation": {
        "origin_stage": "preprocessing",
        "affected_stages": ["model_training", "inference"],
        "propagation_path": "Invalid tensor type affects model weights computation"
    },
    "suggested_new_contracts": [
        {
            "name": "Tensor_Conversion_Contract",
            "description": "Automatic tensor type conversion handling",
            "rationale": "Common issue with tensor type mismatches",
            "examples": ["string to tensor", "numpy to torch tensor"],
            "parent_category": "Single_API_Method.Data_Type.ML_Type",
            "pipeline_stage": "preprocessing"
        }
    ]
}
```

## Severity Guidelines

The system uses the following criteria for severity assessment:

### High Severity
- System crashes (affects ~56.93% of violations)
- Data loss or corruption
- Security vulnerabilities
- Early pipeline stage violations
- Cross-stage impact

### Medium Severity
- Performance degradation
- Partial functionality loss
- Limited pipeline impact
- Workaround available
- Later stage violations

### Low Severity
- Minor inconvenience
- Cosmetic issues
- Single stage impact
- Easy workaround
- No downstream effects

## Pipeline Stage Analysis

The system analyzes violations in the context of ML pipeline stages:

1. **Data Preprocessing**
   - Input validation
   - Data transformation
   - Feature engineering

2. **Model Construction**
   - Architecture definition
   - Layer configuration
   - Hyperparameter setting

3. **Training**
   - Batch processing
   - Optimization
   - Validation

4. **Inference**
   - Prediction
   - Output formatting
   - Post-processing

## Contract Discovery Process

The system actively looks for new contract types by:

1. Analyzing patterns across multiple issues
2. Identifying gaps in existing categories
3. Evaluating emerging challenges
4. Considering pipeline stage context
5. Tracking error propagation

When suggesting new contracts, it provides:
- Clear name and description
- Rationale for creation
- Example scenarios
- Pipeline stage relevance
- Propagation patterns

## Integration

The system integrates with:
- GitHub API for issue fetching
- OpenAI API for analysis
- Local storage for results

## Configuration

Key settings in `settings.py`:
- OpenAI model selection
- Analysis parameters
- Output formats
- Storage locations
- Pipeline stage definitions

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