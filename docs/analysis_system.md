# LLM API Contract Violation Analysis System

## Overview

This system provides a comprehensive research pipeline for analyzing LLM API contract violations in GitHub issues and Stack Overflow posts. Built on a modern, event-driven architecture, it features multi-modal LLM screening capabilities, intelligent data acquisition, and comprehensive reliability validation.

The pipeline integrates multiple LLM providers (OpenAI, DeepSeek, Anthropic) and screening approaches (Traditional, Agentic, Hybrid) to maximize research quality while maintaining high throughput.

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

## Pipeline Usage

### Quick Start

```bash
# Run complete pipeline with all screening modes
python run_pipeline.py

# Run specific screening modes
python run_pipeline.py --mode traditional    # Traditional LLM screening
python run_pipeline.py --mode agentic        # Multi-agent LLM screening
python run_pipeline.py --mode hybrid         # Both approaches

# Run specific pipeline steps
python run_pipeline.py --step acquisition   # Data acquisition only
python run_pipeline.py --step filtering     # Keyword filtering only
python run_pipeline.py --step screening     # LLM screening only
```

### Programmatic Usage - New Architecture

```python
import asyncio
from pipeline.foundation.config import get_config_manager
from pipeline.orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator, PipelineMode

async def analyze_contract_violations():
    # Initialize configuration
    config = get_config_manager()
    config.load_from_env(".env")
    config.load_from_yaml("pipeline_config.yaml")
    
    # Initialize orchestrator
    orchestrator = UnifiedPipelineOrchestrator(config=config)
    await orchestrator.initialize()
    
    # Run agentic screening for high-quality analysis
    results = await orchestrator.execute_pipeline(
        mode=PipelineMode.AGENTIC,
        max_posts_per_stage=1000
    )
    
    print(f"Analysis completed: {results['total_posts_processed']} posts analyzed")
    return results

# Run analysis
asyncio.run(analyze_contract_violations())
```

### Legacy Compatibility

```python
from pipeline.main_pipeline import ResearchPipelineOrchestrator
from pipeline.common.config import PipelineConfig

# Legacy pipeline still works
config = PipelineConfig(
    mongodb_connection_string="mongodb://localhost:27017/",
    openai_api_key="your-key-here"
)

orchestrator = ResearchPipelineOrchestrator(config)
# Use existing methods...
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

## System Architecture Integration

The enhanced pipeline integrates with multiple modern components:

### Data Sources
- **GitHub API**: Closed issues with comments from major LLM provider repositories
- **Stack Overflow API**: Answered questions with community validation
- **MongoDB**: Async database operations with connection pooling

### LLM Providers
- **OpenAI**: GPT-4 for detailed analysis and borderline screening
- **DeepSeek**: High-throughput bulk screening with DeepSeek-R1
- **Anthropic**: Claude models for agentic screening (optional)

### Infrastructure
- **Event Bus**: Decoupled component communication
- **Circuit Breaker**: Resilient API operations
- **Metrics Collection**: Prometheus-compatible monitoring
- **Structured Logging**: JSON logging with correlation tracking

## Configuration System

The unified configuration system supports:

### Environment Variables
```bash
# Database
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=llm_contracts_research

# LLM APIs
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Pipeline Settings
SCREENING_MODE=agentic  # traditional|agentic|hybrid
MAX_POSTS_PER_RUN=50000
```

### YAML Configuration
```yaml
sources:
  github:
    enabled: true
    repositories:
      - owner: openai
        repo: openai-python
    max_issues_per_repo: 1000
    
  stackoverflow:
    enabled: true
    tags: [openai-api, langchain]
    max_questions_per_tag: 5000

llm_screening:
  mode: agentic
  temperature: 0.1
  max_tokens: 2000
```

### Configuration Validation
- Type-safe Pydantic models
- API key validation
- Database connection testing
- LLM provider availability checks

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