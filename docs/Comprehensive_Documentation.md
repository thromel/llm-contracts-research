# Comprehensive Documentation for LLM Contracts Research

## Introduction

This project is a comprehensive research pipeline for analyzing LLM API contract violations in GitHub issues and Stack Overflow posts. It features a modern, event-driven architecture with unified orchestration, multi-modal LLM screening capabilities, and comprehensive testing infrastructure.

## Features

- **Unified Pipeline Architecture**: Modern layered architecture with event-driven communication
- **Multi-Modal LLM Screening**: Traditional, Agentic, and Hybrid screening approaches
- **Enhanced Data Acquisition**: Quality-focused collection from GitHub and Stack Overflow
- **Intelligent Preprocessing**: Advanced keyword filtering with 93%+ recall
- **Type-Safe Operations**: Comprehensive Pydantic models and abstract interfaces
- **Comprehensive Testing**: 121+ tests with async support and fixtures
- **Full Observability**: Structured logging, metrics collection, and monitoring

## Architecture Overview

The pipeline features a modern, layered architecture designed for scalability, maintainability, and testability:

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                      │
│  • UnifiedPipelineOrchestrator  • PipelineStepExecutor     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Core Layer                            │
│  • Interfaces (Abstract Base Classes)                       │
│  • Event System (Pub/Sub Event Bus)                        │
│  • Exception Hierarchy                                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Foundation Layer                         │
│  • Configuration Management  • Logging System               │
│  • Retry & Circuit Breaker  • Type Definitions            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                       │
│  • Database Manager (MongoDB)  • Monitoring & Metrics       │
│  • Storage Abstraction        • External API Clients       │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Components

- **UnifiedPipelineOrchestrator**: Central coordinator for all pipeline operations with mode-based execution
- **Event Bus**: Enables decoupled communication between components via pub/sub architecture
- **Configuration Manager**: Unified configuration from environment variables and YAML files
- **Storage Abstraction**: Database-agnostic storage layer with MongoDB implementation
- **Circuit Breaker Pattern**: Resilient operations with automatic failure handling

## Usage Examples

### Quick Start - Full Pipeline

```bash
# Run complete research pipeline
python run_pipeline.py

# Run specific pipeline steps
python run_pipeline.py --step acquisition   # Data acquisition only
python run_pipeline.py --step screening     # LLM screening only
python run_pipeline.py --stats-only         # Show statistics
```

### Architecture Validation

```bash
# Test new unified architecture
python -c "
from pipeline.foundation.config import ConfigManager
from pipeline.orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator
from pipeline.core.events import EventBus
print('✅ New Architecture OK')
"

# Test legacy compatibility
python -c "
from pipeline.main_pipeline import ResearchPipelineOrchestrator
from run_pipeline import ModernPipelineRunner
print('✅ Legacy Compatibility OK')
"
```

### Programmatic Usage - New Architecture

```python
import asyncio
from pipeline.foundation.config import ConfigManager, get_config_manager
from pipeline.orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator, PipelineMode

async def run_research_pipeline():
    # Load configuration
    config_manager = get_config_manager()
    config_manager.load_from_env(".env")
    config_manager.load_from_yaml("pipeline_config.yaml")
    config_manager.validate()
    
    # Initialize orchestrator
    orchestrator = UnifiedPipelineOrchestrator(config=config_manager)
    await orchestrator.initialize()
    
    # Run full research pipeline
    results = await orchestrator.execute_pipeline(
        mode=PipelineMode.RESEARCH,
        max_posts_per_stage=1000
    )
    
    print(f"Pipeline completed: {results['total_posts_processed']} posts processed")
    await orchestrator.cleanup()

# Run the pipeline
asyncio.run(run_research_pipeline())
```

### Component Testing

```bash
# Run comprehensive test suite (121+ tests)
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_core/test_events.py -v
python -m pytest tests/test_foundation/ -v
python -m pytest tests/test_infrastructure/ -v
```

## Additional Information

For more detailed guidelines, please refer to:
- **Contributor Guidelines**: Review docs/guidelines_claude_new_contracts.md for coding standards and contribution procedures.
- **System Analysis**: Refer to docs/analysis_system.md for an in-depth look at the system architecture and analytical components.

We welcome feedback, contributions, and suggestions to continuously improve this project. 