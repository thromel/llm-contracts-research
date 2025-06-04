"""Common types and enums for the pipeline."""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class ScreeningMode(Enum):
    """LLM screening modes."""
    TRADITIONAL = "traditional"
    AGENTIC = "agentic"
    HYBRID = "hybrid"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


class PipelineStage(Enum):
    """Pipeline processing stages."""
    DATA_ACQUISITION = "data_acquisition"
    KEYWORD_FILTERING = "keyword_filtering"
    LLM_SCREENING = "llm_screening"
    HUMAN_LABELING = "human_labeling"
    RELIABILITY_VALIDATION = "reliability_validation"
    ANALYSIS = "analysis"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error."""
    field: str
    message: str
    value: Any = None

    def __str__(self):
        return f"Configuration error in '{self.field}': {self.message}"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    stage: PipelineStage
    duration_seconds: float
    items_processed: int
    success_count: int
    error_count: int
    metadata: Dict[str, Any] = None