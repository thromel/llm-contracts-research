"""
Core components for the LLM Contracts Research Pipeline.

This module contains the foundational classes and interfaces used throughout
the pipeline, including abstract base classes, interfaces, and core utilities.
"""

from .interfaces import (
    DataSource,
    Filter,
    Screener,
    Storage,
    Pipeline,
    PipelineStep,
    ConfigProvider
)

from .exceptions import (
    PipelineError,
    DataAcquisitionError,
    FilteringError,
    ScreeningError,
    StorageError,
    ConfigurationError,
    ValidationError
)

from .events import (
    Event,
    EventBus,
    PipelineEvent,
    DataProcessedEvent,
    ErrorEvent
)

__all__ = [
    # Interfaces
    'DataSource',
    'Filter',
    'Screener',
    'Storage',
    'Pipeline',
    'PipelineStep',
    'ConfigProvider',
    
    # Exceptions
    'PipelineError',
    'DataAcquisitionError',
    'FilteringError',
    'ScreeningError',
    'StorageError',
    'ConfigurationError',
    'ValidationError',
    
    # Events
    'Event',
    'EventBus',
    'PipelineEvent',
    'DataProcessedEvent',
    'ErrorEvent'
]