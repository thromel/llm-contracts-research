"""
Storage abstraction layer for the LLM Contracts Research Pipeline.

This module provides abstract storage interfaces and implementations,
allowing the pipeline to work with different storage backends.
"""

from .base import StorageAdapter, StorageTransaction
from .mongodb import MongoDBAdapter
from .factory import StorageFactory
from .repositories import (
    RawPostRepository,
    FilteredPostRepository,
    ScreeningResultRepository,
    LabeledPostRepository
)

__all__ = [
    'StorageAdapter',
    'StorageTransaction',
    'MongoDBAdapter',
    'StorageFactory',
    'RawPostRepository',
    'FilteredPostRepository',
    'ScreeningResultRepository',
    'LabeledPostRepository'
]