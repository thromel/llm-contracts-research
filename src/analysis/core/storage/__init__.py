"""Storage implementations."""

from .base import ResultsStorage
from .json_storage import JSONResultsStorage

__all__ = [
    'ResultsStorage',
    'JSONResultsStorage'
]
