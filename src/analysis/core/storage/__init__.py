"""Storage implementations."""

from src.analysis.core.storage.base import ResultsStorage
from src.analysis.core.storage.json_storage import JSONResultsStorage

__all__ = [
    'ResultsStorage',
    'JSONResultsStorage'
]
