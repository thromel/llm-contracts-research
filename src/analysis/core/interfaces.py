"""Core interfaces for the analysis system."""

from abc import ABC, abstractmethod
from typing import List, Optional, Generic, TypeVar
from datetime import datetime

from src.analysis.core.dto import (
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO
)

T = TypeVar('T')


class IAnalyzer(ABC):
    """Interface for analysis operations."""

    @abstractmethod
    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> ContractAnalysisDTO:
        """Analyze a single issue."""
        pass


class IResultWriter(ABC):
    """Interface for writing analysis results."""

    @abstractmethod
    def save_results(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis results."""
        pass


class IResultReader(ABC):
    """Interface for reading analysis results."""

    @abstractmethod
    def load_results(self, identifier: str) -> AnalysisResultsDTO:
        """Load analysis results."""
        pass


class IStorage(IResultWriter, IResultReader):
    """Combined interface for full storage operations."""
    pass


class ICheckpointManager(ABC):
    """Interface for checkpoint operations."""

    @abstractmethod
    def save_checkpoint(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self) -> Optional[AnalysisResultsDTO]:
        """Load latest checkpoint."""
        pass

    @abstractmethod
    def clear_checkpoint(self) -> None:
        """Clear all checkpoints."""
        pass


class IResponseCleaner(ABC):
    """Interface for cleaning analysis responses."""

    @abstractmethod
    def clean(self, content: str) -> str:
        """Clean response content."""
        pass


class IResponseValidator(ABC):
    """Interface for validating analysis responses."""

    @abstractmethod
    def validate(self, content: dict) -> None:
        """Validate response content."""
        pass


class IStorageFactory(ABC):
    """Interface for storage factory."""

    @abstractmethod
    def create_storage(self, storage_types: Optional[List[str]] = None, is_intermediate: bool = False) -> List[IResultWriter]:
        """Create storage implementations."""
        pass


class IAnalysisOrchestrator(ABC):
    """Interface for orchestrating the analysis process."""

    @abstractmethod
    def run_analysis(self, repo_name: str, num_issues: int, checkpoint_interval: int = 10) -> AnalysisResultsDTO:
        """Run the complete analysis process."""
        pass


class IProgressTracker(ABC):
    """Interface for tracking analysis progress."""

    @abstractmethod
    def update(self, current: int, total: int, message: str) -> None:
        """Update progress."""
        pass

    @abstractmethod
    def complete(self) -> None:
        """Mark progress as complete."""
        pass


class IAnalysisRepository(Generic[T]):
    """Generic interface for analysis data repositories."""

    @abstractmethod
    def save(self, entity: T) -> None:
        """Save an entity."""
        pass

    @abstractmethod
    def get(self, identifier: str) -> Optional[T]:
        """Get an entity by identifier."""
        pass

    @abstractmethod
    def get_all(self) -> List[T]:
        """Get all entities."""
        pass

    @abstractmethod
    def delete(self, identifier: str) -> None:
        """Delete an entity."""
        pass
