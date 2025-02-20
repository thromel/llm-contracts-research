from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..dto.github import RepositoryDTO, IssueDTO, AnalysisDTO


class DatabaseAdapter(ABC):
    """Base class for database adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    async def create_repository(self, repo: RepositoryDTO) -> str:
        """Create a new repository record."""
        pass

    @abstractmethod
    async def create_issue(self, issue: IssueDTO) -> str:
        """Create a new issue record."""
        pass

    @abstractmethod
    async def create_analysis(self, analysis: AnalysisDTO) -> str:
        """Create a new analysis record."""
        pass

    @abstractmethod
    async def get_repository_by_full_name(self, full_name: str) -> Optional[RepositoryDTO]:
        """Retrieve a repository by its full name."""
        pass

    @abstractmethod
    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[datetime]:
        """Get the timestamp of the last fetched issue for a repository."""
        pass

    @abstractmethod
    async def get_issues_for_repository(
        self,
        repo_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[IssueDTO]:
        """Retrieve issues for a specific repository."""
        pass
