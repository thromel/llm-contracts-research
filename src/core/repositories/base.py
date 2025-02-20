"""Base repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ..dto.github import RepositoryDTO, IssueDTO, AnalysisDTO


class BaseRepository(ABC):
    """Base repository interface defining common operations."""

    @abstractmethod
    async def save_repository(self, repo: RepositoryDTO) -> str:
        """Save a repository record.

        Args:
            repo: Repository data transfer object

        Returns:
            Repository ID
        """
        pass

    @abstractmethod
    async def save_issue(self, issue: IssueDTO) -> str:
        """Save an issue record.

        Args:
            issue: Issue data transfer object

        Returns:
            Issue ID
        """
        pass

    @abstractmethod
    async def save_analysis(self, analysis: AnalysisDTO) -> str:
        """Save an analysis record.

        Args:
            analysis: Analysis data transfer object

        Returns:
            Analysis ID
        """
        pass

    @abstractmethod
    async def get_repository_by_full_name(self, full_name: str) -> Optional[RepositoryDTO]:
        """Get repository by full name.

        Args:
            full_name: Repository full name (owner/repo)

        Returns:
            Repository DTO if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[datetime]:
        """Get timestamp of last fetched issue for repository.

        Args:
            repo_id: Repository ID

        Returns:
            Timestamp of last issue if exists, None otherwise
        """
        pass

    @abstractmethod
    async def get_repository_issues(
        self,
        repo_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[IssueDTO]:
        """Get issues for a repository.

        Args:
            repo_id: Repository ID
            limit: Maximum number of issues to return
            offset: Number of issues to skip

        Returns:
            List of issue DTOs
        """
        pass

    @abstractmethod
    async def get_issue_by_number(self, repo_id: str, issue_number: int) -> Optional[IssueDTO]:
        """Get issue by number.

        Args:
            repo_id: Repository ID
            issue_number: Issue number

        Returns:
            Issue DTO if found, None otherwise
        """
        pass
