"""Interfaces for the core package."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator, List
from dataclasses import dataclass, field


@dataclass
class IssueData:
    """Data transfer object for issue data."""
    github_issue_id: int
    repository_id: Optional[str]
    number: int
    title: str
    body: str
    state: str
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    labels: List[str]
    assignees: Dict[str, int]
    url: str
    comments: List[Dict[str, Any]] = field(default_factory=list)


class IssueRepository(ABC):
    """Interface for issue storage."""

    @abstractmethod
    async def save_issue(self, issue: IssueData) -> str:
        """Save an issue to storage."""
        pass

    @abstractmethod
    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[datetime]:
        """Get the timestamp of the last fetched issue."""
        pass


class IssuesFetcher(ABC):
    """Interface for fetching issues."""

    @abstractmethod
    async def fetch_repository_issues(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        max_issues: Optional[int] = None,
    ) -> AsyncGenerator[IssueData, None]:
        """Fetch issues from a repository."""
        pass


class ProgressTracker(ABC):
    """Interface for tracking progress."""

    @abstractmethod
    def start_operation(self, total: Optional[int], description: str) -> None:
        """Start tracking an operation.

        Args:
            total: Total number of items to process (None for unknown)
            description: Description of the operation
        """
        pass

    @abstractmethod
    def update(self, amount: int = 1, description: Optional[str] = None) -> None:
        """Update progress.

        Args:
            amount: Amount to increment progress by
            description: Optional new description
        """
        pass

    @abstractmethod
    def complete(self) -> None:
        """Complete the operation."""
        pass
