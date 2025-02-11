"""Base protocol for GitHub clients."""

from typing import Protocol, List
from src.analysis.core.dto import GithubIssueDTO


class GitHubClient(Protocol):
    """Protocol for GitHub API clients."""

    def fetch_issues(self, repo_name: str, num_issues: int) -> List[GithubIssueDTO]:
        """Fetch issues from a GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            num_issues: Number of issues to fetch

        Returns:
            List of GitHub issues as DTOs
        """
        pass

    def get_repo_info(self, repo_name: str) -> dict:
        """Get repository information.

        Args:
            repo_name: Repository name (owner/repo)

        Returns:
            Repository information
        """
        pass
