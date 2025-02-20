"""GitHub API client implementation."""
import asyncio
import aiohttp
import backoff
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GitHubConfig:
    """Configuration for GitHub API client."""
    api_url: str
    token: str
    max_retries: int = 3
    per_page: int = 100


class GitHubAPIClient:
    """Handles low-level HTTP requests to GitHub API."""

    def __init__(self, config: GitHubConfig):
        """Initialize the GitHub API client.

        Args:
            config: Configuration for the GitHub API client
        """
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {config.token}',
        }

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=300
    )
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the GitHub API with automatic retries and rate limit handling.

        Args:
            url: The URL to request
            params: Optional query parameters

        Returns:
            Dict containing response data and metadata

        Raises:
            aiohttp.ClientError: If the request fails
            RuntimeError: If called outside context manager
        """
        if not self._session:
            raise RuntimeError(
                "Session not initialized. Use 'async with' context manager.")

        async with self._session.get(url, params=params) as response:
            if response.status == 403:
                # Handle rate limiting
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(0, reset_time - datetime.now().timestamp())
                await asyncio.sleep(wait_time)
                return await self.get(url, params)

            response.raise_for_status()
            return {
                'data': await response.json(),
                'headers': dict(response.headers),
            }

    def build_repo_url(self, owner: str, repo: str) -> str:
        """Build the URL for a repository endpoint.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Full GitHub API URL for the repository
        """
        return f"{self._config.api_url}/repos/{owner}/{repo}"

    def build_issues_url(self, owner: str, repo: str) -> str:
        """Build the URL for repository issues endpoint.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Full GitHub API URL for the repository's issues
        """
        return f"{self.build_repo_url(owner, repo)}/issues"

    def build_comments_url(self, issue_url: str) -> str:
        """Build the URL for issue comments endpoint.

        Args:
            issue_url: Base issue URL

        Returns:
            Full GitHub API URL for the issue's comments
        """
        return f"{issue_url}/comments"
