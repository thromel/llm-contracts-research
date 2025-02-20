"""GitHub API client implementation."""
import asyncio
import aiohttp
import backoff
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import re
import urllib.parse


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
        # Extract base GitHub URL from API URL
        self._github_base_url = self._extract_github_base_url(config.api_url)

    def _extract_github_base_url(self, api_url: str) -> str:
        """Extract base GitHub URL from API URL.

        Args:
            api_url: GitHub API URL

        Returns:
            Base GitHub URL (e.g., https://github.com)
        """
        # Handle both github.com and enterprise URLs
        if 'github.com' in api_url:
            return 'https://github.com'
        # For enterprise, remove api/v3 or api prefix
        return re.sub(r'/api(?:/v3)?$', '', api_url)

    def build_github_url(self, owner: str, repo: str, type: str = None, number: Optional[int] = None) -> str:
        """Build a GitHub web URL.

        Args:
            owner: Repository owner
            repo: Repository name
            type: URL type (issues, pulls, etc.)
            number: Optional issue/PR number

        Returns:
            GitHub web URL
        """
        url = f"{self._github_base_url}/{owner}/{repo}"
        if type:
            url = f"{url}/{type}"
        if number is not None:
            url = f"{url}/{number}"
        return url

    def build_api_url(self, owner: str, repo: str, type: str = None, number: Optional[int] = None) -> str:
        """Build a GitHub API URL.

        Args:
            owner: Repository owner
            repo: Repository name
            type: URL type (issues, pulls, etc.)
            number: Optional issue/PR number

        Returns:
            GitHub API URL
        """
        url = f"{self._config.api_url}/repos/{owner}/{repo}"
        if type:
            url = f"{url}/{type}"
        if number is not None:
            url = f"{url}/{number}"
        return url

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
    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Make a GET request to the GitHub API with automatic retries and rate limit handling.

        Args:
            url: The URL to request
            params: Optional query parameters

        Returns:
            Tuple of (response data, response headers)

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
                return await self._get(url, params)

            response.raise_for_status()
            return await response.json(), dict(response.headers)

    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Repository information with URLs
        """
        url = self.build_api_url(owner, repo)
        data, _ = await self._get(url)
        # Add web URLs to response
        data.update({
            'html_url': self.build_github_url(owner, repo),
            'issues_url': self.build_github_url(owner, repo, 'issues'),
            'pulls_url': self.build_github_url(owner, repo, 'pulls')
        })
        return data

    async def get_total_issues_count(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        with_comments_only: bool = True
    ) -> int:
        """Get total number of issues in a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            since: Optional timestamp to fetch issues from
            with_comments_only: Whether to count only issues with comments

        Returns:
            Total number of issues
        """
        url = self.build_api_url(owner, repo, 'issues')
        params = {
            'state': 'all',
            'per_page': 1,
            'page': 1,
            'sort': 'created',
            'direction': 'asc'
        }
        if since:
            params['since'] = since.isoformat()

        data, headers = await self._get(url, params)

        # Get total from Link header
        if 'Link' in headers:
            for link in headers['Link'].split(', '):
                if 'rel="last"' in link:
                    # Extract the last page number using regex
                    match = re.search(r'[&?]page=(\d+)', link)
                    if match:
                        last_page = int(match.group(1))

                        if with_comments_only:
                            # Get the actual count of issues with comments
                            params['per_page'] = 100
                            total_with_comments = 0

                            for page in range(1, last_page + 1):
                                params['page'] = page
                                page_data, _ = await self._get(url, params)
                                total_with_comments += sum(
                                    1 for issue in page_data if issue['comments'] > 0)

                            return total_with_comments
                        else:
                            # If we don't need to filter by comments, return total issues
                            return last_page * self._config.per_page

        # If no pagination info, count issues in the response
        return len(data)

    async def get_repository_issues(
        self,
        owner: str,
        repo: str,
        page: int = 1,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get a page of repository issues.

        Args:
            owner: Repository owner
            repo: Repository name
            page: Page number
            since: Optional timestamp to fetch issues from

        Returns:
            List of issues with URLs
        """
        url = self.build_api_url(owner, repo, 'issues')
        params = {
            'state': 'all',
            'per_page': self._config.per_page,
            'page': page,
            'sort': 'created',
            'direction': 'asc'
        }
        if since:
            params['since'] = since.isoformat()

        data, _ = await self._get(url, params)

        # Add web URLs to each issue
        for issue in data:
            number = issue['number']
            issue.update({
                'html_url': self.build_github_url(owner, repo, 'issues', number),
                'comments_url': self.build_github_url(owner, repo, f'issues/{number}#issuecomment')
            })
        return data

    async def get_issue_comments(self, issue_url: str) -> List[Dict[str, Any]]:
        """Get all comments for an issue.

        Args:
            issue_url: The issue's URL

        Returns:
            List of comments with URLs
        """
        # Extract owner, repo, and issue number from URL
        match = re.search(r'/repos/([^/]+)/([^/]+)/issues/(\d+)', issue_url)
        if not match:
            raise ValueError(f"Invalid issue URL format: {issue_url}")

        owner, repo, number = match.groups()
        comments_url = self.build_api_url(
            owner, repo, f'issues/{number}/comments')

        all_comments = []
        page = 1

        while True:
            params = {'page': page, 'per_page': self._config.per_page}
            page_comments, _ = await self._get(comments_url, params)

            if not page_comments:
                break

            # Add web URLs to each comment
            for comment in page_comments:
                comment_id = comment['id']
                comment.update({
                    'html_url': f"{self.build_github_url(owner, repo, 'issues', int(number))}#issuecomment-{comment_id}",
                    'user': {
                        'login': comment['user']['login'],
                        'id': comment['user']['id'],
                        'html_url': f"{self._github_base_url}/{comment['user']['login']}"
                    },
                    'body': comment['body'],
                    'created_at': datetime.fromisoformat(comment['created_at'].rstrip('Z')),
                    'updated_at': datetime.fromisoformat(comment['updated_at'].rstrip('Z'))
                })
                all_comments.append(comment)

            if len(page_comments) < self._config.per_page:
                break

            page += 1

        return all_comments

    def build_repo_url(self, owner: str, repo: str) -> str:
        """Build the URL for a repository endpoint.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Full GitHub API URL for the repository
        """
        return self.build_api_url(owner, repo)

    def build_issues_url(self, owner: str, repo: str) -> str:
        """Build the URL for repository issues endpoint.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Full GitHub API URL for the repository's issues
        """
        return self.build_api_url(owner, repo, 'issues')

    def build_comments_url(self, issue_url: str) -> str:
        """Build the URL for issue comments endpoint.

        Args:
            issue_url: Base issue URL

        Returns:
            Full GitHub API URL for the issue's comments
        """
        return self.build_api_url(owner, repo, f'issues/{number}/comments')
