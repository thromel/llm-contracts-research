"""
GitHub Data Acquisition for LLM Contracts Research Pipeline.

Captures provider-specific crashes and contract violations from GitHub Issues 
and Discussions across OpenAI, Anthropic, and other LLM provider repositories.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, AsyncGenerator, Dict, Any
import httpx
from tqdm import tqdm

from ..common.models import RawPost, Platform
from ..common.database import MongoDBManager

logger = logging.getLogger(__name__)


class GitHubAcquisition:
    """
    Enhanced GitHub acquisition focused on LLM contract research.

    Builds on existing GitHub fetcher but adds:
    - Multi-repository targeting (OpenAI, Anthropic, etc.)
    - LLM-specific filtering
    - Research provenance tracking
    - Normalized data output
    """

    def __init__(self, github_token: str, db_manager: MongoDBManager, config: Optional[Dict[str, Any]] = None):
        """Initialize GitHub acquisition.

        Args:
            github_token: GitHub API token
            db_manager: MongoDB manager for storage
            config: Optional configuration dict with filtering criteria
        """
        self.token = github_token
        self.db = db_manager
        self.base_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'LLM-Contracts-Research/1.0'
        }

        # Load filtering configuration
        self.config = config or {}
        github_filtering = self.config.get('sources', {}).get(
            'github', {}).get('filtering', {})

        # Configurable filtering criteria
        self.min_comments = github_filtering.get('min_comments', 1)
        self.require_closed = github_filtering.get('require_closed', False)
        self.exclude_labels = github_filtering.get('exclude_labels', [])
        self.check_duplicates = github_filtering.get('check_duplicates', True)

        # LLM Provider repositories (broad selection for maximum coverage)
        self.target_repositories = [
            # OpenAI repositories
            'openai/openai-python',
            'openai/openai-node',
            'openai/openai-cookbook',

            # Anthropic repositories
            'anthropics/anthropic-sdk-python',
            'anthropics/anthropic-sdk-typescript',

            # LangChain
            'langchain-ai/langchain',

            # Google
            'google/generative-ai-python',

            # Microsoft
            'microsoft/semantic-kernel',

            # Multi-provider wrappers
            'BerriAI/litellm',

            # ML/AI frameworks
            'huggingface/transformers',

            # Utility tools
            'simonw/llm',
        ]

        # No filtering patterns - minimal filtering only

    async def acquire_all_repositories(
        self,
        since_days: int = 30,
        max_issues_per_repo: int = 1000,
        include_discussions: bool = True
    ) -> AsyncGenerator[RawPost, None]:
        """
        Acquire data from all target repositories.

        Args:
            since_days: Only fetch issues/discussions from last N days
            max_issues_per_repo: Maximum issues per repository
            include_discussions: Whether to include GitHub Discussions

        Yields:
            RawPost objects for each issue/discussion
        """
        since_date = datetime.utcnow() - timedelta(days=since_days)

        with tqdm(total=len(self.target_repositories), desc="Repositories") as pbar:
            for repo in self.target_repositories:
                try:
                    owner, name = repo.split('/')

                    # Fetch issues
                    async for post in self._fetch_repository_issues(
                        owner, name, since_date, max_issues_per_repo
                    ):
                        yield post

                    # Fetch discussions if enabled
                    if include_discussions:
                        async for post in self._fetch_repository_discussions(
                            owner, name, since_date, max_issues_per_repo
                        ):
                            yield post

                    pbar.update(1)

                except Exception as e:
                    logger.error(
                        f"Error processing repository {repo}: {str(e)}")
                    pbar.update(1)
                    continue

    async def _fetch_repository_issues(
        self,
        owner: str,
        repo: str,
        since: datetime,
        max_issues: int
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch issues from a specific repository."""

        async with httpx.AsyncClient(headers=self.headers, timeout=30) as client:
            page = 1
            issues_fetched = 0

            while issues_fetched < max_issues:
                try:
                    url = f"{self.base_url}/repos/{owner}/{repo}/issues"
                    params = {
                        'state': 'closed' if self.require_closed else 'all',
                        'since': since.isoformat(),
                        'per_page': min(100, max_issues - issues_fetched),
                        'page': page,
                        'sort': 'updated',
                        'direction': 'desc'
                    }

                    # Log filtering configuration for debugging
                    if page == 1:  # Only log on first page to avoid spam
                        logger.info(
                            f"GitHub filtering for {owner}/{repo}: state={params['state']}, min_comments={self.min_comments}, exclude_labels={self.exclude_labels}, check_duplicates={self.check_duplicates}")

                    response = await client.get(url, params=params)

                    if response.status_code == 403:
                        # Rate limit handling
                        reset_time = int(response.headers.get(
                            'X-RateLimit-Reset', 0))
                        wait_time = max(
                            0, reset_time - datetime.now().timestamp())
                        if wait_time > 0:
                            logger.info(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue

                    response.raise_for_status()
                    issues = response.json()

                    if not issues:
                        break

                    for issue in issues:
                        # Skip pull requests (they appear in issues API)
                        if 'pull_request' in issue:
                            continue

                        # Check if we already have this issue (if enabled)
                        if self.check_duplicates:
                            issue_id = str(issue['number'])
                            existing = await self.db.find_one(
                                'raw_posts',
                                {
                                    'platform': 'github',
                                    'source_id': issue_id,
                                    'url': {'$regex': f'{owner}/{repo}'}
                                }
                            )
                            if existing:
                                continue

                        # Stage 1: Check minimum comments
                        if issue.get('comments', 0) < self.min_comments:
                            continue

                        # Stage 2: Check excluded labels
                        if self.exclude_labels:
                            issue_labels = [label['name'].lower()
                                            for label in issue.get('labels', [])]
                            if any(excluded.lower() in issue_labels for excluded in self.exclude_labels):
                                continue

                        # Convert to RawPost (with comments)
                        raw_post = await self._convert_issue_to_rawpost(
                            issue, owner, repo, client
                        )

                        yield raw_post
                        issues_fetched += 1

                        if issues_fetched >= max_issues:
                            break

                    page += 1

                except httpx.HTTPError as e:
                    logger.error(
                        f"HTTP error fetching {owner}/{repo} issues: {str(e)}")
                    break
                except Exception as e:
                    logger.error(
                        f"Error fetching {owner}/{repo} issues: {str(e)}")
                    break

    async def _fetch_repository_discussions(
        self,
        owner: str,
        repo: str,
        since: datetime,
        max_discussions: int
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch discussions using GitHub GraphQL API."""

        # GraphQL query for discussions
        query = """
        query($owner: String!, $name: String!, $after: String) {
          repository(owner: $owner, name: $name) {
            discussions(first: 20, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
                title
                body
                createdAt
                updatedAt
                url
                author {
                  login
                }
                category {
                  name
                }
                upvoteCount
                comments(first: 5) {
                  totalCount
                  nodes {
                    body
                  }
                }
              }
            }
          }
        }
        """

        async with httpx.AsyncClient(headers=self.headers, timeout=30) as client:
            cursor = None
            discussions_fetched = 0

            while discussions_fetched < max_discussions:
                try:
                    variables = {
                        'owner': owner,
                        'name': repo,
                        'after': cursor
                    }

                    response = await client.post(
                        'https://api.github.com/graphql',
                        json={'query': query, 'variables': variables}
                    )

                    response.raise_for_status()
                    data = response.json()

                    if 'errors' in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        break

                    discussions_data = data['data']['repository']['discussions']
                    discussions = discussions_data['nodes']

                    if not discussions:
                        break

                    for discussion in discussions:
                        # Check if discussion is recent enough
                        updated_at = datetime.fromisoformat(
                            discussion['updatedAt'].rstrip('Z')
                        )
                        if updated_at < since:
                            return  # No more recent discussions

                        # No filtering for discussions - include all

                        # Convert to RawPost
                        raw_post = await self._convert_discussion_to_rawpost(
                            discussion, owner, repo
                        )

                        yield raw_post
                        discussions_fetched += 1

                        if discussions_fetched >= max_discussions:
                            break

                    # Check for next page
                    page_info = discussions_data['pageInfo']
                    if not page_info['hasNextPage']:
                        break

                    cursor = page_info['endCursor']

                except Exception as e:
                    logger.error(
                        f"Error fetching {owner}/{repo} discussions: {str(e)}")
                    break

    async def _fetch_issue_comments(
        self,
        issue_number: int,
        owner: str,
        repo: str,
        client: httpx.AsyncClient
    ) -> str:
        """Fetch all comments for an issue."""

        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
            response = await client.get(url)
            response.raise_for_status()

            comments = response.json()
            comments_text = []

            for comment in comments:
                author = comment.get('user', {}).get('login', 'unknown')
                body = comment.get('body', '').strip()
                created_at = comment.get('created_at', '')

                if body:
                    comments_text.append(
                        f"Comment by {author} ({created_at}):\n{body}")

            return '\n\n'.join(comments_text)

        except Exception as e:
            logger.error(
                f"Error fetching comments for issue {issue_number}: {str(e)}")
            return ""

    def _contains_code(self, text: str) -> bool:
        """Stage 1: Check if text contains executable code."""
        if not text:
            return False

        # Code indicators specific to GitHub issues
        code_indicators = [
            '```',  # Markdown code blocks
            '`',    # Inline code
            'import ',  # Python imports
            'from ',  # Python imports
            'def ',  # Python functions
            'class ',  # Python classes
            'function ',  # JavaScript functions
            'const ',  # JavaScript constants
            'let ',  # JavaScript variables
            'var ',  # JavaScript variables
            'openai.',  # OpenAI API calls
            'anthropic.',  # Anthropic API calls
            'client.',  # API client calls
            'await ',  # Async code
            'async ',  # Async code
            '.api.',  # API calls
            'response =',  # API responses
            'request =',  # API requests
            'error',  # Error discussions
            'traceback',  # Python tracebacks
            'stack trace',  # Stack traces
            'exception',  # Exception handling
            'pip install',  # Package installation
            'npm install',  # Package installation
            'curl ',  # API calls
            'POST ',  # HTTP methods
            'GET ',  # HTTP methods
        ]

        text_lower = text.lower()
        code_count = sum(
            1 for indicator in code_indicators if indicator.lower() in text_lower)

        # Require at least 2 code indicators for high confidence
        return code_count >= 2

    async def _convert_issue_to_rawpost(
        self,
        issue: Dict[str, Any],
        owner: str,
        repo: str,
        client: httpx.AsyncClient
    ) -> RawPost:
        """Convert GitHub issue to RawPost format with comments."""

        # Fetch comments for the issue
        comments_text = await self._fetch_issue_comments(
            issue['number'], owner, repo, client
        )

        # Combine body with comments
        body_with_comments = issue.get('body', '') or ''
        if comments_text:
            body_with_comments += f"\n\n--- COMMENTS ---\n{comments_text}"

        return RawPost(
            platform=Platform.GITHUB,
            source_id=str(issue['id']),
            url=issue['html_url'],
            title=issue['title'],
            body_md=body_with_comments,
            created_at=datetime.fromisoformat(issue['created_at'].rstrip('Z')),
            updated_at=datetime.fromisoformat(issue['updated_at'].rstrip('Z')),
            score=0,  # GitHub issues don't have scores like SO
            tags=[f"{owner}/{repo}"],
            author=issue.get('user', {}).get('login', 'unknown'),
            state=issue['state'],
            labels=[label['name'] for label in issue.get('labels', [])],
            comments_count=issue.get('comments', 0),
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

    async def _convert_discussion_to_rawpost(
        self,
        discussion: Dict[str, Any],
        owner: str,
        repo: str
    ) -> RawPost:
        """Convert GitHub discussion to RawPost format."""

        return RawPost(
            platform=Platform.GITHUB,
            source_id=discussion['id'],
            url=discussion['url'],
            title=discussion['title'],
            body_md=discussion.get('body', '') or '',
            created_at=datetime.fromisoformat(
                discussion['createdAt'].rstrip('Z')),
            updated_at=datetime.fromisoformat(
                discussion['updatedAt'].rstrip('Z')),
            score=discussion.get('upvoteCount', 0),
            tags=[f"{owner}/{repo}", "discussion"],
            author=discussion.get('author', {}).get('login', 'unknown'),
            state="open",  # Discussions don't have explicit state
            labels=[discussion.get('category', {}).get('name', 'general')],
            comments_count=discussion.get('comments', {}).get('totalCount', 0),
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

    async def save_to_database(self, raw_post: RawPost) -> str:
        """Save RawPost to MongoDB with deduplication."""

        # Check for existing post
        existing = await self.db.find_one(
            'raw_posts',
            {
                'platform': raw_post.platform.value,
                'source_id': raw_post.source_id
            }
        )

        if existing:
            # Update if newer
            if raw_post.updated_at > existing.get('updated_at', datetime.min):
                await self.db.update_one(
                    'raw_posts',
                    {'_id': existing['_id']},
                    {'$set': raw_post.to_dict()}
                )
                return str(existing['_id'])
            else:
                return str(existing['_id'])
        else:
            # Insert new
            result = await self.db.insert_one('raw_posts', raw_post.to_dict())
            return str(result.inserted_id)

    async def get_repository_stats(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get statistics about a repository's LLM-related content."""

        async with httpx.AsyncClient(headers=self.headers, timeout=30) as client:
            # Get repository info
            repo_response = await client.get(f"{self.base_url}/repos/{owner}/{repo}")
            repo_data = repo_response.json()

            # Get recent issues count
            issues_response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}/issues",
                params={'state': 'all', 'per_page': 1}
            )

            # Extract total count from Link header or response
            total_issues = len(issues_response.json()
                               ) if issues_response.json() else 0
            if 'Link' in issues_response.headers:
                # Parse Link header for total count (simplified)
                link_header = issues_response.headers['Link']
                if 'last' in link_header:
                    # Extract page number from last link
                    import re
                    match = re.search(
                        r'page=(\d+)[^>]*>; rel="last"', link_header)
                    if match:
                        total_issues = int(match.group(1)) * 100  # Approximate

            return {
                'repository': f"{owner}/{repo}",
                'stars': repo_data.get('stargazers_count', 0),
                'forks': repo_data.get('forks_count', 0),
                'language': repo_data.get('language', 'Unknown'),
                'total_issues': total_issues,
                'description': repo_data.get('description', ''),
                'created_at': repo_data.get('created_at'),
                'updated_at': repo_data.get('updated_at')
            }
