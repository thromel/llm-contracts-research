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

    def __init__(self, github_token: str, db_manager: MongoDBManager):
        """Initialize GitHub acquisition.

        Args:
            github_token: GitHub API token
            db_manager: MongoDB manager for storage
        """
        self.token = github_token
        self.db = db_manager
        self.base_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'LLM-Contracts-Research/1.0'
        }

        # LLM Provider repositories to target
        self.target_repositories = [
            # OpenAI
            'openai/openai-python',
            'openai/openai-cookbook',
            'openai/openai-node',

            # Anthropic
            'anthropics/anthropic-sdk-python',
            'anthropics/anthropic-sdk-typescript',

            # Other major LLM providers
            'langchain-ai/langchain',
            'huggingface/transformers',
            'microsoft/semantic-kernel',
            'google/generative-ai-python',

            # Community/wrapper libraries
            'hwchase17/langchain',
            'simonw/llm',
            'BerriAI/litellm'
        ]

        # LLM-specific keywords for initial filtering
        self.llm_keywords = [
            # API contract terms
            'max_tokens', 'temperature', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'stream', 'logprobs',

            # Error patterns
            'rate_limit', 'context_length', 'token_limit', 'quota',
            'invalid_request', 'model_not_found', 'insufficient_quota',

            # JSON/Schema related
            'json_schema', 'function_calling', 'tools', 'response_format',

            # Common error scenarios
            'timeout', 'connection_error', 'authentication', 'api_key'
        ]

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
                        'state': 'all',
                        'since': since.isoformat(),
                        'per_page': min(100, max_issues - issues_fetched),
                        'page': page,
                        'sort': 'updated',
                        'direction': 'desc'
                    }

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

                        # Quick LLM-relevance filter
                        if not self._is_potentially_llm_relevant(issue):
                            continue

                        # Convert to RawPost
                        raw_post = await self._convert_issue_to_rawpost(
                            issue, owner, repo
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

                        # Quick LLM-relevance filter
                        if not self._is_potentially_llm_relevant_discussion(discussion):
                            continue

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

    def _is_potentially_llm_relevant(self, issue: Dict[str, Any]) -> bool:
        """Quick check if issue might be LLM-related."""
        text_to_check = f"{issue.get('title', '')} {issue.get('body', '')}"
        text_lower = text_to_check.lower()

        # Check for LLM keywords
        for keyword in self.llm_keywords:
            if keyword.lower() in text_lower:
                return True

        # Check labels
        labels = [label.get('name', '').lower()
                  for label in issue.get('labels', [])]
        llm_labels = ['api', 'bug', 'error', 'rate-limit', 'timeout', 'json']
        for label in llm_labels:
            if any(label in l for l in labels):
                return True

        return False

    def _is_potentially_llm_relevant_discussion(self, discussion: Dict[str, Any]) -> bool:
        """Quick check if discussion might be LLM-related."""
        text_to_check = f"{discussion.get('title', '')} {discussion.get('body', '')}"
        text_lower = text_to_check.lower()

        # Check for LLM keywords
        for keyword in self.llm_keywords:
            if keyword.lower() in text_lower:
                return True

        # Check category
        category = discussion.get('category', {}).get('name', '').lower()
        relevant_categories = ['help', 'q&a', 'support', 'bugs', 'general']
        if any(cat in category for cat in relevant_categories):
            return True

        return False

    async def _convert_issue_to_rawpost(
        self,
        issue: Dict[str, Any],
        owner: str,
        repo: str
    ) -> RawPost:
        """Convert GitHub issue to RawPost format."""

        return RawPost(
            platform=Platform.GITHUB,
            source_id=str(issue['id']),
            url=issue['html_url'],
            title=issue['title'],
            body_md=issue.get('body', '') or '',
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
