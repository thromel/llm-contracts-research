import asyncio
import aiohttp
import backoff
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator, List
from tqdm.asyncio import tqdm
from .config import config
from .db.base import DatabaseAdapter
from .dto.github import RepositoryDTO, IssueDTO
from analysis.core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class GitHubFetcher:
    def __init__(self, db_adapter: DatabaseAdapter):
        self.db = db_adapter
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {config.github.token}',
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=config.github.max_retries,
        max_time=300
    )
    async def _make_request(self, url: str, params: dict = None) -> dict:
        """Make a GitHub API request with exponential backoff retry."""
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Use 'async with' context manager.")

        async with self.session.get(url, params=params) as response:
            if response.status == 403:
                # Rate limit exceeded
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(0, reset_time - datetime.now().timestamp())
                logger.warning(
                    f"Rate limit exceeded. Waiting {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                return await self._make_request(url, params)

            response.raise_for_status()
            return await response.json()

    async def fetch_repository_issues(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        max_issues: Optional[int] = None,
        include_closed: bool = True,
        progress_bar: Optional[tqdm] = None
    ) -> AsyncGenerator[IssueDTO, None]:
        """Fetch all issues for a repository with pagination and checkpointing."""
        repo_url = f"{config.github.api_url}/repos/{owner}/{repo}"

        # First, ensure repository exists in database
        repo_data = await self._make_request(repo_url)
        repo_dto = RepositoryDTO(
            github_repo_id=repo_data['id'],
            owner=owner,
            name=repo,
            full_name=f"{owner}/{repo}",
            url=repo_url,
            created_at=datetime.fromisoformat(
                repo_data['created_at'].rstrip('Z')),
            updated_at=datetime.fromisoformat(
                repo_data['updated_at'].rstrip('Z'))
        )
        repo_id = await self.db.create_repository(repo_dto)

        # Create progress bar for issues if enabled
        issue_count = 0
        issues_pbar = None
        if progress_bar is not None:
            # Get total issue count
            total_issues = repo_data.get('open_issues_count', 0)
            if include_closed:
                closed_issues = await self._make_request(f"{repo_url}/issues", {'state': 'closed', 'per_page': 1})
                total_issues += len(closed_issues)
            if max_issues:
                total_issues = min(total_issues, max_issues)
            issues_pbar = tqdm(
                total=total_issues, desc=f"Issues for {owner}/{repo}", unit="issue", leave=False)

        # Fetch issues with pagination
        page = 1
        while True:
            params = {
                'state': 'all' if include_closed else 'open',
                'per_page': config.github.per_page,
                'page': page,
                'sort': 'updated',
                'direction': 'asc',
            }

            if since:
                params['since'] = since.isoformat()

            issues_url = f"{repo_url}/issues"
            issues = await self._make_request(issues_url, params)

            if not issues:
                break

            for issue in issues:
                # Transform issue data to DTO
                issue_dto = IssueDTO(
                    github_issue_id=issue['id'],
                    repository_id=repo_id,
                    title=issue['title'],
                    body=issue['body'] or '',
                    status=issue['state'],
                    labels=[label['name'] for label in issue['labels']],
                    assignees={
                        assignee['login']: assignee['id']
                        for assignee in issue['assignees']
                    },
                    created_at=datetime.fromisoformat(
                        issue['created_at'].rstrip('Z')),
                    updated_at=datetime.fromisoformat(
                        issue['updated_at'].rstrip('Z')),
                    closed_at=(
                        datetime.fromisoformat(issue['closed_at'].rstrip('Z'))
                        if issue['closed_at']
                        else None
                    )
                )

                # Store in database and yield
                await self.db.create_issue(issue_dto)
                yield issue_dto

                # Update progress
                issue_count += 1
                if issues_pbar:
                    issues_pbar.update(1)

                # Check if we've reached the maximum issues
                if max_issues and issue_count >= max_issues:
                    if issues_pbar:
                        issues_pbar.close()
                    return

            page += 1

        if issues_pbar:
            issues_pbar.close()

    async def fetch_repositories(
        self,
        repo_list: List[str],
        progress_bar: Optional[tqdm] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        since_date: Optional[datetime] = None,
        max_issues: Optional[int] = None,
        include_closed: bool = True
    ) -> None:
        """Fetch issues from multiple repositories with concurrency control."""
        sem = asyncio.Semaphore(config.max_concurrent_requests)

        # Load checkpoint if available
        start_index = 0
        if checkpoint_manager:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                start_index = checkpoint_data['current_index']

        async def fetch_with_semaphore(repo_spec: str, index: int) -> None:
            async with sem:
                try:
                    owner, repo = repo_spec.split('/')
                    # Get last updated timestamp for checkpoint
                    repo_dto = await self.db.get_repository_by_full_name(f"{owner}/{repo}")
                    if repo_dto:
                        last_updated = await self.db.get_last_issue_timestamp(repo_dto.id)
                        since = max(
                            last_updated, since_date) if since_date else last_updated
                    else:
                        since = since_date

                    logger.info(f"Fetching issues for {repo_spec}")
                    issues_processed = 0
                    async for issue in self.fetch_repository_issues(
                        owner, repo, since, max_issues, include_closed, progress_bar
                    ):
                        issues_processed += 1
                        if checkpoint_manager and issues_processed % config.checkpoint_interval == 0:
                            checkpoint_manager.save_checkpoint(
                                analyzed_issues=[],  # TODO: Implement if needed
                                current_index=index,
                                total_issues=[],  # TODO: Implement if needed
                                repo_name=repo_spec
                            )

                    if progress_bar:
                        progress_bar.update(1)

                except Exception as e:
                    logger.error(f"Error fetching {repo_spec}: {str(e)}")
                    raise

        tasks = [
            fetch_with_semaphore(repo, i)
            for i, repo in enumerate(repo_list[start_index:], start=start_index)
        ]
        await asyncio.gather(*tasks)
