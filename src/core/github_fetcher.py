"""GitHub issues fetcher implementation."""
from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any
import logging

from .interfaces import IssuesFetcher, IssueRepository, ProgressTracker, IssueData
from .clients.github_api_client import GitHubAPIClient, GitHubConfig

logger = logging.getLogger(__name__)


class GitHubFetcher(IssuesFetcher):
    """Fetches issues from GitHub repositories."""

    def __init__(
        self,
        api_client: GitHubAPIClient,
        repository: IssueRepository,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """Initialize the GitHub fetcher.

        Args:
            api_client: Client for making GitHub API requests
            repository: Repository for storing issues
            progress_tracker: Optional progress tracker
        """
        self._api_client = api_client
        self._repository = repository
        self._progress_tracker = progress_tracker

    async def _fetch_issue_comments(self, issue_url: str) -> list[Dict[str, Any]]:
        """Fetch comments for an issue.

        Args:
            issue_url: The issue's URL

        Returns:
            List of comment data
        """
        comments = []
        page = 1

        while True:
            params = {'page': page, 'per_page': 100}
            response = await self._api_client.get(
                self._api_client.build_comments_url(issue_url),
                params
            )

            page_comments = response['data']
            if not page_comments:
                break

            comments.extend([{
                'id': comment['id'],
                'user': {
                    'login': comment['user']['login'],
                    'id': comment['user']['id']
                },
                'body': comment['body'],
                'created_at': datetime.fromisoformat(comment['created_at'].rstrip('Z')),
                'updated_at': datetime.fromisoformat(comment['updated_at'].rstrip('Z'))
            } for comment in page_comments])

            page += 1

        return comments

    async def _process_issue(self, issue_data: Dict[str, Any], repository_id: Optional[str] = None) -> IssueData:
        """Convert raw issue data to IssueData object.

        Args:
            issue_data: Raw issue data from GitHub API
            repository_id: Optional repository ID

        Returns:
            Processed IssueData object
        """
        # Fetch comments for the issue
        comments = await self._fetch_issue_comments(issue_data['url'])

        return IssueData(
            github_issue_id=issue_data['id'],
            repository_id=repository_id,
            number=issue_data['number'],
            title=issue_data['title'],
            body=issue_data['body'] or '',
            state=issue_data['state'],
            created_at=datetime.fromisoformat(
                issue_data['created_at'].rstrip('Z')),
            updated_at=datetime.fromisoformat(
                issue_data['updated_at'].rstrip('Z')),
            closed_at=(
                datetime.fromisoformat(issue_data['closed_at'].rstrip('Z'))
                if issue_data['closed_at']
                else None
            ),
            labels=[label['name'] for label in issue_data['labels']],
            assignees={
                assignee['login']: assignee['id']
                for assignee in issue_data['assignees']
            },
            comments=comments,
            url=issue_data['url']
        )

    async def fetch_repository_issues(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        max_issues: Optional[int] = None,
    ) -> AsyncGenerator[IssueData, None]:
        """Fetch all issues for a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            since: Optional timestamp to fetch issues from
            max_issues: Optional maximum number of issues to fetch

        Yields:
            IssueData objects for each issue
        """
        # First get repository info
        try:
            repo_response = await self._api_client.get(
                self._api_client.build_repo_url(owner, repo)
            )
            repo_data = repo_response['data']
            repository_id = str(repo_data['id'])
        except Exception as e:
            logger.error(
                f"Error fetching repository info for {owner}/{repo}: {str(e)}")
            return

        issue_count = 0
        page = 1

        while True:
            if max_issues and issue_count >= max_issues:
                logger.debug(f"Reached max issues limit: {max_issues}")
                break

            params = {
                'state': 'all',
                'per_page': 100,
                'page': page,
                'sort': 'created',
                'direction': 'asc'
            }

            if since:
                params['since'] = since.isoformat()

            try:
                response = await self._api_client.get(
                    self._api_client.build_issues_url(owner, repo),
                    params
                )

                issues = response['data']
                if not issues:
                    break

                logger.debug(
                    f"Processing page {page} with {len(issues)} issues")

                for issue in issues:
                    if max_issues and issue_count >= max_issues:
                        break

                    try:
                        issue_data = await self._process_issue(issue, repository_id)
                        await self._repository.save_issue(issue_data)

                        if self._progress_tracker:
                            self._progress_tracker.update()

                        issue_count += 1
                        yield issue_data

                    except Exception as e:
                        logger.error(
                            f"Error processing issue #{issue.get('number', 'unknown')}: {str(e)}")
                        continue

                page += 1

            except Exception as e:
                logger.error(f"Error fetching page {page}: {str(e)}")
                break

        logger.debug(
            f"Completed fetching {issue_count} issues for {owner}/{repo}")
