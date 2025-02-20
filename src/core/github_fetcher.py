"""GitHub issues fetcher implementation."""
from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any, List
import logging

from .interfaces import IssuesFetcher, IssueRepository, ProgressTracker, IssueData
from .clients.github_api_client import GitHubAPIClient
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class GitHubFetcher(IssuesFetcher):
    """Fetches issues from GitHub repositories with robust checkpointing."""

    def __init__(
        self,
        api_client: GitHubAPIClient,
        repository: IssueRepository,
        progress_tracker: Optional[ProgressTracker] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """Initialize the GitHub fetcher.

        Args:
            api_client: Client for making GitHub API requests
            repository: Repository for storing issues
            progress_tracker: Optional progress tracker
            checkpoint_manager: Optional checkpoint manager for saving/resuming state
        """
        self._api_client = api_client
        self._repository = repository
        self._progress_tracker = progress_tracker
        self._checkpoint_manager = checkpoint_manager

    def _create_issue_dto(
        self,
        raw_issue: Dict[str, Any],
        repository_id: str,
        comments: List[Dict[str, Any]]
    ) -> IssueData:
        """Convert raw issue data to IssueData object.

        Args:
            raw_issue: Raw issue data from GitHub API
            repository_id: Repository ID
            comments: List of comments for the issue

        Returns:
            Processed IssueData object
        """
        return IssueData(
            github_issue_id=raw_issue['id'],
            repository_id=repository_id,
            number=raw_issue['number'],
            title=raw_issue['title'],
            body=raw_issue['body'] or '',
            state=raw_issue['state'],
            created_at=datetime.fromisoformat(
                raw_issue['created_at'].rstrip('Z')),
            updated_at=datetime.fromisoformat(
                raw_issue['updated_at'].rstrip('Z')),
            closed_at=(
                datetime.fromisoformat(raw_issue['closed_at'].rstrip('Z'))
                if raw_issue.get('closed_at')
                else None
            ),
            labels=[label['name'] for label in raw_issue.get('labels', [])],
            assignees={
                assignee['login']: assignee['id']
                for assignee in raw_issue.get('assignees', [])
            },
            comments=comments,
            url=raw_issue['url']
        )

    async def _fetch_issue_comments(self, issue_url: str) -> List[Dict[str, Any]]:
        """Fetch all comments for an issue.

        Args:
            issue_url: The issue's URL

        Returns:
            List of comments
        """
        comments = []
        page = 1

        while True:
            params = {'page': page,
                      'per_page': self._api_client._config.per_page}
            page_comments = await self._api_client.get_issue_comments(issue_url)

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

            if len(page_comments) < self._api_client._config.per_page:
                break

            page += 1

        return comments

    async def fetch_repository_issues(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        max_issues: Optional[int] = None,
    ) -> AsyncGenerator[IssueData, None]:
        """Fetch all issues for a repository with checkpointing.

        Args:
            owner: Repository owner
            repo: Repository name
            since: Optional timestamp to fetch issues from
            max_issues: Optional maximum number of issues to fetch

        Yields:
            IssueData objects for each issue with comments
        """
        try:
            # Load checkpoint state if available
            state = {}
            if self._checkpoint_manager:
                state = await self._checkpoint_manager.load_checkpoint(owner, repo) or {}
                if state:
                    logger.info(
                        f"Resuming fetch for {owner}/{repo} from page {state.get('page', 1)}")

            current_page = state.get('page', 1)
            issue_count = state.get('issue_count', 0)
            skipped_count = 0  # Track skipped issues for accurate progress

            # Get repository info
            repo_info = await self._api_client.get_repository(owner, repo)
            repository_id = str(repo_info['id'])

            # Initialize progress tracking if needed
            if self._progress_tracker:
                self._progress_tracker.start_operation(
                    total=None,  # Don't set total, just show current progress
                    description=f"Fetching {owner}/{repo} issues with comments (page {current_page})"
                )

            while True:
                if max_issues and issue_count >= max_issues:
                    logger.info(
                        f"Reached max issues limit ({max_issues}) for {owner}/{repo}")
                    break

                # Fetch page of issues
                try:
                    issues = await self._api_client.get_repository_issues(
                        owner=owner,
                        repo=repo,
                        page=current_page,
                        since=since
                    )
                except Exception as e:
                    logger.error(
                        f"Error fetching page {current_page} for {owner}/{repo}: {e}")
                    # Save checkpoint before breaking
                    if self._checkpoint_manager:
                        await self._checkpoint_manager.save_checkpoint(owner, repo, {
                            'page': current_page,
                            'issue_count': issue_count
                        })
                    raise

                if not issues:
                    logger.info(
                        f"No more issues found for {owner}/{repo} after page {current_page}")
                    break

                # Process each issue
                for issue in issues:
                    if max_issues and issue_count >= max_issues:
                        break

                    # Skip issues without comments
                    if issue['comments'] == 0:
                        skipped_count += 1
                        continue

                    try:
                        # Fetch comments
                        comments = await self._api_client.get_issue_comments(issue['url'])

                        # Create and save issue DTO
                        issue_data = self._create_issue_dto(
                            issue, repository_id, comments)
                        await self._repository.save_issue(issue_data)

                        issue_count += 1
                        if self._progress_tracker:
                            self._progress_tracker.update(
                                description=(
                                    f"Fetched {issue_count} issues with comments from {owner}/{repo} "
                                    f"(page {current_page}, skipped {skipped_count})"
                                )
                            )

                        # Save checkpoint after each issue
                        if self._checkpoint_manager:
                            await self._checkpoint_manager.save_checkpoint(owner, repo, {
                                'page': current_page,
                                'issue_count': issue_count,
                                'skipped_count': skipped_count
                            })

                        yield issue_data

                    except Exception as e:
                        logger.error(
                            f"Error processing issue #{issue.get('number', 'unknown')}: {str(e)}")
                        # Save checkpoint on error
                        if self._checkpoint_manager:
                            await self._checkpoint_manager.save_checkpoint(owner, repo, {
                                'page': current_page,
                                'issue_count': issue_count,
                                'skipped_count': skipped_count
                            })
                        continue

                current_page += 1

        except Exception as e:
            logger.error(f"Error fetching issues for {owner}/{repo}: {str(e)}")
            raise

        finally:
            if self._progress_tracker:
                self._progress_tracker.complete()

            # Only clear checkpoint if we completed successfully
            if self._checkpoint_manager and not max_issues:
                await self._checkpoint_manager.clear_checkpoint(owner, repo)
                logger.info(
                    f"Cleared checkpoint for {owner}/{repo} after successful completion")

            logger.info(
                f"Completed fetching {issue_count} issues with comments for {owner}/{repo} "
                f"(skipped {skipped_count} issues without comments)"
            )
