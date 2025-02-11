"""GitHub API client implementation."""

import logging
from typing import List, Dict, Any
from datetime import datetime

from github import Github
from tqdm import tqdm

from src.config import settings
from src.analysis.core.dto import GithubIssueDTO, CommentDTO

logger = logging.getLogger(__name__)


class GitHubAPIClient:
    """GitHub API client implementation."""

    def __init__(self, token: str):
        """Initialize GitHub client.

        Args:
            token: GitHub API token
        """
        self.client = Github(token)

    def fetch_issues(self, repo_name: str, num_issues: int) -> List[GithubIssueDTO]:
        """Fetch issues from GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            num_issues: Number of issues to fetch

        Returns:
            List of GithubIssueDTO objects
        """
        try:
            logger.info(f"Fetching {num_issues} issues from {repo_name}")
            repo = self.client.get_repo(repo_name)

            issues = []
            total_fetched = 0
            skipped_prs = 0

            with tqdm(total=num_issues, desc="Fetching issues", unit="issue") as pbar:
                for issue in repo.get_issues(state='all'):
                    if total_fetched >= num_issues:
                        break

                    if not issue.pull_request:
                        issue_dto = self._process_issue(issue)
                        issues.append(issue_dto)
                        total_fetched += 1
                        pbar.update(1)
                    else:
                        skipped_prs += 1

            logger.info(
                f"Successfully fetched {total_fetched} issues (skipped {skipped_prs} pull requests)")
            return issues

        except Exception as e:
            logger.error(f"Error fetching issues: {e}")
            raise

    def get_repo_info(self, repo_name: str) -> dict:
        """Get repository information.

        Args:
            repo_name: Repository name (owner/repo)

        Returns:
            Repository information
        """
        try:
            repo = self.client.get_repo(repo_name)
            return {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description,
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'open_issues': repo.open_issues_count,
                'created_at': repo.created_at.isoformat(),
                'updated_at': repo.updated_at.isoformat(),
                'language': repo.language
            }
        except Exception as e:
            logger.error(f"Error fetching repo info: {e}")
            raise

    def _process_issue(self, issue) -> GithubIssueDTO:
        """Process a single GitHub issue.

        Args:
            issue: GitHub issue object

        Returns:
            GithubIssueDTO object
        """
        try:
            comments = []
            if issue.comments > 0:
                try:
                    comments = [
                        CommentDTO(
                            body=comment.body,
                            created_at=comment.created_at.isoformat(),
                            user=comment.user.login if comment.user else None
                        )
                        for comment in issue.get_comments()[:settings.MAX_COMMENTS_PER_ISSUE]
                    ]
                except Exception as e:
                    logger.warning(
                        f"Error fetching comments for issue #{issue.number}: {e}")

            return GithubIssueDTO(
                number=issue.number,
                title=issue.title,
                body=issue.body,
                state=issue.state,
                created_at=issue.created_at.isoformat(),
                url=issue.html_url,
                labels=[label.name for label in issue.labels],
                first_comments=comments,
                user=issue.user.login if issue.user else None,
                closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
                resolution_time=(
                    issue.closed_at - issue.created_at).total_seconds() / 3600 if issue.closed_at else None
            )
        except Exception as e:
            logger.error(f"Error processing issue {issue.number}: {e}")
            raise
