"""GitHub issues analyzer implementation."""

import logging
from typing import List, Optional

from ..interfaces import (
    IAnalyzer,
    IResultWriter,
    ICheckpointManager,
    IProgressTracker
)
from ..clients.base import LLMClient
from ..clients.github import GitHubAPIClient
from ..processors.cleaner import MarkdownResponseCleaner
from ..processors.validator import ContractAnalysisValidator
from ..processors.checkpoint import CheckpointHandler
from ..dto import ContractAnalysisDTO, AnalysisResultsDTO
from ..storage.factory import StorageFactory
from core.repositories import BaseRepository, RepositoryFactory
from core.config import AppConfig

# Import analyzer classes
from . import ContractAnalyzer, AnalysisOrchestrator, DefaultProgressTracker

logger = logging.getLogger(__name__)


class GitHubIssuesAnalyzer:
    """Analyzer for GitHub issues."""

    def __init__(
        self,
        llm_client: LLMClient,
        github_token: str,
        repository: BaseRepository,
        checkpoint_handler: Optional[ICheckpointManager] = None,
        progress_tracker: Optional[IProgressTracker] = None
    ):
        """Initialize analyzer.

        Args:
            llm_client: LLM client for analysis
            github_token: GitHub API token
            repository: Repository instance for data storage
            checkpoint_handler: Optional checkpoint handler
            progress_tracker: Optional progress tracker
        """
        # Create core components
        self.github_client = GitHubAPIClient(github_token)
        self.repository = repository

        # Create analyzer
        self.contract_analyzer = ContractAnalyzer(
            llm_client=llm_client,
            response_cleaner=MarkdownResponseCleaner(),
            response_validator=ContractAnalysisValidator()
        )

        # Initialize checkpoint and progress tracking
        self.checkpoint_manager = checkpoint_handler or CheckpointHandler()
        self.progress_tracker = progress_tracker or DefaultProgressTracker()

    async def analyze_repository(
        self,
        repo_name: str,
        num_issues: int = 100,
        checkpoint_interval: int = 10
    ) -> AnalysisResultsDTO:
        """Analyze issues from a GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            num_issues: Number of issues to analyze
            checkpoint_interval: Number of issues between checkpoints

        Returns:
            Analysis results
        """
        try:
            # Get repository from database
            repo = await self.repository.get_repository_by_full_name(repo_name)
            if not repo:
                raise ValueError(f"Repository not found: {repo_name}")

            # Get issues to analyze
            issues = await self.repository.get_repository_issues(
                repo.id,
                limit=num_issues
            )

            # Initialize progress tracking
            self.progress_tracker.start_analysis(
                total_issues=len(issues),
                repository_name=repo_name
            )

            analyzed_issues = []
            for i, issue in enumerate(issues):
                try:
                    # Analyze issue
                    analysis = await self.contract_analyzer.analyze_issue(issue)

                    # Save analysis results
                    analysis_id = await self.repository.save_analysis(analysis)
                    analyzed_issues.append(analysis)

                    # Update progress
                    self.progress_tracker.update_progress(1)

                    # Save checkpoint if needed
                    if i > 0 and i % checkpoint_interval == 0:
                        self.checkpoint_manager.save_checkpoint(
                            analyzed_issues=analyzed_issues,
                            current_index=i,
                            total_issues=issues,
                            repo_name=repo_name
                        )

                except Exception as e:
                    logger.error(
                        f"Error analyzing issue {issue.number}: {str(e)}")
                    continue

            # Create final results
            results = AnalysisResultsDTO(
                repository=repo,
                analyzed_issues=analyzed_issues,
                total_issues=len(issues),
                completed_issues=len(analyzed_issues)
            )

            return results

        except Exception as e:
            logger.error(f"Error analyzing repository {repo_name}: {str(e)}")
            raise

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> ContractAnalysisDTO:
        """Analyze a GitHub issue for contract violations.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Analysis results as ContractAnalysisDTO
        """
        return self.contract_analyzer.analyze_issue(title, body, comments)

    def _get_error_analysis(self, error_msg: str) -> ContractAnalysisDTO:
        """Generate error analysis result.

        Args:
            error_msg: Error message

        Returns:
            Error analysis as ContractAnalysisDTO
        """
        return self.contract_analyzer._get_error_analysis(error_msg)

    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        github_token: str,
        checkpoint_handler: Optional[ICheckpointManager] = None,
        progress_tracker: Optional[IProgressTracker] = None
    ) -> 'GitHubIssuesAnalyzer':
        """Create analyzer instance with default repository.

        Args:
            llm_client: LLM client for analysis
            github_token: GitHub API token
            checkpoint_handler: Optional checkpoint handler
            progress_tracker: Optional progress tracker

        Returns:
            Analyzer instance
        """
        config = AppConfig()
        repository = RepositoryFactory.create_repository(config)
        return cls(
            llm_client=llm_client,
            github_token=github_token,
            repository=repository,
            checkpoint_handler=checkpoint_handler,
            progress_tracker=progress_tracker
        )
