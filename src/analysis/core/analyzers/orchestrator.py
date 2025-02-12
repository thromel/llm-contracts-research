"""Analysis orchestrator implementation."""

import logging
from datetime import datetime
from typing import List, Optional

from src.config import settings
from src.analysis.core.interfaces import (
    IAnalyzer,
    IResultWriter,
    IAnalysisOrchestrator,
    IProgressTracker,
    ICheckpointManager
)
from src.analysis.core.clients.github import GitHubAPIClient
from src.analysis.core.dto import (
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO
)

logger = logging.getLogger(__name__)


class AnalysisOrchestrator(IAnalysisOrchestrator):
    """Orchestrates the analysis process."""

    def __init__(
        self,
        analyzer: IAnalyzer,
        github_client: GitHubAPIClient,
        storage: List[IResultWriter],
        checkpoint_manager: ICheckpointManager,
        progress_tracker: Optional[IProgressTracker] = None
    ):
        """Initialize orchestrator.

        Args:
            analyzer: Analysis implementation
            github_client: GitHub API client
            storage: List of storage implementations
            checkpoint_manager: Checkpoint manager
            progress_tracker: Optional progress tracker
        """
        self.analyzer = analyzer
        self.github_client = github_client
        self.storage = storage if isinstance(storage, list) else [storage]
        self.checkpoint_manager = checkpoint_manager
        self.progress_tracker = progress_tracker

    def run_analysis(
        self,
        repo_name: str,
        num_issues: int = 100,
        checkpoint_interval: int = 10
    ) -> AnalysisResultsDTO:
        """Run analysis on a GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            num_issues: Number of issues to analyze
            checkpoint_interval: Number of issues between checkpoints

        Returns:
            Analysis results
        """
        try:
            repo_info = self.github_client.get_repo_info(repo_name)
            logger.info(f"Analyzing repository: {repo_info['full_name']}")

            issues = self.github_client.fetch_issues(repo_name, num_issues)
            metadata = self._create_metadata(repo_info, len(issues))

            analyzed_issues = []
            for i, issue in enumerate(issues):
                try:
                    analysis = self._analyze_single_issue(issue, repo_info)
                    analyzed_issues.append(analysis)

                    if self.progress_tracker:
                        self.progress_tracker.update(
                            i + 1, len(issues), f"Analyzed issue {issue.number}")

                    self._handle_intermediate_save(
                        analyzed_issues, metadata, i)
                    self._handle_checkpoint(
                        analyzed_issues, metadata, i, checkpoint_interval)

                except Exception as e:
                    logger.error(f"Error analyzing issue {issue.number}: {e}")
                    continue

            # Save final results if we have analyzed any issues
            if analyzed_issues:
                if self.is_shutting_down:
                    logger.info("Saving final checkpoint before shutdown...")
                    self.checkpoint_manager.save_checkpoint(
                        analyzed_issues=analyzed_issues,
                        metadata=metadata
                    )
                else:
                    logger.info("Analysis complete. Saving final results...")
                    # Create final results DTO
                    results = AnalysisResultsDTO(
                        metadata=metadata, analyzed_issues=analyzed_issues
                    )
                    # Clear checkpoint since we completed successfully
                    self.checkpoint_manager.clear_checkpoint()

                    if self.progress_tracker:
                        self.progress_tracker.complete()

                    return results

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    def _analyze_single_issue(self, issue: dict, repo_info: dict) -> ContractAnalysisDTO:
        """Analyze a single issue.

        Args:
            issue: Issue data
            repo_info: Repository information

        Returns:
            Analysis results
        """
        analysis = self.analyzer.analyze_issue(
            title=issue.title,
            body=issue.body,
            comments=', '.join(
                comment.body for comment in issue.get_comments())
        )

        # Add issue metadata
        analysis.issue_url = issue.html_url
        analysis.issue_number = issue.number
        analysis.issue_title = issue.title
        analysis.repository_name = repo_info.get('name')
        analysis.repository_owner = repo_info.get('owner', {}).get('login')
        analysis.analysis_timestamp = datetime.now().isoformat()

        return analysis

    def _handle_intermediate_save(
        self,
        analyzed_issues: List[ContractAnalysisDTO],
        metadata: AnalysisMetadataDTO,
        current_index: int
    ) -> None:
        """Handle intermediate results saving.

        Args:
            analyzed_issues: List of analyzed issues
            metadata: Analysis metadata
            current_index: Current issue index
        """
        if not settings.SAVE_INTERMEDIATE:
            return

        try:
            from src.analysis.core.storage.factory import StorageFactory
            intermediate_metadata = self._create_intermediate_metadata(
                metadata, current_index + 1)
            intermediate_storage = StorageFactory.create_storage(
                storage_types=['json'],
                is_intermediate=True
            )

            # Ensure intermediate_storage is a list
            if not isinstance(intermediate_storage, list):
                intermediate_storage = [intermediate_storage]

            for storage in intermediate_storage:
                storage.save_results(
                    analyzed_issues=analyzed_issues[:current_index + 1],
                    metadata=intermediate_metadata
                )
            logger.info(
                f"Saved intermediate results after analyzing {current_index + 1} issues")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")

    def _create_metadata(self, repo_info: dict, num_issues: int) -> AnalysisMetadataDTO:
        """Create analysis metadata.

        Args:
            repo_info: Repository information
            num_issues: Number of issues analyzed

        Returns:
            Analysis metadata
        """
        return AnalysisMetadataDTO(
            repository=repo_info['full_name'],
            analysis_timestamp=datetime.now().isoformat(),
            num_issues=num_issues,
            repository_url=repo_info.get('html_url'),
            repository_owner=repo_info.get('owner', {}).get('login'),
            repository_name=repo_info.get('name'),
            repository_description=repo_info.get('description'),
            repository_stars=repo_info.get('stargazers_count'),
            repository_forks=repo_info.get('forks_count'),
            repository_language=repo_info.get('language'),
            analysis_version=settings.ANALYSIS_VERSION,
            analysis_model=settings.ANALYSIS_MODEL,
            analysis_batch_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def _create_intermediate_metadata(
        self,
        base_metadata: AnalysisMetadataDTO,
        current_count: int
    ) -> AnalysisMetadataDTO:
        """Create metadata for intermediate results.

        Args:
            base_metadata: Base metadata
            current_count: Current number of analyzed issues

        Returns:
            Intermediate metadata
        """
        return AnalysisMetadataDTO(
            **base_metadata.__dict__,
            num_issues=current_count,
            analysis_batch_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_intermediate_{current_count}"
        )

    def _handle_checkpoint(
        self,
        analyzed_issues: List[ContractAnalysisDTO],
        metadata: AnalysisMetadataDTO,
        current_index: int,
        checkpoint_interval: int
    ) -> None:
        """Handle checkpoint saving.

        Args:
            analyzed_issues: List of analyzed issues
            metadata: Analysis metadata
            current_index: Current issue index
            checkpoint_interval: Checkpoint interval
        """
        if (current_index + 1) % checkpoint_interval == 0:
            self.checkpoint_manager.save_checkpoint(
                analyzed_issues=analyzed_issues,
                metadata=metadata
            )
            logger.info(f"Saved checkpoint after {current_index + 1} issues")
