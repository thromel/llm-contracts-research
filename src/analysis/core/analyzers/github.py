"""GitHub issues analyzer implementation."""

import logging
from typing import List, Optional

from src.analysis.core.interfaces import (
    IAnalyzer,
    IResultWriter,
    ICheckpointManager,
    IProgressTracker
)
from src.analysis.core.clients.base import LLMClient
from src.analysis.core.clients.github import GitHubAPIClient
from src.analysis.core.processors.cleaner import MarkdownResponseCleaner
from src.analysis.core.processors.validator import ContractAnalysisValidator
from src.analysis.core.processors.checkpoint import CheckpointHandler
from src.analysis.core.dto import ContractAnalysisDTO, AnalysisResultsDTO
from src.analysis.core.storage.factory import StorageFactory

# Import analyzer classes
from . import ContractAnalyzer, AnalysisOrchestrator, DefaultProgressTracker

logger = logging.getLogger(__name__)


class GitHubIssuesAnalyzer:
    """Analyzer for GitHub issues."""

    def __init__(
        self,
        llm_client: LLMClient,
        github_token: str,
        checkpoint_handler: Optional[ICheckpointManager] = None,
        progress_tracker: Optional[IProgressTracker] = None
    ):
        """Initialize analyzer.

        Args:
            llm_client: LLM client for analysis
            github_token: GitHub API token
            checkpoint_handler: Optional checkpoint handler
            progress_tracker: Optional progress tracker
        """
        # Create core components
        self.github_client = GitHubAPIClient(github_token)

        # Create analyzer
        self.contract_analyzer = ContractAnalyzer(
            llm_client=llm_client,
            response_cleaner=MarkdownResponseCleaner(),
            response_validator=ContractAnalysisValidator()
        )

        # Initialize storage
        storage = StorageFactory.create_storage()

        # Create orchestrator
        self.orchestrator = AnalysisOrchestrator(
            analyzer=self.contract_analyzer,
            github_client=self.github_client,
            storage=storage,
            checkpoint_manager=checkpoint_handler or CheckpointHandler(),
            progress_tracker=progress_tracker or DefaultProgressTracker()
        )

    def analyze_repository(
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
        return self.orchestrator.run_analysis(
            repo_name=repo_name,
            num_issues=num_issues,
            checkpoint_interval=checkpoint_interval
        )

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
