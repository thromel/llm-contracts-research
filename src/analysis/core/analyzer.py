"""Core analyzer for GitHub issues."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from github import Github
from tqdm import tqdm

from src.config import settings
from src.utils.logger import setup_logger
from src.analysis.core.prompts import get_system_prompt, get_user_prompt
from src.analysis.core.clients.base import LLMClient
from src.analysis.core.processors.cleaner import MarkdownResponseCleaner
from src.analysis.core.processors.validator import ContractAnalysisValidator
from src.analysis.core.storage.base import ResultsStorage
from src.analysis.core.dto import (
    GithubIssueDTO,
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO,
    dict_to_contract_analysis_dto,
    dict_to_github_issue_dto
)
from src.analysis.core.clients.github import GitHubAPIClient
from src.analysis.core.processors.checkpoint import CheckpointHandler

logger = logging.getLogger(__name__)


class ContractAnalyzer:
    """Core contract violation analyzer."""

    def __init__(
        self,
        llm_client: LLMClient,
        response_cleaner: MarkdownResponseCleaner,
        response_validator: ContractAnalysisValidator,
        results_storage: ResultsStorage
    ):
        """Initialize analyzer with components.

        Args:
            llm_client: LLM API client
            response_cleaner: Response cleaning strategy
            response_validator: Response validation strategy
            results_storage: Results storage strategy
        """
        self.llm_client = llm_client
        self.response_cleaner = response_cleaner
        self.response_validator = response_validator
        self.results_storage = results_storage
        self.system_prompt = get_system_prompt()

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> ContractAnalysisDTO:
        """Analyze a GitHub issue for contract violations.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Analysis results as ContractAnalysisDTO
        """
        try:
            # Format comments if present
            formatted_comments = None
            if comments:
                if isinstance(comments, list):
                    # Format list of comment dictionaries
                    formatted_comments = "\n\n".join([
                        f"Comment by {c.get('user', 'unknown')} at {c.get('created_at', 'unknown')}:\n{c.get('body', '')}"
                        for c in comments
                    ])
                else:
                    # Use comments string as is
                    formatted_comments = comments

            # Generate and get analysis
            user_prompt = get_user_prompt(title, body, formatted_comments)
            content = self.llm_client.get_analysis(
                self.system_prompt, user_prompt)

            if not content:
                raise ValueError("Empty response from LLM")

            # Clean and parse response
            cleaned_content = self.response_cleaner.clean(content)
            try:
                analysis_dict = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse response: {e}\nCleaned content: {cleaned_content}")
                return self._get_error_analysis("Failed to parse analysis response")

            # Validate and convert to DTO
            try:
                self.response_validator.validate(analysis_dict)
                return dict_to_contract_analysis_dto(analysis_dict)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Invalid analysis structure: {e}")
                return self._get_error_analysis("Invalid analysis structure")

        except Exception as e:
            logger.error(f"Error analyzing issue: {e}")
            return self._get_error_analysis(str(e))

    def save_results(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis results.

        Args:
            analyzed_issues: List of analysis results
            metadata: Analysis metadata
        """
        results = AnalysisResultsDTO(
            metadata=metadata, analyzed_issues=analyzed_issues)
        self.results_storage.save_results(
            analyzed_issues=analyzed_issues,
            metadata=metadata
        )

    def _get_error_analysis(self, error_msg: str) -> ContractAnalysisDTO:
        """Generate error analysis result.

        Args:
            error_msg: Error message

        Returns:
            Error analysis as ContractAnalysisDTO
        """
        error_dict = {
            "has_violation": False,
            "violation_type": "ERROR",
            "severity": "low",
            "description": f"Analysis failed: {error_msg}",
            "confidence": "low",
            "root_cause": "Analysis error",
            "effects": ["Unable to determine contract violations"],
            "resolution_status": "ERROR",
            "resolution_details": "Please try analyzing the issue again",
            "contract_category": "unknown",
            "comment_analysis": {
                "supporting_evidence": [],
                "frequency": "unknown",
                "workarounds": [],
                "impact": "unknown"
            },
            "error_propagation": {
                "affected_stages": [],
                "propagation_path": "Analysis error contained"
            }
        }
        return dict_to_contract_analysis_dto(error_dict)


class GitHubIssuesAnalyzer:
    """Analyzer for GitHub issues."""

    def __init__(
        self,
        llm_client: LLMClient,
        github_token: str,
        storage: ResultsStorage,
        checkpoint_handler: Optional[CheckpointHandler] = None
    ):
        """Initialize analyzer.

        Args:
            llm_client: LLM client for analysis
            github_token: GitHub API token
            storage: Results storage
            checkpoint_handler: Optional checkpoint handler
        """
        self.llm_client = llm_client
        self.github_client = GitHubAPIClient(github_token)
        self.storage = storage
        self.checkpoint_handler = checkpoint_handler or CheckpointHandler()

        # Initialize contract analyzer with concrete implementations
        self.contract_analyzer = ContractAnalyzer(
            llm_client=llm_client,
            response_cleaner=MarkdownResponseCleaner(),
            response_validator=ContractAnalysisValidator(),
            results_storage=storage
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
        try:
            # Get repository info
            repo_info = self.github_client.get_repo_info(repo_name)
            logger.info(f"Analyzing repository: {repo_info['full_name']}")

            # Fetch issues
            issues = self.github_client.fetch_issues(repo_name, num_issues)

            # Create metadata
            metadata = AnalysisMetadataDTO(
                repository=repo_info['full_name'],
                analysis_timestamp=datetime.now().isoformat(),
                num_issues=len(issues),
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

            # Analyze issues
            analyzed_issues = []
            for i, issue in enumerate(issues):
                try:
                    # Update issue metadata
                    analysis = self.contract_analyzer.analyze_issue(
                        title=issue.title,
                        body=issue.body,
                        comments=', '.join(
                            comment.body for comment in issue.first_comments)
                    )
                    # Add issue metadata
                    analysis.issue_url = issue.html_url
                    analysis.issue_number = issue.number
                    analysis.issue_title = issue.title
                    analysis.repository_name = repo_info.get('name')
                    analysis.repository_owner = repo_info.get(
                        'owner', {}).get('login')
                    analysis.analysis_timestamp = datetime.now().isoformat()

                    analyzed_issues.append(analysis)

                    # Save checkpoint if needed
                    if (i + 1) % checkpoint_interval == 0:
                        self.checkpoint_handler.save_checkpoint(
                            analyzed_issues=analyzed_issues,
                            metadata=metadata
                        )
                        logger.info(f"Saved checkpoint after {i + 1} issues")

                except Exception as e:
                    logger.error(f"Error analyzing issue {issue.number}: {e}")
                    continue

            # Save final results
            results = AnalysisResultsDTO(
                metadata=metadata,
                analyzed_issues=analyzed_issues
            )
            self.contract_analyzer.save_results(
                analyzed_issues=analyzed_issues, metadata=metadata)
            logger.info("Analysis complete")

            return results

        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
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
