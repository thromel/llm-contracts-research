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
from src.analysis.core.processors.cleaner import ResponseCleaner
from src.analysis.core.processors.validator import ResponseValidator
from src.analysis.core.storage.base import ResultsStorage
from src.analysis.core.dto import (
    GithubIssueDTO,
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO,
    dict_to_contract_analysis_dto,
    dict_to_github_issue_dto
)

logger = logging.getLogger(__name__)


class ContractAnalyzer:
    """Core contract violation analyzer."""

    def __init__(
        self,
        llm_client: LLMClient,
        response_cleaner: ResponseCleaner,
        response_validator: ResponseValidator,
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
            [issue.__dict__ for issue in results.analyzed_issues],
            results.metadata.__dict__
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
                "origin_stage": "analysis",
                "affected_stages": [],
                "propagation_path": "Analysis error contained"
            }
        }
        return dict_to_contract_analysis_dto(error_dict)


class GitHubIssuesAnalyzer:
    """GitHub issues analyzer implementation."""

    def __init__(self, repo_name: Optional[str] = None):
        """Initialize GitHub issues analyzer.

        Args:
            repo_name: Optional repository name
        """
        self.repo_name = repo_name

        # Initialize components
        openai_settings = {
            'api_key': settings.OPENAI_API_KEY,
            'model': settings.OPENAI_MODEL,
            'max_retries': settings.MAX_RETRIES,
            'timeout': 30.0,
            'temperature': settings.OPENAI_TEMPERATURE,
            'max_tokens': settings.OPENAI_MAX_TOKENS,
            'top_p': settings.OPENAI_TOP_P,
            'frequency_penalty': settings.OPENAI_FREQUENCY_PENALTY,
            'presence_penalty': settings.OPENAI_PRESENCE_PENALTY
        }

        # Add base URL if specified
        if hasattr(settings, 'OPENAI_BASE_URL'):
            openai_settings['base_url'] = settings.OPENAI_BASE_URL

        from src.analysis.core.clients.openai import OpenAIClient
        from src.analysis.core.processors.cleaner import MarkdownResponseCleaner
        from src.analysis.core.processors.validator import ContractAnalysisValidator
        from src.analysis.core.storage.json_storage import JSONResultsStorage

        llm_client = OpenAIClient(**openai_settings)
        response_cleaner = MarkdownResponseCleaner()
        response_validator = ContractAnalysisValidator()
        results_storage = JSONResultsStorage(
            Path(settings.DATA_DIR) / 'analysis')

        self.analyzer = ContractAnalyzer(
            llm_client=llm_client,
            response_cleaner=response_cleaner,
            response_validator=response_validator,
            results_storage=results_storage
        )

        self.github_client = Github(settings.GITHUB_TOKEN)

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> ContractAnalysisDTO:
        """Analyze a single issue."""
        return self.analyzer.analyze_issue(title, body, comments)

    def save_results(self, analyzed_issues: List[ContractAnalysisDTO]) -> None:
        """Save analysis results."""
        metadata = AnalysisMetadataDTO(
            repository=self.repo_name,
            analysis_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            num_issues=len(analyzed_issues)
        )
        self.analyzer.save_results(analyzed_issues, metadata)

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
            repo = self.github_client.get_repo(repo_name)

            issues = []
            total_fetched = 0
            skipped_prs = 0

            with tqdm(total=num_issues, desc="Fetching issues", unit="issue") as pbar:
                for issue in repo.get_issues(state='all'):
                    if total_fetched >= num_issues:
                        break

                    if not issue.pull_request:
                        issue_data = self._process_issue(issue)
                        issues.append(dict_to_github_issue_dto(issue_data))
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

    def _process_issue(self, issue) -> Dict[str, Any]:
        """Process a single GitHub issue.

        Args:
            issue: GitHub issue object

        Returns:
            Issue data dictionary
        """
        comments = []
        if issue.comments > 0:
            try:
                comments = [
                    {
                        'body': comment.body,
                        'created_at': comment.created_at.isoformat(),
                        'user': comment.user.login if comment.user else None
                    }
                    for comment in issue.get_comments()[:settings.MAX_COMMENTS_PER_ISSUE]
                ]
            except Exception as e:
                logger.warning(
                    f"Error fetching comments for issue #{issue.number}: {e}")

        return {
            'number': issue.number,
            'title': issue.title,
            'body': issue.body,
            'state': issue.state,
            'created_at': issue.created_at.isoformat(),
            'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
            'labels': [label.name for label in issue.labels],
            'url': issue.html_url,
            'user': issue.user.login if issue.user else None,
            'first_comments': comments,
            'resolution_time': (issue.closed_at - issue.created_at).total_seconds() / 3600 if issue.closed_at else None
        }
