"""Core analyzer for GitHub issues."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
import time
import backoff

import yaml
from github import Github
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm
import json
import logging

from src.config import settings
from src.utils.logger import setup_logger
from src.analysis.core.prompts import get_system_prompt, get_user_prompt

logger = logging.getLogger(__name__)


class ResponseCleaner(Protocol):
    """Protocol for response cleaning strategies."""

    def clean(self, content: str) -> str:
        """Clean the response content."""
        pass


class MarkdownResponseCleaner:
    """Cleans markdown-formatted responses."""

    def clean(self, content: str) -> str:
        """Remove markdown formatting from response.

        Args:
            content: Raw response content

        Returns:
            Cleaned content
        """
        if content.startswith('```json\n'):
            content = content.replace('```json\n', '', 1)
            if content.endswith('\n```'):
                content = content[:-4]
        elif content.startswith('```\n'):
            content = content.replace('```\n', '', 1)
            if content.endswith('\n```'):
                content = content[:-4]
        return content.strip()


class ResponseValidator(Protocol):
    """Protocol for response validation strategies."""

    def validate(self, analysis: Dict[str, Any]) -> None:
        """Validate the analysis response."""
        pass


class ContractAnalysisValidator:
    """Validates contract analysis responses."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(self, analysis: Dict[str, Any]) -> None:
        """Validate contract analysis response.

        Args:
            analysis: Analysis dict to validate

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
            TypeError: If field types are incorrect
        """
        self._validate_required_fields(analysis)
        self._validate_field_types(analysis)
        self._validate_field_values(analysis)
        self._validate_optional_fields(analysis)

    def _validate_required_fields(self, analysis: Dict[str, Any]) -> None:
        """Validate presence of required fields."""
        required_fields = [
            "has_violation", "violation_type", "severity", "description",
            "confidence", "root_cause", "effects", "resolution_status",
            "resolution_details", "contract_category"
        ]
        missing = [field for field in required_fields if field not in analysis]
        if missing:
            raise KeyError(f"Missing required fields in analysis: {missing}")

    def _validate_field_types(self, analysis: Dict[str, Any]) -> None:
        """Validate field types."""
        if not isinstance(analysis["has_violation"], bool):
            raise TypeError("has_violation must be a boolean")
        if not isinstance(analysis["effects"], list):
            raise TypeError("effects must be an array")

    def _validate_field_values(self, analysis: Dict[str, Any]) -> None:
        """Validate field values."""
        if analysis["severity"] not in ["high", "medium", "low"]:
            raise ValueError("severity must be one of: high, medium, low")
        if analysis["confidence"] not in ["high", "medium", "low"]:
            raise ValueError("confidence must be one of: high, medium, low")

    def _validate_optional_fields(self, analysis: Dict[str, Any]) -> None:
        """Validate optional fields if present."""
        if "error_propagation" in analysis:
            self._validate_error_propagation(analysis["error_propagation"])
        if "suggested_new_contracts" in analysis:
            self._validate_suggested_contracts(
                analysis["suggested_new_contracts"])

    def _validate_error_propagation(self, error_prop: Dict[str, Any]) -> None:
        """Validate error propagation fields."""
        required_fields = ["origin_stage",
                           "affected_stages", "propagation_path"]
        missing = [field for field in required_fields if field not in error_prop]
        if missing:
            raise KeyError(
                f"Missing required fields in error_propagation: {missing}")

    def _validate_suggested_contracts(self, contracts: List[Dict[str, Any]]) -> None:
        """Validate suggested contracts."""
        if not isinstance(contracts, list):
            raise TypeError("suggested_new_contracts must be an array")

        required_fields = [
            "name", "description", "rationale", "examples",
            "parent_category", "pattern_frequency"
        ]
        for contract in contracts:
            missing = [
                field for field in required_fields if field not in contract]
            if missing:
                raise KeyError(
                    f"Missing required fields in suggested contract: {missing}")


class LLMClient(Protocol):
    """Protocol for LLM API clients."""

    def get_analysis(self, system_prompt: str, user_prompt: str) -> str:
        """Get analysis from LLM."""
        pass


class OpenAIClient:
    """OpenAI API client implementation."""

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use
            **kwargs: Additional settings for completions
        """
        # Extract OpenAI client kwargs
        client_kwargs = {
            'api_key': api_key,
            'max_retries': kwargs.pop('max_retries', 3),
            'timeout': kwargs.pop('timeout', 30.0)
        }
        if 'base_url' in kwargs:
            client_kwargs['base_url'] = kwargs.pop('base_url')

        # Initialize client
        self.client = OpenAI(**client_kwargs)

        # Store model and completion settings
        self.model = model
        self.completion_settings = {
            'temperature': kwargs.get('temperature', 0.1),
            'max_tokens': kwargs.get('max_tokens', 1000),
            'top_p': kwargs.get('top_p', 1.0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0)
        }

        # Log configuration
        logger.info(f"Initialized OpenAI client with model: {model}")
        if 'base_url' in client_kwargs:
            logger.info(f"Using custom base URL: {client_kwargs['base_url']}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=settings.MAX_RETRIES,
        giveup=lambda e: not self._should_retry(e)
    )
    def get_analysis(self, system_prompt: str, user_prompt: str) -> str:
        """Get analysis from OpenAI.

        Args:
            system_prompt: System context prompt
            user_prompt: User query prompt

        Returns:
            Analysis response content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.completion_settings
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise

    def _should_retry(self, e: Exception) -> bool:
        """Determine if error should trigger retry."""
        if hasattr(e, 'status_code'):
            if e.status_code == 429:  # Rate limit
                logger.warning("Rate limit hit, backing off...")
                return True
            elif e.status_code >= 500:  # Server error
                logger.warning(
                    f"OpenAI server error {e.status_code}, retrying...")
                return True
        return False


class ResultsStorage(Protocol):
    """Protocol for analysis results storage."""

    def save_results(self, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """Save analysis results."""
        pass


class JSONResultsStorage:
    """JSON file storage for analysis results."""

    def __init__(self, output_dir: Path):
        """Initialize storage with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """Save results to JSON file.

        Args:
            results: List of analysis results
            metadata: Analysis metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / \
            f"github_issues_analysis_{timestamp}_final_violation_analysis.json"

        with open(output_file, 'w') as f:
            json.dump({
                "metadata": metadata,
                "analyzed_issues": results
            }, f, indent=2)

        logger.info(f"Saved analysis results to {output_file}")


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

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a GitHub issue for contract violations.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Analysis results dict
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
                analysis = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse response: {e}\nCleaned content: {cleaned_content}")
                return self._get_error_analysis("Failed to parse analysis response")

            # Validate analysis
            try:
                self.response_validator.validate(analysis)
                return analysis
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Invalid analysis structure: {e}")
                return self._get_error_analysis("Invalid analysis structure")

        except Exception as e:
            logger.error(f"Error analyzing issue: {e}")
            return self._get_error_analysis(str(e))

    def save_results(self, analyzed_issues: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """Save analysis results.

        Args:
            analyzed_issues: List of analysis results
            metadata: Analysis metadata
        """
        self.results_storage.save_results(analyzed_issues, metadata)

    def _get_error_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Generate error analysis result.

        Args:
            error_msg: Error message

        Returns:
            Error analysis dict
        """
        return {
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
            "error_propagation": {
                "origin_stage": "analysis",
                "affected_stages": [],
                "propagation_path": "Analysis error contained"
            }
        }


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

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single issue."""
        return self.analyzer.analyze_issue(title, body, comments)

    def save_results(self, analyzed_issues: List[Dict[str, Any]]) -> None:
        """Save analysis results."""
        metadata = {
            "repository": self.repo_name,
            "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_issues": len(analyzed_issues)
        }
        self.analyzer.save_results(analyzed_issues, metadata)

    def fetch_issues(self, repo_name: str, num_issues: int) -> List[Dict[str, Any]]:
        """Fetch issues from GitHub repository.

        Args:
            repo_name: Repository name (owner/repo)
            num_issues: Number of issues to fetch

        Returns:
            List of issue dictionaries
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
                        issues.append(issue_data)
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
            Processed issue dictionary
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
