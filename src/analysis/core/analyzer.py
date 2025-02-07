"""Core analyzer for GitHub issues."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from github import Github
from openai import OpenAI
from tqdm import tqdm

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class IssueAnalyzer(ABC):
    """Interface for issue analyzers."""

    @abstractmethod
    def analyze_issue(self, title: str, body: str, comments: str = None) -> Dict[str, Any]:
        """Analyze a single issue.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Dict containing analysis results
        """
        pass

    @abstractmethod
    def save_results(self, results: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> None:
        """Save analysis results.

        Args:
            results: List of analysis results
            output_dir: Optional output directory
        """
        pass


class IssueFetcher(ABC):
    """Interface for issue fetchers."""

    @abstractmethod
    def fetch_issues(self, repo_name: str, num_issues: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues from a repository.

        Args:
            repo_name: Repository name in format owner/repo
            num_issues: Number of issues to fetch

        Returns:
            List of issue data dictionaries
        """
        pass


class GitHubIssuesAnalyzer(IssueAnalyzer, IssueFetcher):
    """Analyzer for GitHub issues using OpenAI API."""

    def __init__(self, repo_name: str = "openai/openai-python", model: str = None):
        """Initialize the analyzer.

        Args:
            repo_name: Repository name to analyze
            model: Optional model name to use
        """
        self.github_token = settings.GITHUB_TOKEN
        self.repo_name = repo_name

        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.model = model or settings.OPENAI_MODEL

        # Load context files
        self.context_dir = Path(settings.CONTEXT_DIR)
        self.violation_types = self._load_context_file('violation_types.yaml')
        self.severity_criteria = self._load_context_file(
            'severity_criteria.yaml')
        self.categorization = self._load_context_file('categorization.yaml')

    def _load_context_file(self, filename: str) -> Dict[str, Any]:
        """Load a context file from the context directory.

        Args:
            filename: Name of the context file

        Returns:
            Dict containing file contents
        """
        try:
            file_path = self.context_dir / filename
            if not file_path.exists():
                logger.warning("{} not found in context directory at {}".format(
                    filename, self.context_dir))
                return {}

            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as exc:
            logger.error("Error loading context file {}: {}".format(
                filename, str(exc)))
            return {}

    def _build_system_prompt(self) -> str:
        """Build the system prompt for analysis."""
        return (
            "You are an expert in API contract violations. Use the following taxonomy:\n"
            "Violation Types: input_type_violation, input_value_violation, missing_dependency_violation, "
            "missing_option_violation, method_order_violation, memory_out_of_bound, performance_degradation, "
            "incorrect_functionality, hang.\n"
            "Severity: high, medium, low.\n"
            "Guidelines: Analyze the issue description and comments to decide if there is a contract violation. "
            "Respond in JSON with fields: has_violation, violation_type, severity, description, confidence, "
            "resolution_status, resolution_details.\n"
            "Do not add extra text."
        )

    def _build_user_prompt(self, title: str, body: str, comments: str = None) -> str:
        """Build the user prompt for analysis.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Formatted prompt string
        """
        prompt = "Issue Title: {}\n\nIssue Body: {}\n\n".format(title, body)
        if comments:
            prompt += "Issue Comments: {}\n\n".format(comments)
        prompt += (
            "Analyze if this issue shows an API contract violation. Provide your answer as JSON with keys: "
            "has_violation (bool), violation_type (string or null), severity (high/medium/low or null), "
            "description (string), confidence (high/medium/low), resolution_status (string), "
            "resolution_details (string)."
        )
        return prompt

    def analyze_issue(self, title: str, body: str, comments: str = None) -> Dict[str, Any]:
        """Analyze a single issue using the OpenAI API.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Dict containing analysis results
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(title, body, comments)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                top_p=settings.OPENAI_TOP_P,
                frequency_penalty=settings.OPENAI_FREQUENCY_PENALTY,
                presence_penalty=settings.OPENAI_PRESENCE_PENALTY,
                response_format={"type": "json_object"}
            )
            response_content = completion.choices[0].message.content.strip()

            try:
                return yaml.safe_load(response_content) or {
                    "has_violation": False,
                    "error": "Invalid response format",
                    "confidence": "low"
                }
            except yaml.YAMLError as exc:
                logger.error(
                    "Error parsing response content: {}".format(str(exc)))
                logger.error(
                    "Problematic content: {}".format(response_content))
                return {
                    "has_violation": False,
                    "error": "Response parsing error",
                    "confidence": "low",
                    "description": "Failed to parse API response"
                }
        except Exception as exc:
            logger.error("Error calling OpenAI API: {}".format(str(exc)))
            return {
                "has_violation": False,
                "error": str(exc),
                "confidence": "low",
                "description": "API call failed"
            }

    def fetch_issues(self, repo_name: str, num_issues: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues from a GitHub repository.

        Args:
            repo_name: Repository name in format owner/repo
            num_issues: Number of issues to fetch

        Returns:
            List of issue data dictionaries
        """
        logger.info("Fetching {} issues from {}".format(num_issues, repo_name))

        try:
            github_client = Github(self.github_token)
            repo = github_client.get_repo(repo_name)

            issues = []
            total_fetched = 0
            skipped_prs = 0

            with tqdm(total=num_issues, desc="Fetching issues", unit="issue") as pbar:
                for issue in repo.get_issues(state='closed'):
                    if total_fetched >= num_issues:
                        break

                    if not issue.pull_request:  # Skip pull requests
                        issue_data = {
                            'number': issue.number,
                            'title': issue.title,
                            'body': issue.body,
                            'state': issue.state,
                            'created_at': issue.created_at.isoformat(),
                            'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                            'labels': [label.name for label in issue.labels],
                            'url': issue.html_url,
                            'resolution_time': (issue.closed_at - issue.created_at).total_seconds() / 3600 if issue.closed_at else None
                        }
                        issues.append(issue_data)
                        total_fetched += 1
                        pbar.update(1)
                    else:
                        skipped_prs += 1

            logger.info("Successfully fetched {} issues (skipped {} pull requests)".format(
                total_fetched, skipped_prs))
            return issues

        except Exception as exc:
            logger.error("Error fetching issues: {}".format(str(exc)))
            raise

    def save_results(self, results: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> None:
        """Save analysis results to files.

        Args:
            results: List of analysis results
            output_dir: Optional output directory
        """
        if output_dir is None:
            output_dir = Path(settings.DATA_DIR) / 'analyzed'

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = 'github_issues_analysis_{}'.format(timestamp)

        # Save raw data
        raw_csv_path = output_dir / '{}_raw.csv'.format(base_name)
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(raw_csv_path, index=False)
            logger.info("Saved raw data to {}".format(raw_csv_path))
        except Exception as exc:
            logger.error("Error saving raw data: {}".format(str(exc)))

        # Save final results
        final_csv_path = output_dir / '{}_final.csv'.format(base_name)
        try:
            df = pd.DataFrame(results)
            df.to_csv(final_csv_path, index=False)
            logger.info("Saved CSV results to {}".format(final_csv_path))
        except Exception as exc:
            logger.error("Error saving CSV results: {}".format(str(exc)))
