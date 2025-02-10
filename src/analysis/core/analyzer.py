"""Core analyzer for GitHub issues."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from github import Github
from openai import OpenAI
from tqdm import tqdm
import json
import logging

from src.config import settings
from src.utils.logger import setup_logger
from src.analysis.core.prompts import get_system_prompt, get_user_prompt

logger = logging.getLogger(__name__)


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

    def __init__(self, repo_name: Optional[str] = None):
        """Initialize the analyzer.

        Args:
            repo_name: Optional repository name for context
        """
        self.repo_name = repo_name
        self.system_prompt = get_system_prompt()
        self.results_dir = Path(settings.DATA_DIR) / 'analysis'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.github_client = Github(settings.GITHUB_TOKEN)

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a GitHub issue for contract violations using the enhanced taxonomy.

        Args:
            title: Issue title
            body: Issue body text
            comments: Optional concatenated comments text

        Returns:
            Dict containing the analysis results in the specified JSON format
        """
        try:
            # Generate user prompt with issue content
            user_prompt = get_user_prompt(title, body, comments)

            # Get analysis from OpenAI
            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000
            )

            # Parse and validate the response
            try:
                analysis = json.loads(response.choices[0].message.content)
                self._validate_analysis(analysis)
                return analysis
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse analysis response: {e}")
                return self._get_error_analysis("Failed to parse analysis response")

        except Exception as e:
            logger.error(f"Error analyzing issue: {e}")
            return self._get_error_analysis(str(e))

    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validate the analysis contains all required fields.

        Args:
            analysis: Analysis dict to validate

        Raises:
            KeyError if required fields are missing
        """
        required_fields = [
            "has_violation",
            "violation_type",
            "severity",
            "description",
            "confidence",
            "root_cause",
            "effects",
            "resolution_status",
            "resolution_details",
            "pipeline_stage",
            "contract_category"
        ]

        missing = [field for field in required_fields if field not in analysis]
        if missing:
            raise KeyError(f"Missing required fields in analysis: {missing}")

    def _get_error_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Generate an error analysis result.

        Args:
            error_msg: Error message to include

        Returns:
            Dict with error information in the analysis format
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
            "pipeline_stage": "analysis",
            "contract_category": "unknown"
        }

    def save_results(self, analyzed_issues: List[Dict[str, Any]]) -> None:
        """Save the analysis results to a JSON file.

        Args:
            analyzed_issues: List of analyzed issues with their results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / \
            f"github_issues_analysis_{timestamp}_final_violation_analysis.json"

        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "repository": self.repo_name,
                    "analysis_timestamp": timestamp,
                    "num_issues": len(analyzed_issues)
                },
                "analyzed_issues": analyzed_issues
            }, f, indent=2)

        logger.info(f"Saved analysis results to {output_file}")

    def fetch_issues(self, repo_name: str, num_issues: int) -> List[Dict[str, Any]]:
        """Fetch issues from GitHub repository.

        Args:
            repo_name: Repository name in owner/repo format
            num_issues: Number of issues to fetch

        Returns:
            List of issue dictionaries
        """
        try:
            logger.info(f"Fetching {num_issues} issues from {repo_name}")

            # Get repository
            repo = self.github_client.get_repo(repo_name)

            issues = []
            total_fetched = 0
            skipped_prs = 0

            # Fetch issues with progress bar
            with tqdm(total=num_issues, desc="Fetching issues", unit="issue") as pbar:
                for issue in repo.get_issues(state='all'):
                    if total_fetched >= num_issues:
                        break

                    # Skip pull requests
                    if not issue.pull_request:
                        # Get comments
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

                        # Format issue data
                        issue_data = {
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
