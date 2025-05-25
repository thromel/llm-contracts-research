"""GitHub issues analyzer for contract violations."""
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime

from ..clients.openai import OpenAIClient
from ..clients.github import GitHubAPIClient
from ..processors.checkpoint import CheckpointHandler
from ..dto import IssueAnalysisDTO

logger = logging.getLogger(__name__)


class GitHubIssuesAnalyzer:
    """Analyzes GitHub issues for contract violations."""

    def __init__(self, llm_client: OpenAIClient, github_token: str, checkpoint_handler: Optional[CheckpointHandler] = None):
        """Initialize the analyzer.

        Args:
            llm_client: LLM client for analysis
            github_token: GitHub API token
            checkpoint_handler: Optional checkpoint handler
        """
        self.llm_client = llm_client
        self.github_token = github_token
        self.checkpoint_handler = checkpoint_handler
        self.github_client = GitHubAPIClient(github_token)

    def _prepare_prompt(self, title: str, body: str, comments: str) -> str:
        """Prepare the prompt for the LLM.

        Args:
            title: Issue title
            body: Issue body
            comments: Issue comments

        Returns:
            Formatted prompt
        """
        return f"""Analyze the following GitHub issue for potential API contract violations. 
        
Issue Title: {title}

Issue Description:
{body}

Issue Comments:
{comments}

Your task is to determine if this issue represents an API contract violation. A contract violation 
occurs when a software component or API breaks its promised behavior, interface, or compatibility.

Please analyze and provide the following information in JSON format:
1. Does this issue represent an API contract violation? (true/false)
2. If yes:
   - What type of violation is it? (e.g., breaking change, undocumented behavior, backward incompatibility)
   - How severe is the violation? (high/medium/low)
   - Your confidence in this assessment (high/medium/low)
   - A brief description of the violation
   - Was the issue resolved? If so, how?

Example response format:
```json
{{
  "has_violation": true,
  "violation_type": "breaking change",
  "severity": "high",
  "confidence": "medium",
  "description": "The API removed the X parameter without warning",
  "resolution_status": "fixed",
  "resolution_details": "The team restored backward compatibility in version Y"
}}
```

Or if no violation:
```json
{{
  "has_violation": false
}}
```

Respond ONLY with valid JSON.
"""

    def analyze_issue(self, title: str, body: str, comments: str) -> IssueAnalysisDTO:
        """Analyze a GitHub issue for contract violations.

        Args:
            title: Issue title
            body: Issue body
            comments: Issue comments

        Returns:
            Analysis results
        """
        try:
            # Prepare the prompt
            prompt = self._prepare_prompt(title, body, comments)

            # Get analysis from LLM
            response = self.llm_client.analyze(prompt)

            # Extract JSON from response
            try:
                # Look for JSON within markdown code blocks
                if "```json" in response and "```" in response.split("```json", 1)[1]:
                    json_str = response.split("```json", 1)[
                        1].split("```", 1)[0].strip()
                elif "```" in response and "```" in response.split("```", 1)[1]:
                    json_str = response.split(
                        "```", 1)[1].split("```", 1)[0].strip()
                else:
                    json_str = response.strip()

                analysis_data = json.loads(json_str)
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response}")
                # Create a default response indicating failure to parse
                analysis_data = {
                    "has_violation": False,
                    "error": f"Failed to parse LLM response: {str(e)}"
                }

            # Create the analysis DTO
            analysis = IssueAnalysisDTO(
                # Placeholder ID
                issue_id=f"temp-{datetime.now().timestamp()}",
                issue_number=0,  # Placeholder number
                issue_title=title,
                issue_url="",  # Will be filled in later
                has_violation=analysis_data.get("has_violation", False),
                violation_type=analysis_data.get("violation_type") if analysis_data.get(
                    "has_violation", False) else None,
                severity=analysis_data.get("severity") if analysis_data.get(
                    "has_violation", False) else None,
                confidence=analysis_data.get("confidence") if analysis_data.get(
                    "has_violation", False) else None,
                description=analysis_data.get("description") if analysis_data.get(
                    "has_violation", False) else None,
                resolution_status=analysis_data.get("resolution_status") if analysis_data.get(
                    "has_violation", False) else None,
                resolution_details=analysis_data.get(
                    "resolution_details") if analysis_data.get("has_violation", False) else None,
                resolution_time=None  # Will be calculated later if closed_at is available
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing issue: {str(e)}")
            # Return a default response on error
            return IssueAnalysisDTO(
                issue_id=f"error-{datetime.now().timestamp()}",
                issue_number=0,
                issue_title=title,
                issue_url="",
                has_violation=False,
                description=f"Error analyzing issue: {str(e)}"
            )

    def fetch_issues(self, repo_name: str, num_issues: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues from GitHub.

        This is a legacy method used for direct GitHub fetching.
        The new architecture uses MongoDB as the data source.

        Args:
            repo_name: Repository name (format: owner/repo)
            num_issues: Number of issues to fetch

        Returns:
            List of issues
        """
        # This is a placeholder method that returns an empty list
        # In the new architecture, we use MongoDB as the data source
        logger.warning(
            "Direct GitHub fetching is deprecated. Use MongoDB as data source instead.")
        return []
