"""GitHub issues analyzer for contract violations research."""

import os
import json
from pathlib import Path
from datetime import datetime
import yaml
from typing import List, Dict, Any
from openai import OpenAI
from src.config import settings
from src.utils.logger import setup_logger
import pandas as pd
import argparse

logger = setup_logger(__name__)

# Constants for the prompt configuration
PROMPT_TEMPERATURE = 0.7
PROMPT_MAX_TOKENS = 1024


class GitHubIssuesAnalyzer:
    def __init__(self, repo_name: str = "openai/openai-python", model: str = None):
        """Initialize the analyzer."""
        self.github_token = settings.GITHUB_TOKEN
        self.repo_name = repo_name

        # Initialize OpenAI client using settings
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        # Use model from settings if none specified
        self.model = model or settings.OPENAI_MODEL

        # Load contract violation types and criteria from the context directory
        self.context_dir = Path(settings.CONTEXT_DIR)
        self.violation_types = self._load_yaml('violation_types.yaml')
        self.severity_criteria = self._load_yaml('severity_criteria.yaml')
        self.categorization = self._load_yaml('categorization.yaml')

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML file from context directory and return a dictionary.
        If the file is missing or parsing fails, log a warning and return an empty dict.
        """
        filepath = self.context_dir / filename
        if not filepath.exists():
            logger.warning("{} not found in context directory at {}".format(
                filename, self.context_dir))
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Error loading {}: {}".format(filename, str(e)))
            return {}

    def fetch_issues(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch closed issues from GitHub using requests."""
        logger.info("Fetching {} closed issues from {}".format(
            limit, self.repo_name))

        import requests
        headers = {
            'Authorization': 'token {}'.format(self.github_token),
            'Accept': 'application/vnd.github.v3+json'
        }

        issues_data = []
        page = 1
        per_page = min(100, limit)  # GitHub max is 100 per page

        while len(issues_data) < limit:
            url = "https://api.github.com/repos/{}/issues".format(
                self.repo_name)
            params = {
                'state': 'closed',  # Only fetch closed issues
                'per_page': per_page,
                'page': page,
                'sort': 'updated',  # Sort by last updated
                'direction': 'desc'  # Most recently updated first
            }

            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                logger.error("Failed to fetch issues: {}".format(
                    response.status_code))
                break

            batch = response.json()
            if not batch:  # No more issues available
                break

            for issue in batch:
                if len(issues_data) >= limit:
                    break

                # Skip pull requests
                if issue.get('pull_request'):
                    continue

                issue_data = {
                    'number': issue.get('number'),
                    'title': issue.get('title'),
                    'body': issue.get('body'),
                    'state': issue.get('state'),
                    'created_at': issue.get('created_at'),
                    'closed_at': issue.get('closed_at'),
                    'labels': [label.get('name') for label in issue.get('labels', [])],
                    'url': issue.get('html_url'),
                    'resolution_time': None  # Will calculate below
                }

                # Calculate resolution time in hours if we have both dates
                if issue_data['created_at'] and issue_data['closed_at']:
                    created = datetime.fromisoformat(
                        issue_data['created_at'].replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(
                        issue_data['closed_at'].replace('Z', '+00:00'))
                    # Convert to hours
                    resolution_time = (closed - created).total_seconds() / 3600
                    issue_data['resolution_time'] = round(resolution_time, 2)

                issues_data.append(issue_data)

            page += 1

        logger.info("Fetched {} closed issues.".format(len(issues_data)))
        return issues_data

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the OpenAI API."""
        violation_types_str = """
        Violation Types:
        - input_type_violation: Incorrect data type provided to API
        - input_value_violation: Invalid value provided within correct type
        - missing_dependency_violation: Required dependency not present
        - missing_option_violation: Required configuration option not provided
        - method_order_violation: API methods called in incorrect sequence
        - memory_out_of_bound: Memory usage exceeds specified limits
        - performance_degradation: Significant unexpected performance issues
        - incorrect_functionality: API behaves differently than documented
        - hang: API call never returns or deadlocks
        """

        severity_criteria_str = """
        Severity Criteria:
        - high: Crashes, data loss, security issues, or complete failure
        - medium: Degraded functionality, workarounds available
        - low: Minor issues, minimal impact on functionality
        """

        categorization_str = """
        Analysis Guidelines:
        1. Examine both the initial issue description AND subsequent comments
        2. Look for maintainer responses that confirm or deny the issue
        3. Consider resolution status and any provided fixes
        4. Check if the issue was a misunderstanding or user error
        5. Evaluate if workarounds were provided and their effectiveness
        6. Note if the issue was fixed in a later release
        
        A true contract violation should be:
        1. Confirmed by maintainers or clearly demonstrated
        2. Not a result of user error or misunderstanding
        3. Related to the API's behavior or requirements
        4. Documented with clear impact and reproduction steps
        
        False positives include:
        1. User misunderstandings of documented behavior
        2. Issues caused by incorrect usage
        3. Feature requests or enhancements
        4. Problems with user's environment/setup
        """

        return "{}\n{}\n{}\n\nAnalyze the provided GitHub issue and its comments to determine if it represents a true API contract violation. Consider the full context of the discussion, including maintainer responses and resolution status.".format(
            violation_types_str, severity_criteria_str, categorization_str)

    def _build_user_prompt(self, title: str, body: str, comments: str = None) -> str:
        """Build the user prompt for issue analysis."""
        prompt = "Issue Title: {}\n\n".format(title)
        prompt += "Issue Body:\n{}\n\n".format(body)

        if comments:
            prompt += "Issue Comments:\n{}\n\n".format(comments)

        prompt += "Please analyze this issue and determine if it represents a contract violation. "
        prompt += "Consider both the initial issue description AND the comments/discussion to make your determination. "
        prompt += "Pay special attention to:\n"
        prompt += "1. Whether the issue was confirmed as a real problem\n"
        prompt += "2. If maintainers acknowledged it as a violation\n"
        prompt += "3. How it was resolved (if it was)\n"
        prompt += "4. Any workarounds or fixes suggested\n\n"
        prompt += "Provide your analysis in JSON format with the following fields:\n"
        prompt += "- has_violation (boolean)\n"
        prompt += "- violation_type (string, null if no violation)\n"
        prompt += "- severity (string: high/medium/low, null if no violation)\n"
        prompt += "- description (string explaining the violation or why it's not one)\n"
        prompt += "- confidence (string: high/medium/low)\n"
        prompt += "- resolution_status (string: fixed/workaround/unresolved/false_positive)\n"
        prompt += "- resolution_details (string describing how it was resolved or why it was determined to be a false positive)"

        return prompt

    def analyze_issue(self, title: str, body: str, comments: str = None) -> Dict[str, Any]:
        """Analyze a single issue for contract violations using the OpenAI API."""
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
                analysis = yaml.safe_load(response_content)
                if not isinstance(analysis, dict):
                    return {
                        "has_violation": False,
                        "error": "Invalid response format",
                        "confidence": "low"
                    }
                return analysis
            except yaml.YAMLError as ye:
                logger.error("Error parsing YAML: {}".format(str(ye)))
                return {
                    "has_violation": False,
                    "error": "YAML parsing error",
                    "confidence": "low"
                }

        except Exception as e:
            logger.error("Error calling OpenAI API: {}".format(str(e)))
            return {
                "has_violation": False,
                "error": str(e),
                "confidence": "low"
            }

    def analyze_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze all issues and combine issue data with analysis results."""
        logger.info("Analyzing issues for contract violations")
        analyses = []
        for issue in issues:
            issue_body = issue.get('body') or ''
            issue_comments = issue.get('comments') or ''
            analysis = self.analyze_issue(
                issue.get('title', 'No Title'),
                issue_body,
                issue_comments)
            # Merge issue data with analysis result
            result = {**issue, **analysis}
            analyses.append(result)
        return analyses

    def save_results(self, results: List[Dict[str, Any]], output_dir: Path = None) -> None:
        """Save analysis results to JSON and optionally CSV formats."""
        if output_dir is None:
            output_dir = Path(settings.DATA_DIR) / 'analyzed'

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = 'github_issues_analysis_{}'.format(timestamp)

        # Save as JSON
        if getattr(settings, "JSON_EXPORT", False):
            json_path = output_dir / '{}.json'.format(base_name)
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info("Saved JSON results to {}".format(json_path))
            except Exception as e:
                logger.error("Error saving JSON results: {}".format(str(e)))

        # Save as CSV if requested
        if getattr(settings, "CSV_EXPORT", False):
            try:
                df = pd.DataFrame(results)
                csv_path = output_dir / '{}.csv'.format(base_name)
                df.to_csv(csv_path, index=False)
                logger.info("Saved CSV results to {}".format(csv_path))
            except Exception as e:
                logger.error("Error saving CSV results: {}".format(str(e)))


def main():
    """Main function to run the analysis."""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Analyze GitHub issues for API contract violations.')
    parser.add_argument('--repo', type=str, default="openai/openai-python",
                        help='GitHub repository in format owner/repo (e.g., openai/openai-python)')
    parser.add_argument('--issues', type=int, default=100,
                        help='Number of most recent closed issues to analyze')
    args = parser.parse_args()

    logger.info("Starting analysis for repository: {}".format(args.repo))
    logger.info("Will analyze {} closed issues".format(args.issues))

    analyzer = GitHubIssuesAnalyzer(repo_name=args.repo)
    output_dir = Path(settings.DATA_DIR) / 'analyzed'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        # Fetch issues from GitHub
        issues = analyzer.fetch_issues(limit=args.issues)
        logger.info("Fetched {} issues".format(len(issues)))

        # Save raw data first
        raw_df = pd.DataFrame(issues)
        raw_csv_path = output_dir / "github_issues_raw_{}_{}.csv".format(
            args.repo.replace('/', '_'), timestamp)
        raw_df.to_csv(raw_csv_path, index=False)
        logger.info("Saved raw data to {}".format(raw_csv_path))

        # Initialize analysis columns
        raw_df['has_violation'] = False
        raw_df['violation_type'] = None
        raw_df['severity'] = None
        raw_df['description'] = None
        raw_df['confidence'] = None
        raw_df['analysis_error'] = None

        # Analyze issues one by one and save progress regularly
        for idx, issue in enumerate(issues):
            try:
                logger.info(
                    "Analyzing issue {}/{}".format(idx + 1, len(issues)))
                analysis = analyzer.analyze_issue(
                    issue.get('title', 'No Title'),
                    issue.get('body', ''),
                    issue.get('comments', '')
                )

                if analysis:
                    raw_df.loc[idx, 'has_violation'] = analysis.get(
                        'has_violation', False)
                    raw_df.loc[idx, 'violation_type'] = analysis.get(
                        'violation_type')
                    raw_df.loc[idx, 'severity'] = analysis.get('severity')
                    raw_df.loc[idx, 'description'] = analysis.get(
                        'description')
                    raw_df.loc[idx, 'confidence'] = analysis.get('confidence')
                else:
                    raw_df.loc[idx, 'analysis_error'] = 'Analysis returned None'

                # Save progress every 10 issues
                if (idx + 1) % 10 == 0:
                    progress_csv_path = output_dir / "github_issues_analysis_{}_{}_{}.csv".format(
                        args.repo.replace('/', '_'), timestamp, idx + 1)
                    raw_df.to_csv(progress_csv_path, index=False)
                    logger.info(
                        "Saved progress after {} issues".format(idx + 1))

            except Exception as e:
                error_msg = str(e)
                logger.error("Error analyzing issue {}: {}".format(
                    idx + 1, error_msg))
                raw_df.loc[idx, 'analysis_error'] = error_msg
                continue

        # Save final results
        final_csv_path = output_dir / "github_issues_analysis_{}_final_{}.csv".format(
            args.repo.replace('/', '_'), timestamp)
        raw_df.to_csv(final_csv_path, index=False)
        logger.info(
            "Analysis complete. Final results saved to {}".format(final_csv_path))

        # Save JSON version if requested
        if settings.JSON_EXPORT:
            json_path = output_dir / "github_issues_analysis_{}_final_{}.json".format(
                args.repo.replace('/', '_'), timestamp)
            raw_df.to_json(json_path, orient='records', indent=2)
            logger.info("Saved JSON results to {}".format(json_path))

    except Exception as e:
        logger.error("Error during overall analysis: {}".format(str(e)))
        # Try to save whatever data we have
        try:
            error_csv_path = output_dir / "github_issues_partial_{}_{}.csv".format(
                args.repo.replace('/', '_'), timestamp)
            raw_df.to_csv(error_csv_path, index=False)
            logger.info("Saved partial data to {}".format(error_csv_path))
        except Exception as save_error:
            logger.error(
                "Could not save partial data: {}".format(str(save_error)))


if __name__ == '__main__':
    main()
