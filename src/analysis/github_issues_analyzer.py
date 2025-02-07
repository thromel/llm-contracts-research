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
from tqdm import tqdm  # Progress bar
from github import Github

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

        # Load context files (violation types, severity criteria, categorization)
        self.context_dir = Path(settings.CONTEXT_DIR)
        self.violation_types = self._load_context_file('violation_types.yaml')
        self.severity_criteria = self._load_context_file(
            'severity_criteria.yaml')
        self.categorization = self._load_context_file('categorization.yaml')

    def _load_context_file(self, filename: str) -> Dict[str, Any]:
        """Load a context file from the context directory."""
        try:
            file_path = self.context_dir / filename
            if not file_path.exists():
                logger.warning("{} not found in context directory at {}".format(
                    filename, self.context_dir))
                return {}

            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Error loading context file {}: {}".format(
                filename, str(e)))
            return {}

    def fetch_issues(self, repo_name: str, num_issues: int = 100) -> List[Dict[str, Any]]:
        """Fetch issues from a GitHub repository."""
        logger.info("Fetching {} issues from {}".format(num_issues, repo_name))

        try:
            # Initialize GitHub client
            g = Github(self.github_token)
            repo = g.get_repo(repo_name)

            # Get closed issues
            issues = []
            total_fetched = 0
            skipped_prs = 0

            # Create progress bar for issue fetching
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

        except Exception as e:
            logger.error("Error fetching issues: {}".format(str(e)))
            raise

    def _build_system_prompt(self) -> str:
        """Build a concise system prompt for analysis."""
        # Use a short summary for the smaller model
        system_prompt = (
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
        return system_prompt

    def _build_user_prompt(self, title: str, body: str, comments: str = None) -> str:
        """Build a concise user prompt with issue details."""
        prompt = f"Issue Title: {title}\n\nIssue Body: {body}\n\n"
        if comments:
            prompt += f"Issue Comments: {comments}\n\n"
        prompt += (
            "Analyze if this issue shows an API contract violation. Provide your answer as JSON with keys: "
            "has_violation (bool), violation_type (string or null), severity (high/medium/low or null), "
            "description (string), confidence (high/medium/low), resolution_status (string), resolution_details (string)."
        )
        return prompt

    def analyze_issue(self, title: str, body: str, comments: str = None) -> Dict[str, Any]:
        """Analyze a single issue using the OpenAI API with refined prompts."""
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

            # Remove any markdown code block markers if present
            response_content = response_content.replace(
                '```json', '').replace('```', '').strip()

            try:
                # Try parsing as JSON first
                analysis = json.loads(response_content)
                if not isinstance(analysis, dict):
                    return {"has_violation": False, "error": "Invalid response format", "confidence": "low"}
                return analysis
            except json.JSONDecodeError as je:
                try:
                    # Fallback to YAML parsing if JSON fails
                    analysis = yaml.safe_load(response_content)
                    if not isinstance(analysis, dict):
                        return {"has_violation": False, "error": "Invalid response format", "confidence": "low"}
                    return analysis
                except yaml.YAMLError as ye:
                    logger.error(f"Error parsing response content: {ye}")
                    logger.error(f"Problematic content: {response_content}")
                    return {
                        "has_violation": False,
                        "error": "Response parsing error",
                        "confidence": "low",
                        "description": "Failed to parse API response"
                    }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "has_violation": False,
                "error": str(e),
                "confidence": "low",
                "description": "API call failed"
            }

    def analyze_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze all issues and merge the analysis results with issue data."""
        logger.info("Analyzing issues for contract violations")
        analyses = []
        for issue in tqdm(issues, desc="Analyzing issues"):
            issue_body = issue.get('body') or ''
            issue_comments = issue.get('comments') or ''
            sanitized_body = self._sanitize_text(issue_body)
            sanitized_comments = self._sanitize_text(issue_comments)
            analysis = self.analyze_issue(
                issue.get('title', 'No Title'),
                sanitized_body,
                sanitized_comments
            )
            safe_issue_data = {
                'number': issue.get('number'),
                'title': issue.get('title'),
                'state': issue.get('state'),
                'created_at': issue.get('created_at'),
                'closed_at': issue.get('closed_at'),
                'labels': issue.get('labels', []),
                'url': issue.get('url'),
                'resolution_time': issue.get('resolution_time')
            }
            result = {**safe_issue_data, **analysis}
            analyses.append(result)
        return analyses

    def _sanitize_text(self, text: str) -> str:
        """Remove potential sensitive information from text."""
        if not text:
            return ""
        patterns = [
            (r'sk-[a-zA-Z0-9]{48}', '[OPENAI_KEY]'),
            (r'ghp_[a-zA-Z0-9]{36}', '[GITHUB_TOKEN]'),
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
            (r'(?i)api[_-]?key[_-]?(?:=|\s*:\s*)\s*["\']?[\w\-]{32,}["\']?', '[API_KEY]'),
            (r'(?i)secret[_-]?key[_-]?(?:=|\s*:\s*)\s*["\']?[\w\-]{32,}["\']?', '[SECRET_KEY]'),
            (r'(?i)password[_-]?(?:=|\s*:\s*)\s*["\']?[^\s"\']{8,}["\']?', '[PASSWORD]'),
            (r'(?i)token[_-]?(?:=|\s*:\s*)\s*["\']?[\w\-]{32,}["\']?', '[TOKEN]')
        ]
        import re
        sanitized = text
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized)
        return sanitized

    def save_results(self, results: List[Dict[str, Any]], output_dir: Path = None) -> None:
        """Save analysis results to JSON and optionally CSV formats."""
        if output_dir is None:
            output_dir = Path(settings.DATA_DIR) / 'analyzed'

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = 'github_issues_analysis_{}'.format(timestamp)

        # Save raw data first
        raw_csv_path = output_dir / '{}_raw.csv'.format(base_name)
        try:
            df = pd.DataFrame(results)
            df.to_csv(raw_csv_path, index=False)
            logger.info("Saved raw data to {}".format(raw_csv_path))
        except Exception as e:
            logger.error("Error saving raw data: {}".format(str(e)))

        # Save final results
        final_csv_path = output_dir / '{}_final.csv'.format(base_name)
        try:
            df = pd.DataFrame(results)
            df.to_csv(final_csv_path, index=False)
            logger.info("Saved CSV results to {}".format(final_csv_path))
        except Exception as e:
            logger.error("Error saving CSV results: {}".format(str(e)))

        # Save as JSON
        json_path = output_dir / '{}_final.json'.format(base_name)
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Saved JSON results to {}".format(json_path))
        except Exception as e:
            logger.error("Error saving JSON results: {}".format(str(e)))


def main():
    """Main function to run the GitHub issues analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze GitHub issues for contract violations.')
    parser.add_argument('--repo', type=str, default='openai/openai-python',
                        help='GitHub repository to analyze (format: owner/repo)')
    parser.add_argument('--issues', type=int, default=100,
                        help='Number of issues to analyze')
    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = GitHubIssuesAnalyzer()

        # Fetch and analyze issues
        issues = analyzer.fetch_issues(
            repo_name=args.repo, num_issues=args.issues)
        logger.info("Fetched {} issues".format(len(issues)))

        # Save raw data first
        output_dir = Path(settings.DATA_DIR) / 'analyzed'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_csv_path = output_dir / \
            'github_issues_raw_{}.csv'.format(timestamp)

        df = pd.DataFrame(issues)
        df.to_csv(raw_csv_path, index=False)
        logger.info("Saved raw data to {}".format(raw_csv_path))

        # Analyze issues
        logger.info("Starting issue analysis")
        analyzed_issues = []

        # Create progress bar for issue analysis
        for issue in tqdm(issues, desc="Analyzing issues", unit="issue"):
            try:
                analysis = analyzer.analyze_issue(
                    issue.get('title', ''), issue.get('body', ''))
                analyzed_issue = {**issue, **analysis}
                analyzed_issues.append(analyzed_issue)

                # Save intermediate results every 10 issues
                if len(analyzed_issues) % 10 == 0:
                    intermediate_path = output_dir / \
                        'github_issues_analysis_{}_{}.csv'.format(
                            timestamp, len(analyzed_issues))
                    pd.DataFrame(analyzed_issues).to_csv(
                        intermediate_path, index=False)
                    logger.info("Saved intermediate results to {}".format(
                        intermediate_path))

            except Exception as e:
                logger.error("Error analyzing issue {}: {}".format(
                    issue.get('number'), str(e)))
                continue

        # Save final results
        analyzer.save_results(analyzed_issues)

    except Exception as e:
        logger.error("Error during overall analysis: {}".format(str(e)))


if __name__ == '__main__':
    main()
