"""GitHub issues analyzer for contract violations research."""

import os
import json
import signal
from pathlib import Path
from datetime import datetime
import yaml
from typing import List, Dict, Any, Optional
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


class CheckpointManager:
    """Manages checkpoints for the GitHub issues analysis process."""

    def __init__(self, output_dir: Path = None):
        """Initialize the checkpoint manager."""
        if output_dir is None:
            output_dir = Path(settings.DATA_DIR) / 'analyzed'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / 'analysis_checkpoint.json'

    def save_checkpoint(self, analyzed_issues: List[Dict], current_index: int,
                        total_issues: List[Dict], repo_name: str) -> None:
        """Save the current analysis state to a checkpoint file."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'analyzed_issues': analyzed_issues,
            'current_index': current_index,
            'total_issues': total_issues,
            'repo_name': repo_name
        }

        # Save to temporary file first to prevent corruption
        temp_checkpoint = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            # Atomic rename to prevent corruption
            temp_checkpoint.replace(self.checkpoint_file)
            logger.info(f"Checkpoint saved at index {current_index}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()

    def load_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint from index {
                        checkpoint_data['current_index']}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear the existing checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.error(f"Error clearing checkpoint: {str(e)}")


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
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint if it exists')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Number of issues to process before creating a checkpoint')
    args = parser.parse_args()

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()

    # Initialize analyzer
    analyzer = GitHubIssuesAnalyzer()

    # Flag to track if we're shutting down
    is_shutting_down = False

    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        nonlocal is_shutting_down
        if is_shutting_down:
            logger.warning("Forced shutdown requested. Exiting immediately.")
            exit(1)

        logger.info(
            "Shutdown signal received. Saving checkpoint before exit...")
        is_shutting_down = True

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Check for existing checkpoint if resume is requested
        checkpoint_data = None
        if args.resume:
            checkpoint_data = checkpoint_mgr.load_checkpoint()
            if checkpoint_data:
                logger.info("Resuming from checkpoint...")
                analyzed_issues = checkpoint_data['analyzed_issues']
                current_index = checkpoint_data['current_index']
                issues = checkpoint_data['total_issues']
                if args.repo != checkpoint_data['repo_name']:
                    logger.warning(
                        f"Warning: Checkpoint is for repo {
                            checkpoint_data['repo_name']}, "
                        f"but analyzing {args.repo}")
            else:
                logger.info("No checkpoint found. Starting fresh analysis.")

        # If no checkpoint or not resuming, fetch issues
        if not checkpoint_data:
            issues = analyzer.fetch_issues(
                repo_name=args.repo, num_issues=args.issues)
            logger.info("Fetched {} issues".format(len(issues)))
            analyzed_issues = []
            current_index = 0

        # Save raw data
        output_dir = Path(settings.DATA_DIR) / 'analyzed'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_csv_path = output_dir / f'github_issues_raw_{timestamp}.csv'

        df = pd.DataFrame(issues)
        df.to_csv(raw_csv_path, index=False)
        logger.info("Saved raw data to {}".format(raw_csv_path))

        # Analyze issues
        logger.info("Starting issue analysis")

        # Create progress bar for issue analysis
        with tqdm(total=len(issues), initial=current_index,
                  desc="Analyzing issues", unit="issue") as pbar:
            while current_index < len(issues) and not is_shutting_down:
                try:
                    issue = issues[current_index]
                    analysis = analyzer.analyze_issue(
                        issue.get('title', ''), issue.get('body', ''))
                    analyzed_issue = {**issue, **analysis}
                    analyzed_issues.append(analyzed_issue)

                    # Update progress
                    current_index += 1
                    pbar.update(1)

                    # Create checkpoint at specified intervals
                    if current_index % args.checkpoint_interval == 0:
                        checkpoint_mgr.save_checkpoint(
                            analyzed_issues, current_index, issues, args.repo)

                except Exception as e:
                    logger.error("Error analyzing issue {}: {}".format(
                        issue.get('number'), str(e)))
                    current_index += 1  # Skip problematic issue
                    continue

        # Save final results if we have analyzed any issues
        if analyzed_issues:
            if is_shutting_down:
                logger.info("Saving final checkpoint before shutdown...")
                checkpoint_mgr.save_checkpoint(
                    analyzed_issues, current_index, issues, args.repo)
            else:
                logger.info("Analysis complete. Saving final results...")
                analyzer.save_results(analyzed_issues)
                # Clear checkpoint since we completed successfully
                checkpoint_mgr.clear_checkpoint()

    except Exception as e:
        logger.error("Error during overall analysis: {}".format(str(e)))
        # Save checkpoint on error
        if 'analyzed_issues' in locals() and 'current_index' in locals():
            checkpoint_mgr.save_checkpoint(
                analyzed_issues, current_index, issues, args.repo)


if __name__ == '__main__':
    main()
