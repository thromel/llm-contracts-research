"""Main script for GitHub issues analysis."""

import argparse
import signal
from pathlib import Path

from src.analysis.core.analyzer import GitHubIssuesAnalyzer
from src.analysis.core.checkpoint import CheckpointManager
from src.analysis.core.data_loader import CSVDataLoader, DataLoadError
from src.config import settings
from src.utils.logger import setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)


class AnalysisOrchestrator:
    """Orchestrates the GitHub issues analysis process."""

    def __init__(self, analyzer: GitHubIssuesAnalyzer, checkpoint_mgr: CheckpointManager):
        """Initialize the orchestrator.

        Args:
            analyzer: Issue analyzer instance
            checkpoint_mgr: Checkpoint manager instance
        """
        self.analyzer = analyzer
        self.checkpoint_mgr = checkpoint_mgr
        self.is_shutting_down = False

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            if self.is_shutting_down:
                logger.warning(
                    "Forced shutdown requested. Exiting immediately.")
                exit(1)
            logger.info(
                "Shutdown signal received. Saving checkpoint before exit...")
            self.is_shutting_down = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run_analysis(self, repo_name: str = None, num_issues: int = None,
                     checkpoint_interval: int = 5, resume: bool = False,
                     input_csv: Path = None):
        """Run the analysis process.

        Args:
            repo_name: Repository to analyze (not needed if using input_csv)
            num_issues: Number of issues to analyze (not needed if using input_csv)
            checkpoint_interval: Interval between checkpoints
            resume: Whether to resume from checkpoint
            input_csv: Optional path to input CSV file
        """
        try:
            # Check for existing checkpoint if resume is requested
            checkpoint_data = None
            if resume:
                checkpoint_data = self.checkpoint_mgr.load_checkpoint()
                if checkpoint_data:
                    logger.info("Resuming from checkpoint...")
                    analyzed_issues = checkpoint_data['analyzed_issues']
                    current_index = checkpoint_data['current_index']
                    issues = checkpoint_data['total_issues']
                    if repo_name and repo_name != checkpoint_data['repo_name']:
                        logger.warning(
                            "Warning: Checkpoint is for repo {}, but analyzing {}".format(
                                checkpoint_data['repo_name'], repo_name))
                else:
                    logger.info(
                        "No checkpoint found. Starting fresh analysis.")

            # If no checkpoint or not resuming, get issues
            if not checkpoint_data:
                if input_csv:
                    try:
                        issues = CSVDataLoader.load_from_csv(input_csv)
                        repo_name = "csv_import"  # Use placeholder for CSV imports
                    except DataLoadError as exc:
                        logger.error(
                            "Failed to load CSV file: {}".format(str(exc)))
                        raise
                else:
                    if not repo_name or not num_issues:
                        raise ValueError(
                            "Either input_csv or both repo_name and num_issues must be provided")
                    issues = self.analyzer.fetch_issues(
                        repo_name=repo_name, num_issues=num_issues)
                    logger.info("Fetched {} issues".format(len(issues)))
                analyzed_issues = []
                current_index = 0

            # Create progress bar for issue analysis
            with tqdm(total=len(issues), initial=current_index,
                      desc="Analyzing issues", unit="issue") as pbar:
                while current_index < len(issues) and not self.is_shutting_down:
                    try:
                        issue = issues[current_index]
                        analysis = self.analyzer.analyze_issue(
                            issue.get('title', ''),
                            issue.get('body', ''),
                            ', '.join(comment['body'] for comment in issue.get(
                                'first_comments', []))
                        )
                        analyzed_issue = {**issue, **analysis}
                        analyzed_issues.append(analyzed_issue)

                        # Update progress
                        current_index += 1
                        pbar.update(1)

                        # Create checkpoint at specified intervals
                        if current_index % checkpoint_interval == 0:
                            self.checkpoint_mgr.save_checkpoint(
                                analyzed_issues, current_index, issues, repo_name)

                    except Exception as exc:
                        logger.error("Error analyzing issue {}: {}".format(
                            issue.get('number'), str(exc)))
                        current_index += 1  # Skip problematic issue
                        continue

            # Save final results if we have analyzed any issues
            if analyzed_issues:
                if self.is_shutting_down:
                    logger.info("Saving final checkpoint before shutdown...")
                    self.checkpoint_mgr.save_checkpoint(
                        analyzed_issues, current_index, issues, repo_name)
                else:
                    logger.info("Analysis complete. Saving final results...")
                    self.analyzer.save_results(analyzed_issues)
                    # Clear checkpoint since we completed successfully
                    self.checkpoint_mgr.clear_checkpoint()

        except Exception as exc:
            logger.error("Error during analysis: {}".format(str(exc)))
            # Save checkpoint on error
            if 'analyzed_issues' in locals() and 'current_index' in locals():
                self.checkpoint_mgr.save_checkpoint(
                    analyzed_issues, current_index, issues, repo_name)
            raise


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description='Analyze GitHub issues for contract violations.')

    # Create mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--repo', type=str,
                             help='GitHub repository to analyze (format: owner/repo)')
    input_group.add_argument('--input-csv', type=Path,
                             help='Path to input CSV file containing issues')

    # Other arguments
    parser.add_argument('--issues', type=int,
                        help='Number of issues to analyze (required with --repo)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint if it exists')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Number of issues to process before creating a checkpoint')
    args = parser.parse_args()

    # Validate arguments
    if args.repo and not args.issues:
        parser.error("--issues is required when using --repo")

    # Initialize components
    output_dir = Path(settings.DATA_DIR) / 'analyzed'
    analyzer = GitHubIssuesAnalyzer(repo_name=args.repo if args.repo else None)
    checkpoint_mgr = CheckpointManager(output_dir=output_dir)

    # Create and run orchestrator
    orchestrator = AnalysisOrchestrator(analyzer, checkpoint_mgr)
    orchestrator.setup_signal_handlers()

    try:
        orchestrator.run_analysis(
            repo_name=args.repo,
            num_issues=args.issues,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            input_csv=args.input_csv
        )
    except Exception as exc:
        logger.error("Analysis failed: {}".format(str(exc)))
        exit(1)


if __name__ == '__main__':
    main()
