"""Main script for GitHub issues analysis."""

import argparse
import signal
from pathlib import Path
from datetime import datetime

from src.analysis.core.analyzers import GitHubIssuesAnalyzer
from src.analysis.core.processors.checkpoint import CheckpointHandler
from src.analysis.core.data_loader import CSVDataLoader, DataLoadError
from src.analysis.core.storage.factory import StorageFactory
from src.analysis.core.clients.openai import OpenAIClient
from src.analysis.core.dto import AnalysisMetadataDTO, AnalysisResultsDTO
from src.config import settings
from src.utils.logger import setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)


class AnalysisOrchestrator:
    """Orchestrates the GitHub issues analysis process."""

    def __init__(self, analyzer: GitHubIssuesAnalyzer, checkpoint_mgr: CheckpointHandler):
        """Initialize the orchestrator.

        Args:
            analyzer: Issue analyzer instance
            checkpoint_mgr: Checkpoint manager instance
        """
        self.analyzer = analyzer
        self.checkpoint_mgr = checkpoint_mgr
        self.is_shutting_down = False
        # Initialize storage
        self.storage = StorageFactory.create_storage()

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
                    issues = self.analyzer.github_client.fetch_issues(
                        repo_name=repo_name, num_issues=num_issues)
                    logger.info("Fetched {} issues".format(len(issues)))
                analyzed_issues = []
                current_index = 0

            # Create metadata for checkpoints and results
            metadata = AnalysisMetadataDTO(
                repository=repo_name,
                analysis_timestamp=datetime.now().isoformat(),
                num_issues=len(issues)
            )

            # Create progress bar for issue analysis
            with tqdm(total=len(issues), initial=current_index,
                      desc="Analyzing issues", unit="issue") as pbar:
                while current_index < len(issues) and not self.is_shutting_down:
                    try:
                        issue = issues[current_index]
                        # Access DTO attributes directly
                        analysis = self.analyzer.analyze_issue(
                            title=issue.title,
                            body=issue.body,
                            comments=', '.join(
                                comment.body for comment in issue.first_comments)
                        )
                        analyzed_issues.append(analysis)

                        # Update progress
                        current_index += 1
                        pbar.update(1)

                        # Handle intermediate saves if enabled
                        if settings.SAVE_INTERMEDIATE:
                            intermediate_metadata = AnalysisMetadataDTO(
                                repository=repo_name,
                                analysis_timestamp=datetime.now().isoformat(),
                                num_issues=current_index
                            )
                            intermediate_storage = StorageFactory.create_storage(
                                storage_types=['json'],
                                is_intermediate=True
                            )
                            for storage in intermediate_storage:
                                storage.save_results(
                                    analyzed_issues=analyzed_issues[:current_index],
                                    metadata=intermediate_metadata
                                )
                            logger.info(
                                f"Saved intermediate results after {current_index} issues")

                        # Create checkpoint at specified intervals
                        if current_index % checkpoint_interval == 0:
                            self.checkpoint_mgr.save_checkpoint(
                                analyzed_issues=analyzed_issues,
                                metadata=metadata
                            )
                            logger.info(
                                "Saved checkpoint after {} issues".format(current_index))

                    except Exception as exc:
                        logger.error("Error analyzing issue {}: {}".format(
                            issue.number, str(exc)))
                        current_index += 1  # Skip problematic issue
                        continue

            # Save final results if we have analyzed any issues
            if analyzed_issues:
                if self.is_shutting_down:
                    logger.info("Saving final checkpoint before shutdown...")
                    self.checkpoint_mgr.save_checkpoint(
                        analyzed_issues=analyzed_issues,
                        metadata=metadata
                    )
                else:
                    logger.info("Analysis complete. Saving final results...")
                    # Create final results DTO
                    results = AnalysisResultsDTO(
                        metadata=metadata,
                        analyzed_issues=analyzed_issues
                    )

                    # Save final results using all configured storage types
                    try:
                        for storage in self.storage:
                            storage.save_results(
                                analyzed_issues=analyzed_issues,
                                metadata=metadata
                            )
                            logger.info(
                                f"Saved final results using {storage.__class__.__name__}")
                    except Exception as e:
                        logger.error(f"Error saving final results: {str(e)}")
                        raise

                    # Clear checkpoint since we completed successfully
                    self.checkpoint_mgr.clear_checkpoint()

                    return results

        except Exception as exc:
            logger.error("Error during analysis: {}".format(str(exc)))
            # Save checkpoint on error
            if 'analyzed_issues' in locals() and 'current_index' in locals():
                self.checkpoint_mgr.save_checkpoint(
                    analyzed_issues=analyzed_issues,
                    metadata=metadata
                )
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
    checkpoint_mgr = CheckpointHandler()

    # Initialize OpenAI client
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
    if hasattr(settings, 'OPENAI_BASE_URL'):
        openai_settings['base_url'] = settings.OPENAI_BASE_URL
    llm_client = OpenAIClient(**openai_settings)

    # Create and run orchestrator
    analyzer = GitHubIssuesAnalyzer(
        llm_client=llm_client,
        github_token=settings.GITHUB_TOKEN,
        checkpoint_handler=checkpoint_mgr
    )
    orchestrator = AnalysisOrchestrator(analyzer, checkpoint_mgr)
    orchestrator.setup_signal_handlers()

    try:
        results = orchestrator.run_analysis(
            repo_name=args.repo,
            num_issues=args.issues,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            input_csv=args.input_csv
        )
        print(
            f"Analysis completed successfully. Results saved for repository: {results.metadata.repository}")
        if settings.JSON_EXPORT:
            print(f"JSON results saved to: {settings.EXPORT_DIR}/json/")
        if settings.CSV_EXPORT:
            print(f"CSV results saved to: {settings.EXPORT_DIR}/csv/")
        if settings.MONGODB_ENABLED:
            print("Results also saved to MongoDB")
    except Exception as exc:
        logger.error("Analysis failed: {}".format(str(exc)))
        exit(1)


if __name__ == '__main__':
    main()
