"""Main script for GitHub issues analysis."""

import argparse
import signal
import asyncio
from pathlib import Path
from datetime import datetime
import os

from src.analysis.core.analyzers import GitHubIssuesAnalyzer
from src.analysis.core.processors.checkpoint import CheckpointHandler
from src.analysis.core.data_loader import CSVDataLoader, MongoDBDataLoader, DataLoadError
from src.analysis.core.storage.factory import StorageFactory
from src.analysis.core.clients.openai import OpenAIClient
from src.analysis.core.dto import AnalysisMetadataDTO, AnalysisResultsDTO
from src.core.config import load_config
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

    async def run_analysis(self, repo_name: str = None, num_issues: int = None,
                           checkpoint_interval: int = 5, resume: bool = False,
                           input_csv: Path = None, use_mongodb: bool = False):
        """Run the analysis process.

        Args:
            repo_name: Repository to analyze (not needed if using input_csv)
            num_issues: Number of issues to analyze (not needed if using input_csv)
            checkpoint_interval: Interval between checkpoints
            resume: Whether to resume from checkpoint
            input_csv: Optional path to input CSV file
            use_mongodb: Whether to use MongoDB as data source
        """
        try:
            # Load configuration
            config = load_config()

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
                elif use_mongodb:
                    if not repo_name:
                        raise ValueError(
                            "Repository name is required when using MongoDB")

                    # Initialize and connect to MongoDB
                    logger.info(
                        f"Loading issues from MongoDB for repository: {repo_name}")
                    mongodb_uri = os.getenv('MONGODB_URI')
                    mongodb_db = os.getenv('MONGODB_DB')

                    if not mongodb_uri or not mongodb_db:
                        raise ValueError(
                            "MongoDB URI and DB name must be set in environment variables")

                    mongo_loader = MongoDBDataLoader(mongodb_uri, mongodb_db)
                    await mongo_loader.connect()

                    try:
                        issues = await mongo_loader.load_repository_issues(repo_name, num_issues)
                        logger.info(
                            f"Loaded {len(issues)} issues from MongoDB")
                    finally:
                        await mongo_loader.disconnect()
                else:
                    if not repo_name or not num_issues:
                        raise ValueError(
                            "Either input_csv, use_mongodb, or both repo_name and num_issues must be provided")
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
                        comments_text = ""
                        if hasattr(issue, 'first_comments') and issue.first_comments:
                            comments_text = ', '.join(
                                comment.body for comment in issue.first_comments if comment.body)

                        analysis = self.analyzer.analyze_issue(
                            title=issue.title,
                            body=issue.body,
                            comments=comments_text
                        )
                        analyzed_issues.append(analysis)

                        # Update progress
                        current_index += 1
                        pbar.update(1)

                        # Handle intermediate saves if enabled
                        if os.getenv('SAVE_INTERMEDIATE', 'false').lower() == 'true':
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


async def main_async():
    """Async main entry point for the analysis script."""
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
                        help='Number of issues to analyze (optional with --repo)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint if it exists')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Number of issues to process before creating a checkpoint')
    parser.add_argument('--config', type=str,
                        help='Path to YAML config file (optional)')
    parser.add_argument('--use-mongodb', action='store_true',
                        help='Use MongoDB as data source instead of fetching from GitHub')
    args = parser.parse_args()

    # Validate arguments
    if args.repo and not args.use_mongodb and not args.issues:
        parser.error(
            "--issues is required when using --repo without --use-mongodb")

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointHandler()

        # Initialize OpenAI client with settings from config
        openai_settings = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': os.getenv('OPENAI_MODEL', 'gpt-4'),
            'max_retries': config.github.max_retries,
            'timeout': 30.0,
            'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
            'top_p': float(os.getenv('OPENAI_TOP_P', '1.0')),
            'frequency_penalty': float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0')),
            'presence_penalty': float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0'))
        }
        api_base = os.getenv('OPENAI_BASE_URL')
        if api_base:
            openai_settings['base_url'] = api_base

        llm_client = OpenAIClient(**openai_settings)

        # Create and run orchestrator
        analyzer = GitHubIssuesAnalyzer(
            llm_client=llm_client,
            github_token=config.github.token,
            checkpoint_handler=checkpoint_mgr
        )
        orchestrator = AnalysisOrchestrator(analyzer, checkpoint_mgr)
        orchestrator.setup_signal_handlers()

        results = await orchestrator.run_analysis(
            repo_name=args.repo,
            num_issues=args.issues,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            input_csv=args.input_csv,
            use_mongodb=args.use_mongodb
        )

        print(
            f"Analysis completed successfully. Results saved for repository: {results.metadata.repository}")
        print(
            f"JSON results saved to: {os.getenv('EXPORT_DIR', 'exports')}/json/")
        if os.getenv('CSV_EXPORT', 'true').lower() == 'true':
            print(
                f"CSV results saved to: {os.getenv('EXPORT_DIR', 'exports')}/csv/")
        if os.getenv('MONGODB_ENABLED', 'true').lower() == 'true':
            print("Results also saved to MongoDB")

    except Exception as exc:
        logger.error("Analysis failed: {}".format(str(exc)))
        exit(1)


def main():
    """Main entry point for the analysis script."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
