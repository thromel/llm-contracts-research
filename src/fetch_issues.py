"""Script to fetch GitHub issues."""
import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

from core.config import load_config, AppConfig
from core.repositories import RepositoryFactory
from core.github_fetcher import GitHubFetcher
from core.clients.github_api_client import GitHubAPIClient, GitHubConfig
from core.progress import TqdmProgressTracker
from core.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce logging verbosity
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('backoff').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch GitHub issues for specified repositories')
    parser.add_argument(
        'config',
        nargs='?',  # Make positional argument optional
        default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--config',
        dest='config_opt',
        help='Path to YAML config file (alternative to positional argument)'
    )
    args = parser.parse_args()

    # Use either positional or --config argument
    config_path = args.config_opt if args.config_opt else args.config
    return config_path


async def main():
    """Main entry point."""
    repository = None
    try:
        # Parse command line arguments
        config_path = parse_args()

        # Load configuration
        try:
            config = load_config(config_path)
        except ValueError as e:
            logger.error(str(e))
            print(f"\nError: {str(e)}")
            print("\nPlease either:")
            print("1. Set required environment variables in .env:")
            print("   REPOSITORIES=owner1/repo1,owner2/repo2")
            print("   GITHUB_TOKEN=your_token")
            print("\n2. Or provide a config.yaml file:")
            print("   python src/fetch_issues.py config.yaml")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            print(f"\nError: Config file not found: {config_path}")
            sys.exit(1)

        # Set logging level from config
        logging.getLogger().setLevel(config.log_level)

        # Log configuration summary
        logger.info("Starting GitHub Issues Fetcher with configuration:")
        logger.info(f"Repositories: {', '.join(config.repositories)}")
        logger.info(f"GitHub API URL: {config.github.api_url}")
        logger.info(
            f"Max issues per repo: {config.max_issues_per_repo or 'unlimited'}")
        logger.info(f"Since days: {config.since_days or 'all'}")
        logger.info(
            f"Checkpointing: {'enabled' if config.checkpoint_enabled else 'disabled'}")

        # Initialize GitHub API client
        github_config = GitHubConfig(
            api_url=config.github.api_url,
            token=config.github.token,
            max_retries=config.github.max_retries,
            per_page=config.github.per_page
        )
        api_client = GitHubAPIClient(github_config)

        # Initialize repository
        repository = RepositoryFactory.create_repository(config)
        await repository.connect()

        # Initialize progress tracker and checkpoint manager
        progress_tracker = TqdmProgressTracker() if config.show_progress else None
        checkpoint_manager = CheckpointManager(
            config.checkpoint_dir) if config.checkpoint_enabled else None

        # Calculate since date if needed
        since_date = (
            datetime.now(timezone.utc) - timedelta(days=config.since_days)
            if config.since_days
            else None
        )

        # Create fetcher
        fetcher = GitHubFetcher(
            api_client=api_client,
            repository=repository,
            progress_tracker=progress_tracker,
            checkpoint_manager=checkpoint_manager
        )

        # Fetch issues for each repository
        async with api_client:
            for repo_spec in config.repositories:
                try:
                    owner, repo = repo_spec.split('/')
                    logger.info(f"Starting to fetch issues for {repo_spec}")

                    async for _ in fetcher.fetch_repository_issues(
                        owner=owner,
                        repo=repo,
                        since=since_date,
                        max_issues=config.max_issues_per_repo
                    ):
                        # Issues are automatically saved by the fetcher
                        pass

                    if progress_tracker:
                        progress_tracker.complete()
                    logger.info(f"Completed fetching issues for {repo_spec}")

                except Exception as e:
                    logger.error(
                        f"Error processing repository {repo_spec}: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

    finally:
        # Cleanup
        if repository:
            await repository.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
