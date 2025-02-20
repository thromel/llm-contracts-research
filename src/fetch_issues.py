"""Script to fetch GitHub issues."""
import asyncio
import argparse
import logging
from datetime import datetime, timedelta

from core.config import AppConfig, load_config
from core.repositories import MongoDBRepository, RepositoryFactory
from core.github_fetcher import GitHubFetcher
from core.clients.github_api_client import GitHubAPIClient, GitHubConfig
from core.progress import TqdmProgressTracker

# Configure logging to file instead of stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce logging verbosity for some modules
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('backoff').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)


async def main(config_file: str = None):
    """Main entry point."""
    repository = None
    try:
        # Load configuration
        config = load_config(config_file) if config_file else AppConfig()

        # Initialize repository
        repository = RepositoryFactory.create_repository(config)
        await repository.connect()

        # Initialize GitHub API client
        github_config = GitHubConfig(
            api_url=config.github.api_url,
            token=config.github.token,
            max_retries=config.github.max_retries,
            per_page=config.github.per_page
        )
        api_client = GitHubAPIClient(github_config)

        # Initialize progress tracker
        progress_tracker = TqdmProgressTracker()

        # Calculate since date if specified
        since_date = None
        if config.since_days:
            since_date = datetime.now(datetime.UTC) - \
                timedelta(days=config.since_days)

        # Create fetcher
        fetcher = GitHubFetcher(
            api_client=api_client,
            repository=repository,
            progress_tracker=progress_tracker
        )

        # Fetch issues for each repository
        async with api_client:
            for repo_spec in config.repositories:
                try:
                    owner, repo = repo_spec.split('/')
                    logger.debug(f"Starting to fetch issues for {repo_spec}")

                    # Start progress tracking for this repository
                    progress_tracker.start_operation(
                        total=config.max_issues_per_repo or 2000,
                        description=f"Fetching {repo_spec}"
                    )

                    async for issue in fetcher.fetch_repository_issues(
                        owner=owner,
                        repo=repo,
                        since=since_date,
                        max_issues=config.max_issues_per_repo
                    ):
                        # Issues are automatically saved by the fetcher
                        pass

                    progress_tracker.complete()
                    logger.debug(f"Completed fetching issues for {repo_spec}")

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
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issues from repositories"
    )
    parser.add_argument(
        "config_file",
        help="Path to repositories configuration file (YAML)",
        nargs="?"  # Make it optional
    )
    args = parser.parse_args()
    asyncio.run(main(args.config_file))
