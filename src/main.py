"""Main application entry point."""
import asyncio
from datetime import datetime, timedelta
from tqdm import tqdm

from core.config import AppConfig
from core.repositories import RepositoryFactory
from core.github_fetcher import GitHubFetcher
from analysis.core.analyzers import GitHubIssuesAnalyzer
from analysis.core.clients import LLMClient
from analysis.core.checkpoint import CheckpointManager
from analysis.core.progress import ConsoleProgressTracker

config = AppConfig()


async def main():
    """Main application entry point."""
    try:
        # Initialize repository
        repository = RepositoryFactory.create_repository(config)
        await repository.connect()

        # Initialize components
        llm_client = LLMClient(config.llm.api_key)
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        progress_tracker = ConsoleProgressTracker()

        # Create fetcher and analyzer
        fetcher = GitHubFetcher(repository)
        analyzer = GitHubIssuesAnalyzer.create(
            llm_client=llm_client,
            github_token=config.github.token,
            checkpoint_handler=checkpoint_manager,
            progress_tracker=progress_tracker
        )

        # Calculate since date if specified
        since_date = None
        if config.since_days:
            since_date = datetime.utcnow() - timedelta(days=config.since_days)

        # Create progress bar for repositories
        with tqdm(total=len(config.repositories), desc="Repositories", unit="repo") as pbar:
            # Fetch issues
            await fetcher.fetch_repositories(
                repo_list=config.repositories,
                progress_bar=pbar,
                checkpoint_manager=checkpoint_manager,
                since_date=since_date,
                max_issues=config.max_issues_per_repo,
                include_closed=config.include_closed
            )

        # Analyze repositories
        for repo in config.repositories:
            try:
                results = await analyzer.analyze_repository(
                    repo_name=repo,
                    num_issues=config.max_issues_per_repo or 100,
                    checkpoint_interval=config.checkpoint_interval
                )
                print(f"\nAnalysis completed for {repo}:")
                print(f"Total issues: {results.total_issues}")
                print(f"Analyzed issues: {results.completed_issues}")
            except Exception as e:
                print(f"\nError analyzing {repo}: {str(e)}")
                continue

    except Exception as e:
        print(f"Application error: {str(e)}")
        raise

    finally:
        # Cleanup
        if repository:
            await repository.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
