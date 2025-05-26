"""Main application entry point for multi-source GitHub analysis with MongoDB integration."""
import asyncio
import os
from datetime import datetime, timedelta
from tqdm import tqdm

from core.config import AppConfig, load_config
from core.repositories import RepositoryFactory
from core.github_fetcher import GitHubFetcher
from core.clients.github_api_client import GitHubAPIClient, GitHubConfig
from core.dto.github import RepositoryDTO
from analysis.core.analyzers import GitHubIssuesAnalyzer
from analysis.core.clients import OpenAIClient
from analysis.core.storage.factory import StorageFactory
from core.checkpoint_manager import CheckpointManager
from core.progress import TqdmProgressTracker

# Load configuration from YAML and environment
try:
    config = load_config('config.yaml')
    print(
        f"📋 Loaded configuration for {len(config.repositories)} repositories")
except Exception as e:
    print(
        f"⚠️  Could not load config.yaml, using default configuration: {str(e)}")
    config = AppConfig()


async def main():
    """Main application entry point with MongoDB integration."""
    print("🚀 Starting GitHub Issues Analysis with MongoDB Integration")

    try:
        # Initialize MongoDB repository
        print("📦 Initializing MongoDB repository...")
        repository = RepositoryFactory.create_repository(config)
        await repository.connect()
        print("✅ Connected to MongoDB")

        # Initialize storage for analysis results (fallback to JSON if MongoDB fails)
        try:
            storage = StorageFactory.create_storage(['mongodb', 'json'])
            print("📦 Initialized storage backends: MongoDB + JSON")
        except Exception as e:
            print(f"⚠️  MongoDB storage failed, using JSON only: {str(e)}")
            storage = StorageFactory.create_storage(['json'])

        # Initialize components
        llm_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        progress_tracker = TqdmProgressTracker()
        github_config = GitHubConfig(
            api_url=config.github.api_url,
            token=config.github.token,
            max_retries=config.github.max_retries,
            per_page=config.github.per_page
        )
        github_api_client = GitHubAPIClient(github_config)

        # Calculate since date if specified
        since_date = None
        if config.since_days:
            since_date = datetime.utcnow() - timedelta(days=config.since_days)
            print(
                f"📅 Fetching issues since: {since_date.strftime('%Y-%m-%d')}")

        # Use GitHub API client in async context
        async with github_api_client:
            # Create fetcher and analyzer
            fetcher = GitHubFetcher(
                api_client=github_api_client,
                repository=repository,
                progress_tracker=progress_tracker,
                checkpoint_manager=checkpoint_manager
            )
            analyzer = GitHubIssuesAnalyzer.create(
                llm_client=llm_client,
                github_token=config.github.token,
                checkpoint_handler=checkpoint_manager,
                progress_tracker=progress_tracker
            )

            print(f"🎯 Processing {len(config.repositories)} repositories")

            # Phase 1: Fetch and store repositories and issues
            print("\n📥 PHASE 1: Fetching Issues from GitHub")
            with tqdm(total=len(config.repositories), desc="Repositories", unit="repo") as pbar:
                for repo_full_name in config.repositories:
                    try:
                        # Parse repository format (owner/repo)
                        if '/' not in repo_full_name:
                            print(
                                f"❌ Invalid repository format: {repo_full_name}. Expected 'owner/repo'")
                            pbar.update(1)
                            continue

                        owner, repo_name = repo_full_name.split('/', 1)
                        print(f"\n🔄 Processing repository: {repo_full_name}")

                        # Get or create repository record in MongoDB
                        repo_info = await github_api_client.get_repository(owner, repo_name)
                        repo_dto = RepositoryDTO(
                            github_repo_id=repo_info['id'],
                            name=repo_name,
                            owner=owner,
                            full_name=repo_full_name,
                            url=repo_info['html_url'],
                            created_at=datetime.fromisoformat(
                                repo_info['created_at'].rstrip('Z')),
                            updated_at=datetime.fromisoformat(
                                repo_info['updated_at'].rstrip('Z'))
                        )

                        repo_id = await repository.save_repository(repo_dto)
                        print(f"📝 Repository saved with ID: {repo_id}")

                        # Check if we should fetch incrementally
                        last_fetched = await repository.get_last_issue_timestamp(str(repo_dto.github_repo_id))
                        effective_since = max(since_date, last_fetched) if (
                            since_date and last_fetched) else (since_date or last_fetched)

                        if effective_since:
                            print(
                                f"🔄 Incremental fetch since: {effective_since.strftime('%Y-%m-%d %H:%M:%S')}")

                        # Fetch issues for this repository
                        issue_count = 0
                        async for issue in fetcher.fetch_repository_issues(
                            owner=owner,
                            repo=repo_name,
                            since=effective_since,
                            max_issues=config.max_issues_per_repo
                        ):
                            issue_count += 1
                            # Store issue in MongoDB
                            await repository.save_issue(issue)

                        print(
                            f"✅ Fetched and stored {issue_count} issues from {repo_full_name}")
                        pbar.update(1)

                    except Exception as e:
                        print(f"❌ Error processing {repo_full_name}: {str(e)}")
                        pbar.update(1)
                        continue

            # Phase 2: Analyze stored issues
            print("\n🧠 PHASE 2: Analyzing Issues with LLM")
            for repo_full_name in config.repositories:
                try:
                    # Get repository from database
                    repo_record = await repository.get_repository_by_full_name(repo_full_name)
                    if not repo_record:
                        print(
                            f"⚠️  Repository not found in database: {repo_full_name}")
                        continue

                    print(f"\n🔍 Analyzing repository: {repo_full_name}")

                    # Get issues from MongoDB
                    issues = await repository.get_repository_issues(
                        str(repo_record.github_repo_id),
                        limit=config.max_issues_per_repo or 100
                    )

                    if not issues:
                        print(f"ℹ️  No issues found for {repo_full_name}")
                        continue

                    print(f"📊 Found {len(issues)} issues to analyze")

                    # Analyze issues
                    results = await analyzer.analyze_repository(
                        repo_name=repo_full_name,
                        num_issues=len(issues),
                        checkpoint_interval=config.checkpoint_interval
                    )

                    # Save analysis results to storage backends
                    for storage_backend in storage:
                        try:
                            # Check if save_results is async or sync
                            if hasattr(storage_backend.save_results, '__call__'):
                                if asyncio.iscoroutinefunction(storage_backend.save_results):
                                    await storage_backend.save_results(
                                        analyzed_issues=results.analyzed_issues,
                                        metadata=results.metadata
                                    )
                                else:
                                    storage_backend.save_results(
                                        analyzed_issues=results.analyzed_issues,
                                        metadata=results.metadata
                                    )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to save to {storage_backend.__class__.__name__}: {str(e)}")

                    print(f"✅ Analysis completed for {repo_full_name}:")
                    print(f"   📈 Total issues: {results.total_issues}")
                    print(f"   ✅ Analyzed issues: {results.completed_issues}")
                    # Avoid division by zero
                    if results.total_issues > 0:
                        print(
                            f"   🎯 Success rate: {(results.completed_issues/results.total_issues*100):.1f}%")
                    else:
                        print(f"   🎯 Success rate: N/A (no issues to analyze)")

                except Exception as e:
                    print(f"❌ Error analyzing {repo_full_name}: {str(e)}")
                    continue

        print("\n🎉 Analysis pipeline completed successfully!")

    except Exception as e:
        print(f"💥 Application error: {str(e)}")
        raise

    finally:
        # Cleanup
        print("🧹 Cleaning up connections...")
        if repository:
            await repository.disconnect()
        print("✅ Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
