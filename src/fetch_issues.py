import asyncio
import logging
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import yaml
from tqdm.asyncio import tqdm
from core.config import config
from core.db.mongo_adapter import MongoAdapter
from core.github_fetcher import GitHubFetcher
from core.analysis.core.checkpoint import CheckpointManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch GitHub issues from repositories.')
    parser.add_argument(
        'config_file', help='Path to repositories configuration file')
    parser.add_argument('--since-days', type=int,
                        help='Only fetch issues updated in the last N days')
    parser.add_argument('--max-issues', type=int,
                        help='Maximum number of issues to fetch per repository')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpointing')
    parser.add_argument('--checkpoint-dir',
                        help='Directory for checkpoint files')
    parser.add_argument('--export-dir', help='Directory for exported data')
    parser.add_argument(
        '--export-format', help='Comma-separated list of export formats (json,csv)')
    parser.add_argument('--batch-size', type=int,
                        help='Number of issues to process in each batch')
    parser.add_argument('--include-closed',
                        action='store_true', help='Include closed issues')
    parser.add_argument('--no-comments',
                        action='store_true', help='Skip fetching comments')
    parser.add_argument('--max-comments', type=int,
                        help='Maximum number of comments to fetch per issue')

    args = parser.parse_args()

    # Update config with command line arguments
    if args.since_days is not None:
        config.since_days = args.since_days
    if args.max_issues is not None:
        config.max_issues_per_repo = args.max_issues
    if args.no_progress:
        config.show_progress = False
    if args.no_checkpoint:
        config.checkpoint_enabled = False
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.export_dir:
        config.export_dir = args.export_dir
    if args.export_format:
        config.export_format = args.export_format.split(',')
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.include_closed:
        config.include_closed = True
    if args.no_comments:
        config.include_comments = False
    if args.max_comments is not None:
        config.max_comments_per_issue = args.max_comments

    return args


def get_db_adapter():
    """Get the appropriate database adapter based on configuration."""
    if config.db_provider == 'mongo':
        return MongoAdapter(config.mongo)
    else:
        raise ValueError(
            f"Unsupported database provider: {config.db_provider}")


async def main(config_path: str):
    """Main function to fetch GitHub issues."""
    try:
        # Load repository list from config file
        with open(config_path, 'r') as f:
            repo_config = yaml.safe_load(f)

        if not isinstance(repo_config, dict) or 'repositories' not in repo_config:
            raise ValueError("Config file must contain a 'repositories' list")

        repo_list: List[str] = repo_config['repositories']

        # Validate repository format
        for repo in repo_list:
            if '/' not in repo or len(repo.split('/')) != 2:
                raise ValueError(
                    f"Invalid repository format: {repo}. Expected format: owner/repo")

        # Initialize checkpoint manager if enabled
        checkpoint_manager = None
        if config.checkpoint_enabled:
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                logger.info(
                    f"Resuming from checkpoint at index {checkpoint_data['current_index']}")
                # TODO: Implement checkpoint restoration logic

        # Initialize database adapter
        db = get_db_adapter()
        await db.connect()

        try:
            # Initialize and run GitHub fetcher
            async with GitHubFetcher(db) as fetcher:
                # Create progress bar if enabled
                if config.show_progress:
                    pbar = tqdm(total=len(repo_list),
                                desc="Repositories", unit="repo")
                else:
                    pbar = None

                # Calculate since date if specified
                since_date = None
                if config.since_days:
                    since_date = datetime.utcnow() - timedelta(days=config.since_days)

                await fetcher.fetch_repositories(
                    repo_list,
                    progress_bar=pbar,
                    checkpoint_manager=checkpoint_manager,
                    since_date=since_date,
                    max_issues=config.max_issues_per_repo,
                    include_closed=config.include_closed
                )

                if pbar:
                    pbar.close()

        finally:
            await db.disconnect()

        logger.info("Successfully completed fetching issues")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.config_file))
