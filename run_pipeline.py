#!/usr/bin/env python3
"""
Multi-Source Data Pipeline Runner

Fetches data from GitHub and Stack Overflow APIs and runs the complete
LLM screening pipeline with configurable sources and independent step execution.
"""

from pipeline.data_acquisition.github import GitHubAcquisition
from pipeline.data_acquisition.stackoverflow import StackOverflowAcquisition
from pipeline.preprocessing.keyword_filter import KeywordPreFilter
from pipeline.llm_screening.screening_orchestrator import ScreeningOrchestrator
from pipeline.common.database import MongoDBManager
from pipeline.common.config import get_development_config
from pipeline.common.models import RawPost, Platform
import asyncio
import logging
import os
import yaml
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiSourcePipelineRunner:
    """Pipeline runner for GitHub and Stack Overflow data with step-by-step execution."""

    def __init__(self, config_file: str = "pipeline_config.yaml"):
        self.config_file = config_file
        self.pipeline_config = self._load_pipeline_config()

        # Apply LLM configuration BEFORE creating other components
        self._apply_llm_config()

        # Now get app config AFTER environment variables are set
        self.app_config = get_development_config()
        self.db = MongoDBManager(os.getenv('MONGODB_URI'))

        # Initialize data acquisition clients
        self.github_client = None
        self.stackoverflow_client = None

        # Initialize processing components
        self.keyword_filter = None
        self.orchestrator = None

    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded pipeline config from {self.config_file}")
                return config
        except FileNotFoundError:
            logger.warning(
                f"Config file {self.config_file} not found, creating default config")
            default_config = self._create_default_config()
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            return default_config

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default pipeline configuration."""
        return {
            'sources': {
                'github': {
                    'enabled': True,
                    'repositories': [
                        {'owner': 'openai', 'repo': 'openai-python'},
                        {'owner': 'anthropics', 'repo': 'anthropic-sdk-python'},
                        {'owner': 'google', 'repo': 'generative-ai-python'}
                    ],
                    'max_issues_per_repo': 50,
                    'days_back': 30
                },
                'stackoverflow': {
                    'enabled': True,
                    'tags': [
                        'openai-api',
                        'langchain'
                    ],
                    'max_questions_per_tag': 100,
                    'days_back': 30
                }
            },
            'deduplication': {
                'enabled': True,
                'similarity_threshold': 0.8,
                'title_weight': 0.6,
                'body_weight': 0.4
            },
            'pipeline_steps': {
                'data_acquisition': True,
                'keyword_filtering': True,
                'llm_screening': True
            },
            'llm_screening': {
                'mode': 'traditional',
                'model': 'gpt-4-turbo-2024-04-09',
                'temperature': 0.1,
                'max_tokens': 2000,
                'provider': 'openai'
            }
        }

    def _apply_llm_config(self):
        """Apply LLM configuration from pipeline config to environment variables."""
        llm_config = self.pipeline_config.get('llm_screening', {})

        # Set screening mode
        screening_mode = llm_config.get('mode', 'traditional')
        os.environ['SCREENING_MODE'] = screening_mode
        logger.info(f"Set screening mode: {screening_mode}")

        # Set OpenAI configuration for traditional screening
        if llm_config.get('provider') == 'openai':
            model = llm_config.get('model', 'gpt-4-turbo-2024-04-09')
            temperature = str(llm_config.get('temperature', 0.1))
            max_tokens = str(llm_config.get('max_tokens', 2000))

            os.environ['OPENAI_MODEL'] = model
            os.environ['OPENAI_TEMPERATURE'] = temperature
            os.environ['OPENAI_MAX_TOKENS'] = max_tokens

            logger.info(f"Set OpenAI model: {model}")
            logger.info(f"Set temperature: {temperature}")
            logger.info(f"Set max_tokens: {max_tokens}")

    async def initialize(self):
        """Initialize database and pipeline components."""
        await self.db.connect()

        # Initialize data acquisition clients
        if self.pipeline_config['sources']['github']['enabled']:
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                self.github_client = GitHubAcquisition(github_token, self.db)
                logger.info("GitHub client initialized")
            else:
                logger.warning(
                    "GITHUB_TOKEN not found, GitHub acquisition disabled")

        if self.pipeline_config['sources']['stackoverflow']['enabled']:
            # Optional - increases rate limits
            so_key = os.getenv('STACKOVERFLOW_API_KEY')
            self.stackoverflow_client = StackOverflowAcquisition(
                self.db, so_key)
            if so_key:
                logger.info(
                    "Stack Overflow client initialized with API key (10k requests/day)")
            else:
                logger.info(
                    "Stack Overflow client initialized without API key (300 requests/day)")
                logger.warning(
                    "Consider setting STACKOVERFLOW_API_KEY for higher rate limits")

        # Initialize processing components
        self.keyword_filter = KeywordPreFilter(self.db)
        self.orchestrator = ScreeningOrchestrator(self.app_config, self.db)

        logger.info("Pipeline initialized successfully")

    def _generate_content_hash(self, title: str, body: str) -> str:
        """Generate a hash for deduplication based on title and body content."""
        # Normalize content for comparison
        normalized_title = title.lower().strip()
        normalized_body = body.lower().strip()

        # Create hash from combined content
        content = f"{normalized_title}|{normalized_body}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _is_duplicate_post(self, content_hash: str) -> bool:
        """Check if a post with this content hash already exists."""
        existing = await self.db.find_one('raw_posts', {'content_hash': content_hash})
        return existing is not None

    async def step_data_acquisition(self) -> List[RawPost]:
        """Step 1: Acquire data from configured sources."""
        logger.info("=== STEP 1: Data Acquisition ===")

        all_posts = []
        duplicates_skipped = 0

        # GitHub data acquisition
        if self.github_client:
            github_config = self.pipeline_config['sources']['github']
            for repo_config in github_config['repositories']:
                logger.info(
                    f"Fetching from {repo_config['owner']}/{repo_config['repo']}")

                since = datetime.now().replace(
                    day=max(1, datetime.now().day - github_config['days_back']))

                async for raw_post in self.github_client._fetch_repository_issues(
                    owner=repo_config['owner'],
                    repo=repo_config['repo'],
                    since=since,
                    max_issues=github_config['max_issues_per_repo']
                ):
                    # Check for duplicates
                    content_hash = self._generate_content_hash(
                        raw_post.title, raw_post.body_md)

                    if await self._is_duplicate_post(content_hash):
                        duplicates_skipped += 1
                        continue

                    # Add content hash to post
                    raw_post.content_hash = content_hash

                    # Save to database
                    await self.github_client.save_to_database(raw_post)
                    all_posts.append(raw_post)

        # Stack Overflow data acquisition
        if self.stackoverflow_client:
            so_config = self.pipeline_config['sources']['stackoverflow']
            for tag in so_config['tags']:
                logger.info(
                    f"Fetching Stack Overflow questions for tag: {tag}")

                async for raw_post in self.stackoverflow_client.acquire_tagged_questions(
                    tags=[tag],
                    since_days=so_config['days_back'],
                    max_questions=so_config['max_questions_per_tag'],
                    include_answers=False  # Answers will be merged into question body
                ):
                    # Check for duplicates
                    content_hash = self._generate_content_hash(
                        raw_post.title, raw_post.body_md)

                    if await self._is_duplicate_post(content_hash):
                        duplicates_skipped += 1
                        continue

                    # Add content hash to post
                    raw_post.content_hash = content_hash

                    # Save to database
                    await self.stackoverflow_client.save_to_database(raw_post)
                    all_posts.append(raw_post)

        logger.info(
            f"Data acquisition complete: {len(all_posts)} new posts, {duplicates_skipped} duplicates skipped")
        return all_posts

    async def step_keyword_filtering(self, raw_posts: Optional[List[RawPost]] = None) -> List[Dict[str, Any]]:
        """Step 2: Apply keyword filtering to raw posts."""
        logger.info("=== STEP 2: Keyword Filtering ===")

        if raw_posts is None:
            # Load unfiltered posts from database
            raw_posts = []
            async for post_doc in self.db.find_many('raw_posts', {'filtered': {'$ne': True}}):
                # Convert back to RawPost object from dictionary
                try:
                    # Convert MongoDB document back to RawPost object
                    raw_post_obj = RawPost.from_dict(post_doc)
                    raw_posts.append(raw_post_obj)
                except Exception as e:
                    logger.error(f"Error converting post to RawPost: {e}")
                    # Skip this post if conversion fails
                    continue

        filtered_posts = []
        passed_count = 0

        for i, raw_post in enumerate(raw_posts):
            # Apply keyword filter
            filter_result = self.keyword_filter.apply_filter(raw_post)

            # Handle both RawPost objects and dict objects
            if hasattr(raw_post, '_id'):
                # RawPost object
                post_id = raw_post._id or f"post_{i}"
                content_hash = raw_post.content_hash
            else:
                # Dict object from database
                post_id = raw_post.get('_id', f"post_{i}")
                content_hash = raw_post.get('content_hash')

            filtered_post = {
                '_id': f"filtered_{post_id}",
                'raw_post_id': post_id,
                'content_hash': content_hash,
                'passed_keyword_filter': filter_result.passed,
                'filter_confidence': filter_result.confidence,
                'matched_keywords': filter_result.matched_keywords,
                'filter_timestamp': datetime.utcnow(),
                'llm_screened': False
            }

            await self.db.insert_one('filtered_posts', filtered_post)

            # Mark raw post as filtered
            await self.db.update_one('raw_posts',
                                     {'_id': post_id},
                                     {'$set': {'filtered': True}})

            filtered_posts.append(filtered_post)

            if filter_result.passed:
                passed_count += 1

        logger.info(
            f"Keyword filtering complete: {passed_count}/{len(raw_posts)} posts passed")
        return filtered_posts

    async def step_llm_screening(self, max_posts: Optional[int] = None) -> Dict[str, Any]:
        """Step 3: Run LLM screening on filtered posts."""
        logger.info("=== STEP 3: LLM Screening ===")

        # Check for posts that need LLM screening and haven't been screened yet
        unscreened_posts = []
        async for post in self.db.find_many('filtered_posts', {
            'passed_keyword_filter': True,
            'llm_screened': False
        }):
            unscreened_posts.append(post)

        unscreened_count = await self.db.count_documents('filtered_posts', {
            'passed_keyword_filter': True,
            'llm_screened': False
        })

        if unscreened_count == 0:
            logger.info("No posts need LLM screening")
            return {'processed': 0, 'message': 'No posts to screen'}

        # Check for posts that have already been screened (avoid duplicate LLM calls)
        screened_hashes = set()
        async for result in self.db.find_many('llm_screening_results', {}):
            if 'content_hash' in result:
                screened_hashes.add(result['content_hash'])

        logger.info(
            f"Found {len(screened_hashes)} posts already screened, will skip duplicates")

        # Filter out already screened posts by content hash
        posts_to_screen = []
        duplicates_skipped = 0

        for post in unscreened_posts:
            content_hash = post.get('content_hash')
            if content_hash in screened_hashes:
                duplicates_skipped += 1
                # Mark as screened even though we're skipping (to avoid reprocessing)
                await self.db.update_one('filtered_posts',
                                         {'_id': post['_id']},
                                         {'$set': {'llm_screened': True, 'skipped_duplicate': True}})
            else:
                posts_to_screen.append(post)

        logger.info(
            f"LLM screening: {len(posts_to_screen)} posts to screen, {duplicates_skipped} duplicates skipped")

        if max_posts:
            posts_to_screen = posts_to_screen[:max_posts]

        if not posts_to_screen:
            return {'processed': 0, 'duplicates_skipped': duplicates_skipped}

        # Run LLM screening
        return await self.orchestrator.run_screening_pipeline(
            max_posts=len(posts_to_screen),
            skip_validation=False
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            'raw_posts': await self.db.count_documents('raw_posts', {}),
            'raw_posts_github': await self.db.count_documents('raw_posts', {'platform': 'github'}),
            'raw_posts_stackoverflow': await self.db.count_documents('raw_posts', {'platform': 'stackoverflow'}),
            'filtered_posts': await self.db.count_documents('filtered_posts', {}),
            'passed_filter': await self.db.count_documents('filtered_posts', {'passed_keyword_filter': True}),
            'screened': await self.db.count_documents('filtered_posts', {'llm_screened': True}),
            'screening_results': await self.db.count_documents('llm_screening_results', {}),
            'duplicates_by_hash': await self.db.count_documents('filtered_posts', {'skipped_duplicate': True})
        }
        return stats

    async def run_full_pipeline(self):
        """Run the complete pipeline end-to-end."""
        logger.info("=== RUNNING FULL PIPELINE ===")

        # Step 1: Data Acquisition
        if self.pipeline_config['pipeline_steps']['data_acquisition']:
            raw_posts = await self.step_data_acquisition()
        else:
            raw_posts = None

        # Step 2: Keyword Filtering
        if self.pipeline_config['pipeline_steps']['keyword_filtering']:
            filtered_posts = await self.step_keyword_filtering(raw_posts)

        # Step 3: LLM Screening
        if self.pipeline_config['pipeline_steps']['llm_screening']:
            screening_results = await self.step_llm_screening()
            logger.info(f"LLM screening results: {screening_results}")

        # Final stats
        stats = await self.get_stats()
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info("Final Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    async def shutdown(self):
        """Cleanup and shutdown."""
        await self.db.disconnect()
        if self.orchestrator:
            await self.orchestrator.cleanup_and_shutdown()


async def main():
    """Main pipeline execution with CLI arguments."""
    parser = argparse.ArgumentParser(description="Multi-Source Data Pipeline")
    parser.add_argument('--step', choices=['acquisition', 'filtering', 'screening', 'full'],
                        default='full', help='Pipeline step to run')
    parser.add_argument('--config', default='pipeline_config.yaml',
                        help='Path to pipeline configuration file')
    parser.add_argument('--max-posts', type=int,
                        help='Maximum posts to process in LLM screening')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics')

    args = parser.parse_args()

    logger.info("Starting Multi-Source Data Pipeline")

    runner = MultiSourcePipelineRunner(config_file=args.config)

    try:
        await runner.initialize()

        if args.stats_only:
            stats = await runner.get_stats()
            logger.info("Current Pipeline Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            return

        if args.step == 'acquisition':
            await runner.step_data_acquisition()
        elif args.step == 'filtering':
            await runner.step_keyword_filtering()
        elif args.step == 'screening':
            await runner.step_llm_screening(max_posts=args.max_posts)
        elif args.step == 'full':
            await runner.run_full_pipeline()

        # Show final stats
        stats = await runner.get_stats()
        logger.info("Pipeline Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
