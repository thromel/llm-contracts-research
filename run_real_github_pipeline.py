#!/usr/bin/env python3
"""
Real GitHub Pipeline Runner

Fetches real GitHub issues from openai/openai-python and runs the complete
LLM screening pipeline on them.
"""

from pipeline.data_acquisition.github import GitHubAcquisition
from pipeline.preprocessing.keyword_filter import KeywordPreFilter
from pipeline.llm_screening.screening_orchestrator import ScreeningOrchestrator
from pipeline.common.database import MongoDBManager
from pipeline.common.config import get_development_config
from pipeline.common.models import RawPost, Platform
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

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


class GitHubPipelineRunner:
    """Simple runner for the GitHub LLM screening pipeline."""

    def __init__(self):
        self.config = get_development_config()
        self.db = MongoDBManager(os.getenv('MONGODB_URI'))
        self.github_client = GitHubAcquisition(
            os.getenv('GITHUB_TOKEN'), self.db)
        self.keyword_filter = KeywordPreFilter(self.db)
        self.orchestrator = ScreeningOrchestrator(self.config, self.db)

    async def initialize(self):
        """Initialize database connection."""
        await self.db.connect()
        logger.info("Pipeline initialized")

    async def fetch_github_issues(self, max_issues: int = 10) -> List[RawPost]:
        """Fetch real GitHub issues from openai/openai-python."""
        logger.info(f"Fetching {max_issues} GitHub issues...")

        issues = []
        async for raw_post in self.github_client._fetch_repository_issues(
            owner="openai",
            repo="openai-python",
            since=datetime.now().replace(day=1),
            max_issues=max_issues
        ):
            issues.append(raw_post)
            await self.github_client.save_to_database(raw_post)

        logger.info(f"Fetched {len(issues)} issues")
        return issues

    async def run_keyword_filtering(self, raw_posts: List[RawPost]) -> List[Dict[str, Any]]:
        """Apply keyword filtering to raw posts."""
        logger.info("Running keyword filtering...")

        filtered_posts = []
        passed_count = 0

        for i, raw_post in enumerate(raw_posts):
            filter_result = self.keyword_filter.apply_filter(raw_post)

            filtered_post = {
                '_id': i + 1,
                'raw_post_id': getattr(raw_post, '_id', i + 1),
                'passed_keyword_filter': filter_result.passed,
                'filter_confidence': filter_result.confidence,
                'matched_keywords': filter_result.matched_keywords,
                'filter_timestamp': datetime.utcnow(),
                'llm_screened': False
            }

            await self.db.insert_one('filtered_posts', filtered_post)
            filtered_posts.append(filtered_post)

            if filter_result.passed:
                passed_count += 1

        logger.info(
            f"Keyword filtering complete: {passed_count}/{len(raw_posts)} posts passed")
        return filtered_posts

    async def run_llm_screening(self, max_posts: int = 10) -> Dict[str, Any]:
        """Run LLM screening on filtered posts."""
        logger.info("Running LLM screening...")

        return await self.orchestrator.run_screening_pipeline(
            max_posts=max_posts,
            skip_validation=False
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'raw_posts': await self.db.count_documents('raw_posts', {}),
            'filtered_posts': await self.db.count_documents('filtered_posts', {}),
            'passed_filter': await self.db.count_documents('filtered_posts', {'passed_keyword_filter': True}),
            'screened': await self.db.count_documents('filtered_posts', {'llm_screened': True}),
            'screening_results': await self.db.count_documents('llm_screening_results', {})
        }

    async def shutdown(self):
        """Cleanup and shutdown."""
        await self.db.disconnect()
        await self.orchestrator.cleanup_and_shutdown()


async def main():
    """Main pipeline execution."""
    logger.info("Starting GitHub LLM Screening Pipeline")

    runner = GitHubPipelineRunner()

    try:
        await runner.initialize()

        # Fetch GitHub issues
        raw_posts = await runner.fetch_github_issues(max_issues=10)

        # Apply keyword filtering
        filtered_posts = await runner.run_keyword_filtering(raw_posts)
        passed_filter = sum(
            1 for p in filtered_posts if p['passed_keyword_filter'])

        # Run LLM screening
        screening_results = await runner.run_llm_screening(max_posts=passed_filter)

        # Show results
        stats = await runner.get_stats()
        logger.info("Pipeline Results:")
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
