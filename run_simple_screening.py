#!/usr/bin/env python3
"""
Simple LLM Screening Pipeline Runner

This script runs the LLM screening pipeline with real services:
- MongoDB Atlas (if configured)
- DeepSeek/OpenAI APIs (if keys provided)
- Real data acquisition (if enabled)

Falls back gracefully to mock mode if external services unavailable.
"""

from pipeline.llm_screening.bulk_screener import BulkScreener
from pipeline.llm_screening.screening_orchestrator import ScreeningOrchestrator
from pipeline.common.database import MongoDBManager
from pipeline.common.config import get_development_config, ScreeningMode
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Add the pipeline to the path
sys.path.insert(0, '.')


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleScreeningRunner:
    """Simple screening pipeline runner with fallback capabilities."""

    def __init__(self):
        self.config = None
        self.db = None
        self.orchestrator = None
        self.use_mock_mode = False

    async def initialize(self) -> bool:
        """Initialize the pipeline components."""
        logger.info("🔧 Initializing LLM screening pipeline...")

        try:
            # Load configuration
            self.config = get_development_config()
            logger.info(
                f"✅ Configuration loaded: {self.config.screening_mode}")

            # Check if we have required environment variables
            mongodb_uri = os.getenv('MONGODB_URI')
            openai_key = os.getenv('OPENAI_API_KEY')
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')

            if not mongodb_uri:
                logger.warning("⚠️ MONGODB_URI not found in environment")
                self.use_mock_mode = True

            if not (openai_key or deepseek_key):
                logger.warning("⚠️ No API keys found in environment")
                self.use_mock_mode = True

            if self.use_mock_mode:
                logger.info(
                    "🔄 Running in MOCK MODE - no external services required")
                return await self._initialize_mock_mode()
            else:
                logger.info(
                    "🌐 Running in PRODUCTION MODE - connecting to external services")
                return await self._initialize_production_mode()

        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False

    async def _initialize_mock_mode(self) -> bool:
        """Initialize in mock mode for testing."""
        from test_pipeline_e2e import MockDatabaseManager, MockLLMScreener

        # Use mock database
        self.db = MockDatabaseManager()
        await self.db.connect()

        # Setup some mock data for testing
        await self._setup_mock_data()

        logger.info("✅ Mock mode initialized")
        return True

    async def _initialize_production_mode(self) -> bool:
        """Initialize with real services."""
        try:
            # Initialize MongoDB
            mongodb_uri = os.getenv('MONGODB_URI')
            self.db = MongoDBManager(mongodb_uri)
            await self.db.connect()
            logger.info("✅ MongoDB connected")

            # Initialize screening orchestrator
            self.orchestrator = ScreeningOrchestrator(self.config, self.db)
            logger.info("✅ Screening orchestrator initialized")

            return True

        except Exception as e:
            logger.error(f"❌ Production mode initialization failed: {str(e)}")
            logger.info("🔄 Falling back to mock mode...")
            self.use_mock_mode = True
            return await self._initialize_mock_mode()

    async def _setup_mock_data(self):
        """Setup mock data for testing."""
        from test_pipeline_e2e import create_mock_raw_posts, simulate_keyword_filtering

        raw_posts = create_mock_raw_posts()
        await simulate_keyword_filtering(self.db, raw_posts)
        logger.info(f"📝 Setup {len(raw_posts)} mock posts for testing")

    async def run_screening(self, max_posts: int = 10) -> Dict[str, Any]:
        """Run the screening pipeline."""
        logger.info(f"🚀 Starting LLM screening (max {max_posts} posts)")

        start_time = datetime.now()

        try:
            if self.use_mock_mode:
                return await self._run_mock_screening(max_posts)
            else:
                return await self._run_production_screening(max_posts)

        except Exception as e:
            logger.error(f"❌ Screening failed: {str(e)}")
            return {'error': str(e), 'processed': 0}

    async def _run_mock_screening(self, max_posts: int) -> Dict[str, Any]:
        """Run screening in mock mode."""
        from test_pipeline_e2e import MockLLMScreener

        screener = MockLLMScreener(self.db)
        results = await screener.screen_batch(max_posts)

        logger.info("✅ Mock screening completed")
        return results

    async def _run_production_screening(self, max_posts: int) -> Dict[str, Any]:
        """Run screening with real services."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        results = await self.orchestrator.run_screening_pipeline(
            max_posts=max_posts,
            skip_validation=False
        )

        logger.info("✅ Production screening completed")
        return results

    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        status = {
            'mode': 'mock' if self.use_mock_mode else 'production',
            'initialized': self.db is not None,
            'timestamp': datetime.now().isoformat()
        }

        if self.use_mock_mode and hasattr(self.db, 'get_stats'):
            status['mock_stats'] = self.db.get_stats()
        elif not self.use_mock_mode and self.orchestrator:
            try:
                status['production_stats'] = await self.orchestrator.get_screening_status()
            except Exception as e:
                status['production_stats'] = {'error': str(e)}

        return status

    async def shutdown(self):
        """Cleanup and shutdown."""
        logger.info("🛑 Shutting down pipeline...")

        try:
            if self.db and hasattr(self.db, 'disconnect'):
                await self.db.disconnect()

            if self.orchestrator and hasattr(self.orchestrator, 'cleanup_and_shutdown'):
                await self.orchestrator.cleanup_and_shutdown()

            logger.info("✅ Shutdown completed")

        except Exception as e:
            logger.error(f"⚠️ Shutdown error: {str(e)}")


async def main():
    """Main entry point."""
    logger.info("🎯 Simple LLM Screening Pipeline")
    logger.info("=" * 50)

    # Check for environment variables
    logger.info("🔍 Environment Check:")
    logger.info(
        f"   MONGODB_URI: {'✅ Set' if os.getenv('MONGODB_URI') else '❌ Not set'}")
    logger.info(
        f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    logger.info(
        f"   DEEPSEEK_API_KEY: {'✅ Set' if os.getenv('DEEPSEEK_API_KEY') else '❌ Not set'}")

    runner = SimpleScreeningRunner()

    try:
        # Initialize
        if not await runner.initialize():
            logger.error("❌ Failed to initialize pipeline")
            return

        # Get initial status
        status = await runner.get_status()
        logger.info(f"📊 Pipeline Status: {status}")

        # Run screening
        logger.info("\n" + "=" * 50)
        logger.info("🤖 Running LLM Screening")
        logger.info("=" * 50)

        results = await runner.run_screening(max_posts=5)

        # Display results
        logger.info("\n📋 Screening Results:")
        for key, value in results.items():
            if key != 'error':
                logger.info(f"   {key}: {value}")

        if 'error' in results:
            logger.error(f"❌ Error: {results['error']}")
        else:
            logger.info("🎉 Screening completed successfully!")

        # Final status
        final_status = await runner.get_status()
        if 'mock_stats' in final_status:
            stats = final_status['mock_stats']
            logger.info(f"\n📈 Final Statistics:")
            logger.info(f"   Raw Posts: {stats.get('raw_posts', 0)}")
            logger.info(f"   Filtered Posts: {stats.get('filtered_posts', 0)}")
            logger.info(f"   Passed Filter: {stats.get('passed_filter', 0)}")
            logger.info(f"   LLM Screened: {stats.get('screened', 0)}")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    # Simple command line interface
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple LLM Screening Pipeline")
    parser.add_argument('--max-posts', type=int, default=10,
                        help='Maximum posts to process')
    parser.add_argument('--mock', action='store_true', help='Force mock mode')

    args = parser.parse_args()

    # Force mock mode if requested
    if args.mock:
        os.environ.pop('MONGODB_URI', None)
        os.environ.pop('OPENAI_API_KEY', None)
        os.environ.pop('DEEPSEEK_API_KEY', None)

    asyncio.run(main())
