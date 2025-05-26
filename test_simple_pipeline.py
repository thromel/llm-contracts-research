#!/usr/bin/env python3
"""
Simple test script for the LLM screening pipeline.

This will test the basic components and help identify any issues.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the pipeline to the path
sys.path.insert(0, '.')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_imports():
    """Test that all basic imports work."""
    logger.info("🔍 Testing imports...")

    try:
        # Test basic imports
        from pipeline.common.config import PipelineConfig, get_development_config
        logger.info("✅ Config imports successful")

        from pipeline.common.database import MongoDBManager
        logger.info("✅ Database imports successful")

        from pipeline.llm_screening.bulk_screener import BulkScreener
        logger.info("✅ Bulk screener imports successful")

        from pipeline.llm_screening.borderline_screener import BorderlineScreener
        logger.info("✅ Borderline screener imports successful")

        from pipeline.llm_screening.screening_orchestrator import ScreeningOrchestrator
        logger.info("✅ Screening orchestrator imports successful")

        from pipeline.preprocessing.keyword_filter import KeywordPreFilter
        logger.info("✅ Keyword filter imports successful")

        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_config():
    """Test configuration loading."""
    logger.info("⚙️ Testing configuration...")

    try:
        from pipeline.common.config import get_development_config, ScreeningMode, LLMProvider

        config = get_development_config()
        logger.info(f"✅ Config loaded: {config.screening_mode}")

        # Test config properties
        logger.info(f"Database: {config.database.database_name}")
        logger.info(f"Screening mode: {config.screening_mode}")

        return True

    except Exception as e:
        logger.error(f"❌ Config test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_models():
    """Test data models."""
    logger.info("📋 Testing data models...")

    try:
        from pipeline.common.models import RawPost, FilteredPost, LLMScreeningResult, Platform

        # Test creating a raw post
        raw_post = RawPost(
            platform=Platform.GITHUB,
            source_id="test-123",
            url="https://github.com/test/repo/issues/1",
            title="Test Issue",
            body_md="Test body content",
            created_at=datetime.utcnow(),
            score=5,
            tags=["test"],
            author="test_user",
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

        logger.info(f"✅ Raw post created: {raw_post.title}")

        # Test creating a screening result
        screening_result = LLMScreeningResult(
            decision="relevant",
            confidence=0.85,
            rationale="Contains API usage patterns",
            model_used="mock-model"
        )

        logger.info(f"✅ Screening result created: {screening_result.decision}")

        return True

    except Exception as e:
        logger.error(f"❌ Models test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_screener_initialization():
    """Test screener initialization without API calls."""
    logger.info("🤖 Testing screener initialization...")

    try:
        from pipeline.llm_screening.bulk_screener import BulkScreener
        from pipeline.common.config import get_development_config

        config = get_development_config()

        # Mock database manager for testing
        class MockDBManager:
            async def connect(self):
                pass

            async def disconnect(self):
                pass

        mock_db = MockDBManager()

        # Test bulk screener initialization
        bulk_screener = BulkScreener(
            api_key="mock-key-12345",
            db_manager=mock_db,
            base_url="http://localhost:8000/v1",
            model="mock-model"
        )

        logger.info("✅ Bulk screener initialized")

        # Test borderline screener
        from pipeline.llm_screening.borderline_screener import BorderlineScreener

        borderline_screener = BorderlineScreener(
            api_key="mock-key-12345",
            db_manager=mock_db,
            model="mock-model"
        )

        logger.info("✅ Borderline screener initialized")

        return True

    except Exception as e:
        logger.error(f"❌ Screener initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_agentic_imports():
    """Test agentic screener imports."""
    logger.info("🤖 Testing agentic imports...")

    try:
        from pipeline.llm_screening.agentic_screener import (
            AgenticScreeningOrchestrator,
            ContractViolationAnalysis,
            TechnicalErrorAnalysis,
            ContextRelevanceAnalysis,
            FinalDecision
        )

        logger.info("✅ Agentic screener imports successful")

        # Test creating analysis objects
        violation_analysis = ContractViolationAnalysis(
            has_violation=True,
            violation_type="rate_limit",
            confidence=0.9,
            evidence=["Rate limit error message"],
            violation_severity="high"
        )

        logger.info(
            f"✅ Violation analysis created: {violation_analysis.violation_type}")

        return True

    except Exception as e:
        logger.error(f"❌ Agentic imports failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_keyword_filter():
    """Test keyword filter functionality."""
    logger.info("🔍 Testing keyword filter...")

    try:
        from pipeline.preprocessing.keyword_filter import KeywordPreFilter

        # Mock database manager
        class MockDBManager:
            async def connect(self):
                pass

            async def disconnect(self):
                pass

        mock_db = MockDBManager()

        # Initialize keyword filter
        keyword_filter = KeywordPreFilter(mock_db)

        # Test keyword matching
        test_text = "I'm getting a rate limit error when calling the OpenAI API"

        # This would normally check against keywords, but we'll just test initialization
        logger.info("✅ Keyword filter initialized")

        return True

    except Exception as e:
        logger.error(f"❌ Keyword filter test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting LLM Screening Pipeline Component Tests")

    tests = [
        ("Basic Imports", test_imports),
        ("Configuration", test_config),
        ("Data Models", test_models),
        ("Screener Initialization", test_screener_initialization),
        ("Agentic Imports", test_agentic_imports),
        ("Keyword Filter", test_keyword_filter),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {str(e)}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info(
            "🎉 All component tests passed! Basic pipeline structure is working.")
        logger.info("💡 Next steps:")
        logger.info("   1. Set up MongoDB connection")
        logger.info("   2. Configure API keys")
        logger.info("   3. Test with real data")
    else:
        logger.info("⚠️ Some tests failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())
