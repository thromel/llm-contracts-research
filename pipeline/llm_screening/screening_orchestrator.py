"""
LLM Screening Orchestrator.

Coordinates different screening approaches based on configuration:
- Traditional: Bulk screening (DeepSeek-R1) + Borderline screening (GPT-4.1)
- Agentic: Multi-agent LangChain pipeline
- Hybrid: Both traditional and agentic approaches
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..common.models import LLMScreeningResult
from ..common.database import MongoDBManager, ProvenanceTracker
from ..common.config import PipelineConfig, ScreeningMode
from .bulk_screener import BulkScreener
from .borderline_screener import BorderlineScreener
from .agentic_screener import AgenticScreeningOrchestrator

logger = logging.getLogger(__name__)


class ScreeningOrchestrator:
    """
    Orchestrates different LLM screening approaches.

    Features:
    - Mode-based screening selection
    - Hybrid approach coordination
    - Performance monitoring
    - Quality validation
    """

    def __init__(self, config: PipelineConfig, db_manager: MongoDBManager):
        """Initialize the screening orchestrator.

        Args:
            config: Pipeline configuration
            db_manager: MongoDB manager
        """
        self.config = config
        self.db = db_manager
        self.provenance = ProvenanceTracker(db_manager)

        # Initialize screeners based on configuration
        self.bulk_screener = None
        self.borderline_screener = None
        self.agentic_screener = None

        self._setup_screeners()

    def _setup_screeners(self):
        """Setup screeners based on configuration."""
        try:
            logger.info(
                f"Setting up screeners for {self.config.screening_mode.value} mode")

            # Traditional screening components
            if self.config.screening_mode in [ScreeningMode.TRADITIONAL, ScreeningMode.HYBRID]:
                if self.config.traditional_screening.bulk_screener_llm:
                    self.bulk_screener = BulkScreener(
                        api_key=self.config.traditional_screening.bulk_screener_llm.api_key,
                        db_manager=self.db,
                        base_url=self.config.traditional_screening.bulk_screener_llm.base_url,
                        model=self.config.traditional_screening.bulk_screener_llm.model_name
                    )
                    logger.info("✅ Bulk screener initialized")

                if self.config.traditional_screening.borderline_screener_llm:
                    self.borderline_screener = BorderlineScreener(
                        api_key=self.config.traditional_screening.borderline_screener_llm.api_key,
                        db_manager=self.db,
                        base_url=self.config.traditional_screening.borderline_screener_llm.base_url,
                        model=self.config.traditional_screening.borderline_screener_llm.model_name
                    )
                    logger.info("✅ Borderline screener initialized")

            # Agentic screening components
            if self.config.screening_mode in [ScreeningMode.AGENTIC, ScreeningMode.HYBRID]:
                logger.info("🔍 Checking agentic screening configuration...")

                if self.config.agentic_screening.contract_detector_llm:
                    logger.info(
                        f"✅ Found contract detector LLM config: {self.config.agentic_screening.contract_detector_llm.model_name}")
                    logger.info(
                        f"🔑 API key present: {bool(self.config.agentic_screening.contract_detector_llm.api_key)}")

                    self.agentic_screener = AgenticScreeningOrchestrator(
                        api_key=self.config.agentic_screening.contract_detector_llm.api_key,
                        model_name=self.config.agentic_screening.contract_detector_llm.model_name,
                        db_manager=self.db,
                        base_url=self.config.agentic_screening.contract_detector_llm.base_url
                    )
                    logger.info("✅ Agentic screener initialized")
                else:
                    logger.warning(
                        "❌ No contract detector LLM configuration found")
                    logger.info(
                        f"Agentic config object: {self.config.agentic_screening}")

        except Exception as e:
            logger.error(f"Error setting up screeners: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def run_screening_pipeline(
        self,
        max_posts: Optional[int] = None,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """Run the complete screening pipeline.

        Args:
            max_posts: Maximum number of posts to process
            skip_validation: Skip API validation checks

        Returns:
            Pipeline execution statistics
        """
        logger.info(
            f"🚀 Starting LLM screening pipeline in {self.config.screening_mode.value} mode")

        start_time = datetime.utcnow()
        pipeline_stats = {
            'mode': self.config.screening_mode.value,
            'start_time': start_time,
            'posts_processed': 0,
            'screening_results': {},
            'errors': [],
            'performance_metrics': {}
        }

        try:
            # Validate API connections
            if not skip_validation:
                await self._validate_api_connections()

            # Determine posts to process
            posts_limit = max_posts or self.config.max_posts_per_run

            # Get count of posts ready for screening
            posts_count = await self._get_posts_ready_for_screening()
            actual_limit = min(posts_count, posts_limit)

            logger.info(
                f"📊 Found {posts_count} posts ready for screening, processing {actual_limit}")

            if actual_limit == 0:
                logger.warning("⚠️ No posts ready for screening")
                return pipeline_stats

            # Execute screening based on mode
            if self.config.screening_mode == ScreeningMode.TRADITIONAL:
                screening_results = await self._run_traditional_screening(actual_limit)
            elif self.config.screening_mode == ScreeningMode.AGENTIC:
                screening_results = await self._run_agentic_screening(actual_limit)
            elif self.config.screening_mode == ScreeningMode.HYBRID:
                screening_results = await self._run_hybrid_screening(actual_limit)
            else:
                raise ValueError(
                    f"Unknown screening mode: {self.config.screening_mode}")

            pipeline_stats['screening_results'] = screening_results
            pipeline_stats['posts_processed'] = screening_results.get(
                'total_processed', 0)

            # Generate performance report
            total_time = (datetime.utcnow() - start_time).total_seconds()
            pipeline_stats['total_time'] = total_time
            pipeline_stats['posts_per_second'] = pipeline_stats['posts_processed'] / \
                max(total_time, 1)

            logger.info(f"✅ Screening pipeline completed in {total_time:.2f}s")
            logger.info(
                f"📈 Processed {pipeline_stats['posts_processed']} posts at {pipeline_stats['posts_per_second']:.2f} posts/sec")

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            pipeline_stats['errors'].append(error_msg)

        return pipeline_stats

    async def _validate_api_connections(self):
        """Validate API connections for configured screeners."""
        logger.info("🔧 Validating API connections...")

        validation_results = {}

        if self.bulk_screener:
            try:
                validation_results['bulk_screener'] = await self.bulk_screener.validate_api_connection()
                logger.info(
                    f"Bulk screener API: {'✅' if validation_results['bulk_screener'] else '❌'}")
            except Exception as e:
                logger.error(f"Bulk screener validation failed: {str(e)}")
                validation_results['bulk_screener'] = False

        if self.borderline_screener:
            try:
                validation_results['borderline_screener'] = await self.borderline_screener.validate_api_connection()
                logger.info(
                    f"Borderline screener API: {'✅' if validation_results['borderline_screener'] else '❌'}")
            except Exception as e:
                logger.error(
                    f"Borderline screener validation failed: {str(e)}")
                validation_results['borderline_screener'] = False

        # Check if any required API failed
        failed_apis = [name for name,
                       success in validation_results.items() if not success]
        if failed_apis:
            raise RuntimeError(
                f"API validation failed for: {', '.join(failed_apis)}")

    async def _get_posts_ready_for_screening(self) -> int:
        """Get count of posts ready for LLM screening.

        Returns:
            Number of posts ready for screening
        """
        count = await self.db.count_documents(
            'filtered_posts',
            {
                'passed_keyword_filter': True,
                'llm_screened': {'$ne': True}
            }
        )
        return count

    async def _run_traditional_screening(self, posts_limit: int) -> Dict[str, Any]:
        """Run traditional screening pipeline.

        Args:
            posts_limit: Maximum posts to process

        Returns:
            Traditional screening results
        """
        logger.info("🔄 Running traditional screening pipeline")

        results = {
            'mode': 'traditional',
            'bulk_screening': {},
            'borderline_screening': {},
            'total_processed': 0
        }

        # Step 1: Bulk screening
        if self.bulk_screener:
            logger.info("📊 Running bulk screening...")
            bulk_results = await self.bulk_screener.screen_batch(
                batch_size=self.config.traditional_screening.bulk_batch_size
            )
            results['bulk_screening'] = bulk_results
            results['total_processed'] += bulk_results.get('processed', 0)

            logger.info(
                f"Bulk screening: {bulk_results.get('processed', 0)} posts processed")

            # Step 2: Borderline screening (re-evaluate uncertain cases)
            if self.borderline_screener:
                logger.info("🎯 Running borderline screening...")
                borderline_results = await self.borderline_screener.screen_borderline_cases(
                    confidence_min=self.config.traditional_screening.borderline_confidence_min,
                    confidence_max=self.config.traditional_screening.borderline_confidence_max,
                    batch_size=self.config.traditional_screening.borderline_batch_size
                )
                results['borderline_screening'] = borderline_results

                logger.info(
                    f"Borderline screening: {borderline_results.get('processed', 0)} posts processed")

        # Primary screening: Use advanced GPT-4 screener for comprehensive analysis
        elif self.borderline_screener:
            logger.info(
                "📊 Running comprehensive GPT-4 screening with comment analysis...")
            comprehensive_results = await self.borderline_screener.screen_all_unscreened_posts(
                batch_size=self.config.traditional_screening.borderline_batch_size
            )
            results['comprehensive_screening'] = comprehensive_results
            results['total_processed'] += comprehensive_results.get(
                'processed', 0)

            logger.info(
                f"Comprehensive screening: {comprehensive_results.get('processed', 0)} posts processed")

        return results

    async def _run_agentic_screening(self, posts_limit: int) -> Dict[str, Any]:
        """Run agentic screening pipeline.

        Args:
            posts_limit: Maximum posts to process

        Returns:
            Agentic screening results
        """
        logger.info("🤖 Running agentic screening pipeline")

        if not self.agentic_screener:
            raise RuntimeError("Agentic screener not initialized")

        results = await self.agentic_screener.screen_batch(
            batch_size=self.config.agentic_screening.batch_size,
            save_detailed_results=self.config.agentic_screening.save_detailed_results
        )

        # Format results for consistency
        return {
            'mode': 'agentic',
            'agentic_screening': results,
            'total_processed': results.get('processed', 0)
        }

    async def _run_hybrid_screening(self, posts_limit: int) -> Dict[str, Any]:
        """Run hybrid screening pipeline.

        Args:
            posts_limit: Maximum posts to process

        Returns:
            Hybrid screening results
        """
        logger.info("🔄🤖 Running hybrid screening pipeline")

        results = {
            'mode': 'hybrid',
            'traditional_results': {},
            'agentic_results': {},
            'total_processed': 0,
            'comparison_metrics': {}
        }

        # Run both traditional and agentic screening on same dataset
        # For comparison and quality validation

        # Split posts between approaches or run both on same posts
        half_limit = posts_limit // 2

        # Traditional screening on first half
        if self.bulk_screener or self.borderline_screener:
            traditional_results = await self._run_traditional_screening(half_limit)
            results['traditional_results'] = traditional_results
            results['total_processed'] += traditional_results.get(
                'total_processed', 0)

        # Agentic screening on second half
        if self.agentic_screener:
            agentic_results = await self._run_agentic_screening(posts_limit - half_limit)
            results['agentic_results'] = agentic_results
            results['total_processed'] += agentic_results.get(
                'total_processed', 0)

        # Generate comparison metrics
        results['comparison_metrics'] = await self._generate_comparison_metrics(
            traditional_results.get('bulk_screening', {}),
            agentic_results.get('agentic_screening', {})
        )

        return results

    async def _generate_comparison_metrics(
        self,
        traditional_stats: Dict[str, Any],
        agentic_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comparison metrics between screening approaches.

        Args:
            traditional_stats: Traditional screening statistics
            agentic_stats: Agentic screening statistics

        Returns:
            Comparison metrics
        """
        metrics = {
            'throughput_comparison': {},
            'decision_distribution': {},
            'confidence_analysis': {},
            'quality_assessment': {}
        }

        # Throughput comparison
        trad_processed = traditional_stats.get('processed', 0)
        trad_time = traditional_stats.get('processing_time', 1)
        agentic_processed = agentic_stats.get('processed', 0)
        agentic_time = agentic_stats.get('processing_time', 1)

        metrics['throughput_comparison'] = {
            'traditional_posts_per_sec': trad_processed / trad_time,
            'agentic_posts_per_sec': agentic_processed / agentic_time,
            'throughput_ratio': (trad_processed / trad_time) / max((agentic_processed / agentic_time), 0.001)
        }

        # Decision distribution comparison
        metrics['decision_distribution'] = {
            'traditional': {
                'positive': traditional_stats.get('positive_decisions', 0),
                'negative': traditional_stats.get('negative_decisions', 0),
                'borderline': traditional_stats.get('borderline_cases', 0)
            },
            'agentic': {
                'positive': agentic_stats.get('positive_decisions', 0),
                'negative': agentic_stats.get('negative_decisions', 0),
                'borderline': agentic_stats.get('borderline_cases', 0)
            }
        }

        return metrics

    async def get_screening_status(self) -> Dict[str, Any]:
        """Get current status of the screening pipeline.

        Returns:
            Status information
        """
        status = {
            'mode': self.config.screening_mode.value,
            'screeners_available': {},
            'posts_pending': 0,
            'recent_activity': {}
        }

        # Check screener availability
        status['screeners_available'] = {
            'bulk_screener': self.bulk_screener is not None,
            'borderline_screener': self.borderline_screener is not None,
            'agentic_screener': self.agentic_screener is not None
        }

        # Get pending posts count
        status['posts_pending'] = await self._get_posts_ready_for_screening()

        # Get recent screening activity
        try:
            recent_results = []
            async for result in self.db.find_many(
                'llm_screening_results',
                {},
                sort=[('created_at', -1)],
                limit=10
            ):
                recent_results.append({
                    'decision': result.get('decision'),
                    'confidence': result.get('confidence'),
                    'model': result.get('model_used'),
                    'timestamp': result.get('created_at')
                })

            status['recent_activity'] = recent_results

        except Exception as e:
            logger.error(f"Error getting recent activity: {str(e)}")
            status['recent_activity'] = []

        return status

    async def cleanup_and_shutdown(self):
        """Cleanup resources and shutdown screeners."""
        logger.info("🧹 Cleaning up screening orchestrator...")

        # No specific cleanup needed for current screeners
        # But this method provides extensibility for future resource management

        logger.info("✅ Screening orchestrator cleanup completed")
