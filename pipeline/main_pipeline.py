"""
Main Pipeline Orchestrator for LLM Contracts Research.

Coordinates the full pipeline following the methodology:
1. Data Acquisition (GitHub + Stack Overflow)
2. Keyword Pre-filtering  
3. LLM Screening (DeepSeek-R1 + GPT-4.1)
4. Human Labelling & Taxonomy
5. Reliability Validation (Fleiss κ ≥ 0.80)
6. Statistical Analysis & Dashboards

All artifacts stored in MongoDB Atlas with full provenance tracking.
"""

import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import os

from .common.database import MongoDBManager, ProvenanceTracker
from .common.config import PipelineConfig
from .data_acquisition import GitHubAcquisition, StackOverflowAcquisition
from .preprocessing import KeywordPreFilter
from .llm_screening import BulkScreener
from .reliability import FleissKappaValidator
from .analysis import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ResearchPipelineOrchestrator:
    """
    Main orchestrator for the LLM Contracts Research Pipeline.

    Coordinates all pipeline stages with full provenance tracking,
    quality gates, and statistical validation.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize database
        self.db = MongoDBManager(
            connection_string=config.mongodb_connection_string,
            database_name=config.database_name
        )
        self.provenance = ProvenanceTracker(self.db)

        # Initialize pipeline components
        self.github_acquisition = None
        self.stackoverflow_acquisition = None
        self.keyword_filter = None
        self.bulk_screener = None
        self.kappa_validator = None
        self.statistical_analyzer = None

        # Pipeline state
        self.pipeline_run_id = None
        self.current_stage = "initialized"
        self.stage_statistics = {}

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Connect to database
            await self.db.connect()
            logger.info("Connected to MongoDB")

            # Initialize components
            self.github_acquisition = GitHubAcquisition(
                github_token=self.config.github_token,
                db_manager=self.db
            )

            self.stackoverflow_acquisition = StackOverflowAcquisition(
                db_manager=self.db,
                api_key=self.config.stackoverflow_api_key
            )

            self.keyword_filter = KeywordPreFilter(self.db)

            self.bulk_screener = BulkScreener(
                api_key=self.config.deepseek_api_key,
                db_manager=self.db
            )

            self.kappa_validator = FleissKappaValidator(self.db)

            self.statistical_analyzer = StatisticalAnalyzer(self.db)

            # Create pipeline run record
            self.pipeline_run_id = await self._create_pipeline_run()

            logger.info(
                f"Pipeline initialized with run ID: {self.pipeline_run_id}")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    async def run_full_pipeline(
        self,
        skip_acquisition: bool = False,
        skip_filtering: bool = False,
        skip_screening: bool = False,
        max_posts_per_stage: int = 10000
    ) -> Dict[str, Any]:
        """Run the complete research pipeline.

        Args:
            skip_acquisition: Skip data acquisition if data already exists
            skip_filtering: Skip keyword filtering if already done
            skip_screening: Skip LLM screening if already done
            max_posts_per_stage: Maximum posts to process per stage

        Returns:
            Pipeline execution results and statistics
        """
        pipeline_start = datetime.utcnow()

        try:
            logger.info("🚀 Starting full research pipeline execution")

            # Stage 1: Data Acquisition
            if not skip_acquisition:
                await self._run_data_acquisition_stage(max_posts_per_stage)

            # Stage 2: Keyword Pre-filtering
            if not skip_filtering:
                await self._run_filtering_stage(max_posts_per_stage)

            # Stage 3: LLM Screening
            if not skip_screening:
                await self._run_screening_stage(max_posts_per_stage)

            # Stage 4: Generate labelling batches (for manual labelling)
            await self._prepare_labelling_batches()

            # Stage 5: Analysis and reporting (on existing labelled data)
            await self._run_analysis_stage()

            # Calculate final statistics
            pipeline_stats = await self._calculate_pipeline_statistics()

            # Update pipeline run record
            await self._update_pipeline_run(
                status="completed",
                end_time=datetime.utcnow(),
                statistics=pipeline_stats
            )

            execution_time = (datetime.utcnow() -
                              pipeline_start).total_seconds()

            logger.info(
                f"✅ Pipeline completed successfully in {execution_time:.1f}s")

            return {
                'pipeline_run_id': self.pipeline_run_id,
                'status': 'completed',
                'execution_time_seconds': execution_time,
                'stage_statistics': self.stage_statistics,
                'overall_statistics': pipeline_stats
            }

        except Exception as e:
            logger.error(f"💥 Pipeline execution failed: {str(e)}")

            # Update pipeline run with error
            await self._update_pipeline_run(
                status="failed",
                end_time=datetime.utcnow(),
                error=str(e)
            )

            raise

    async def _run_data_acquisition_stage(self, max_posts: int) -> None:
        """Run data acquisition from GitHub and Stack Overflow."""
        self.current_stage = "data_acquisition"
        stage_start = datetime.utcnow()

        logger.info("📥 Stage 1: Data Acquisition")

        # GitHub acquisition
        github_stats = {'posts_acquired': 0, 'repositories_processed': 0}

        logger.info("Acquiring data from GitHub repositories...")
        github_count = 0

        async for raw_post in self.github_acquisition.acquire_all_repositories(
            since_days=self.config.acquisition_since_days,
            max_issues_per_repo=max_posts // len(
                self.github_acquisition.target_repositories),
            include_discussions=True
        ):
            await self.github_acquisition.save_to_database(raw_post)
            github_count += 1

            if github_count >= max_posts // 2:  # Split budget between GitHub and SO
                break

        github_stats['posts_acquired'] = github_count
        github_stats['repositories_processed'] = len(
            self.github_acquisition.target_repositories)

        # Stack Overflow acquisition
        so_stats = {'posts_acquired': 0, 'tags_processed': 0}

        logger.info("Acquiring data from Stack Overflow...")
        so_count = 0

        async for raw_post in self.stackoverflow_acquisition.acquire_tagged_questions(
            since_days=self.config.acquisition_since_days,
            max_questions=max_posts // 2,
            include_answers=True
        ):
            await self.stackoverflow_acquisition.save_to_database(raw_post)
            so_count += 1

            if so_count >= max_posts // 2:
                break

        so_stats['posts_acquired'] = so_count
        so_stats['tags_processed'] = len(
            self.stackoverflow_acquisition.llm_tags)

        # Stage statistics
        stage_time = (datetime.utcnow() - stage_start).total_seconds()
        self.stage_statistics['data_acquisition'] = {
            'duration_seconds': stage_time,
            'github': github_stats,
            'stackoverflow': so_stats,
            'total_posts_acquired': github_count + so_count
        }

        logger.info(
            f"✅ Data acquisition completed: {github_count + so_count} posts in {stage_time:.1f}s")

    async def _run_filtering_stage(self, max_posts: int) -> None:
        """Run keyword pre-filtering stage."""
        self.current_stage = "filtering"
        stage_start = datetime.utcnow()

        logger.info("🔍 Stage 2: Keyword Pre-filtering")

        # Run filtering in batches
        batch_size = min(1000, max_posts)
        total_stats = {
            'processed': 0,
            'passed': 0,
            'failed': 0,
            'high_confidence': 0,
            'batches_processed': 0
        }

        while total_stats['processed'] < max_posts:
            batch_stats = await self.keyword_filter.filter_batch(
                batch_size=batch_size,
                confidence_threshold=0.3
            )

            # Accumulate statistics
            for key in ['processed', 'passed', 'failed', 'high_confidence']:
                total_stats[key] += batch_stats.get(key, 0)

            total_stats['batches_processed'] += 1

            logger.info(
                f"Batch {total_stats['batches_processed']}: {batch_stats['processed']} processed, {batch_stats['passed']} passed")

            # Break if no more posts to process
            if batch_stats['processed'] == 0:
                break

        # Calculate filtering efficiency
        if total_stats['processed'] > 0:
            pass_rate = (total_stats['passed'] /
                         total_stats['processed']) * 100
            noise_reduction = (
                (total_stats['processed'] - total_stats['passed']) / total_stats['processed']) * 100
        else:
            pass_rate = 0
            noise_reduction = 0

        stage_time = (datetime.utcnow() - stage_start).total_seconds()
        self.stage_statistics['filtering'] = {
            'duration_seconds': stage_time,
            'posts_processed': total_stats['processed'],
            'posts_passed': total_stats['passed'],
            'posts_filtered_out': total_stats['failed'],
            'pass_rate_percent': round(pass_rate, 1),
            'noise_reduction_percent': round(noise_reduction, 1),
            'high_confidence_posts': total_stats['high_confidence'],
            'batches_processed': total_stats['batches_processed']
        }

        logger.info(
            f"✅ Filtering completed: {total_stats['passed']}/{total_stats['processed']} passed ({pass_rate:.1f}%), {noise_reduction:.1f}% noise reduction")

    async def _run_screening_stage(self, max_posts: int) -> None:
        """Run LLM screening stage."""
        self.current_stage = "llm_screening"
        stage_start = datetime.utcnow()

        logger.info("🧠 Stage 3: LLM Screening")

        # Validate DeepSeek API connection
        if not await self.bulk_screener.validate_api_connection():
            raise RuntimeError("DeepSeek API validation failed")

        # Run bulk screening
        batch_size = min(100, max_posts)
        screening_stats = await self.bulk_screener.screen_batch(
            batch_size=batch_size,
            confidence_threshold=0.4
        )

        stage_time = (datetime.utcnow() - stage_start).total_seconds()
        self.stage_statistics['llm_screening'] = {
            'duration_seconds': stage_time,
            'posts_screened': screening_stats['processed'],
            'positive_decisions': screening_stats['positive_decisions'],
            'negative_decisions': screening_stats['negative_decisions'],
            'borderline_cases': screening_stats['borderline_cases'],
            'high_confidence_decisions': screening_stats['high_confidence'],
            'api_calls_made': screening_stats['api_calls'],
            'errors': screening_stats['errors']
        }

        logger.info(
            f"✅ LLM screening completed: {screening_stats['processed']} posts screened, {screening_stats['positive_decisions']} positive")

    async def _prepare_labelling_batches(self) -> None:
        """Prepare batches for human labelling."""
        self.current_stage = "labelling_prep"
        stage_start = datetime.utcnow()

        logger.info("👥 Stage 4: Preparing Human Labelling Batches")

        # Create labelling session
        session_data = {
            'session_date': datetime.utcnow(),
            'raters': ['R1', 'R2', 'R3'],
            'posts_assigned': 0,
            'posts_completed': 0,
            'moderator': 'auto_generated',
            'status': 'active'
        }

        session_id = await self.db.save_labelling_session(session_data)

        # Count posts ready for labelling (passed LLM screening)
        posts_for_labelling = 0
        async for post in self.db.get_posts_for_labelling(session_id, batch_size=1000):
            posts_for_labelling += 1

        # Update session with assignment count
        await self.db.update_one(
            'labelling_sessions',
            {'_id': session_id},
            {'$set': {'posts_assigned': posts_for_labelling}}
        )

        stage_time = (datetime.utcnow() - stage_start).total_seconds()
        self.stage_statistics['labelling_prep'] = {
            'duration_seconds': stage_time,
            'session_id': str(session_id),
            'posts_ready_for_labelling': posts_for_labelling,
            'raters_assigned': 3
        }

        logger.info(
            f"✅ Labelling preparation completed: {posts_for_labelling} posts ready, session {session_id}")

    async def _run_analysis_stage(self) -> None:
        """Run analysis on existing labelled data."""
        self.current_stage = "analysis"
        stage_start = datetime.utcnow()

        logger.info("📊 Stage 5: Statistical Analysis")

        # Get pipeline statistics
        pipeline_stats = await self.db.get_pipeline_statistics()

        # Check if we have any labelled data to analyze
        if pipeline_stats['labelled_posts']['total'] > 0:
            logger.info(
                f"Found {pipeline_stats['labelled_posts']['total']} labelled posts for analysis")

            # Run reliability analysis on existing sessions
            reliability_results = []
            async for session in self.db.find_many('labelling_sessions', {'status': 'completed'}):
                try:
                    session_id = str(session['_id'])
                    validation_result = await self.kappa_validator.validate_session_quality(session_id)
                    reliability_results.append(validation_result)
                except Exception as e:
                    logger.warning(
                        f"Failed to validate session {session.get('_id')}: {str(e)}")

            analysis_stats = {
                'sessions_analyzed': len(reliability_results),
                'sessions_passing_kappa': sum(1 for r in reliability_results if r.get('meets_threshold', False)),
                'average_kappa': np.mean([r.get('fleiss_kappa', 0) for r in reliability_results]) if reliability_results else 0,
                # Include first 5 results
                'reliability_results': reliability_results[:5]
            }
        else:
            logger.info("No labelled data found for analysis")
            analysis_stats = {
                'sessions_analyzed': 0,
                'sessions_passing_kappa': 0,
                'average_kappa': 0,
                'note': 'No labelled data available for analysis'
            }

        stage_time = (datetime.utcnow() - stage_start).total_seconds()
        self.stage_statistics['analysis'] = {
            'duration_seconds': stage_time,
            **analysis_stats
        }

        logger.info(f"✅ Analysis completed in {stage_time:.1f}s")

    async def _calculate_pipeline_statistics(self) -> Dict[str, Any]:
        """Calculate overall pipeline statistics."""
        pipeline_stats = await self.db.get_pipeline_statistics()

        # Calculate conversion rates
        raw_total = pipeline_stats['raw_posts']['total']
        filtered_passed = pipeline_stats['filtered_posts']['passed_filter']
        labelled_total = pipeline_stats['labelled_posts']['total']

        conversion_rates = {
            'raw_to_filtered': (filtered_passed / raw_total * 100) if raw_total > 0 else 0,
            'filtered_to_labelled': (labelled_total / filtered_passed * 100) if filtered_passed > 0 else 0,
            'raw_to_labelled': (labelled_total / raw_total * 100) if raw_total > 0 else 0
        }

        return {
            'pipeline_efficiency': {
                'total_posts_acquired': raw_total,
                'posts_after_filtering': filtered_passed,
                'posts_labelled': labelled_total,
                'conversion_rates_percent': conversion_rates
            },
            'data_sources': {
                'github_posts': pipeline_stats['raw_posts']['github'],
                'stackoverflow_posts': pipeline_stats['raw_posts']['stackoverflow']
            },
            'quality_metrics': {
                'posts_with_majority_agreement': pipeline_stats['labelled_posts']['majority_agreement'],
                'posts_needing_arbitration': pipeline_stats['labelled_posts']['needs_arbitration'],
                'active_sessions': pipeline_stats['sessions']['active'],
                'completed_sessions': pipeline_stats['sessions']['completed']
            },
            'stage_performance': self.stage_statistics
        }

    async def _create_pipeline_run(self) -> str:
        """Create a new pipeline run record."""
        run_data = {
            'start_time': datetime.utcnow(),
            'status': 'running',
            'config': {
                'acquisition_since_days': self.config.acquisition_since_days,
                'database_name': self.config.database_name,
                'pipeline_version': '1.0.0'
            },
            'stages_completed': [],
            'current_stage': self.current_stage,
            'statistics': {}
        }

        result = await self.db.insert_one('pipeline_runs', run_data)
        return str(result.inserted_id)

    async def _update_pipeline_run(
        self,
        status: str = None,
        end_time: datetime = None,
        statistics: Dict[str, Any] = None,
        error: str = None
    ) -> None:
        """Update pipeline run record."""
        update_data = {
            'last_updated': datetime.utcnow(),
            'current_stage': self.current_stage
        }

        if status:
            update_data['status'] = status

        if end_time:
            update_data['end_time'] = end_time

        if statistics:
            update_data['statistics'] = statistics

        if error:
            update_data['error'] = error

        await self.db.update_one(
            'pipeline_runs',
            {'_id': self.pipeline_run_id},
            {'$set': update_data}
        )

    async def validate_kappa_for_session(self, session_id: str) -> Dict[str, Any]:
        """Validate Fleiss Kappa for a specific session."""
        logger.info(f"🧮 Validating Fleiss Kappa for session: {session_id}")

        try:
            validation_result = await self.kappa_validator.validate_session_quality(session_id)

            if validation_result['passes_validation']:
                logger.info(
                    f"✅ Session {session_id} passes validation (κ={validation_result['fleiss_kappa']:.3f})")
            else:
                logger.warning(
                    f"❌ Session {session_id} fails validation (κ={validation_result.get('fleiss_kappa', 'N/A')})")
                logger.info(
                    f"Recommendations: {validation_result.get('recommendations', [])}")

            return validation_result

        except Exception as e:
            logger.error(f"Error validating session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'passes_validation': False,
                'error': str(e)
            }

    async def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        logger.info("📋 Generating pipeline report...")

        # Get overall statistics
        pipeline_stats = await self._calculate_pipeline_statistics()

        # Get recent pipeline runs
        recent_runs = []
        async for run in self.db.find_many(
            'pipeline_runs',
            {},
            limit=5,
            sort=[('start_time', -1)]
        ):
            recent_runs.append({
                'run_id': str(run['_id']),
                'start_time': run['start_time'],
                'status': run.get('status', 'unknown'),
                'current_stage': run.get('current_stage', 'unknown')
            })

        # Get reliability metrics summary
        reliability_summary = []
        async for metrics in self.db.find_many(
            'reliability_metrics',
            {},
            limit=10,
            sort=[('calculation_date', -1)]
        ):
            reliability_summary.append({
                'session_id': metrics['session_id'],
                'fleiss_kappa': round(metrics['fleiss_kappa'], 3),
                'passes_threshold': metrics['passes_threshold'],
                'calculation_date': metrics['calculation_date'],
                'n_items': metrics['n_items']
            })

        report = {
            'generated_at': datetime.utcnow(),
            'pipeline_statistics': pipeline_stats,
            'recent_runs': recent_runs,
            'reliability_summary': reliability_summary,
            'quality_status': {
                'data_acquisition_healthy': pipeline_stats['pipeline_efficiency']['total_posts_acquired'] > 0,
                'filtering_effective': pipeline_stats['pipeline_efficiency']['conversion_rates_percent']['raw_to_filtered'] > 5,
                'reliability_acceptable': any(r['passes_threshold'] for r in reliability_summary) if reliability_summary else False
            }
        }

        return report

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.db:
            await self.db.disconnect()
        logger.info("Pipeline cleanup completed")


# Import numpy for analysis calculations
