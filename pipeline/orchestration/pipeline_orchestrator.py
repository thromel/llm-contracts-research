"""
Unified Pipeline Orchestrator.

Combines and modernizes both ResearchPipelineOrchestrator and ScreeningOrchestrator
into a single, cohesive orchestrator using the new foundation architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from ..foundation.config import ConfigManager
from ..foundation.logging import PipelineLogger
from ..foundation.retry import with_retry, CircuitBreaker
from ..infrastructure.database import DatabaseManager
from ..infrastructure.monitoring import MetricsCollector
from ..domain.models import PipelineStage


class PipelineMode(Enum):
    """Pipeline execution modes."""
    RESEARCH = "research"  # Full research pipeline (acquisition -> analysis)
    SCREENING_ONLY = "screening_only"  # LLM screening only
    TRADITIONAL = "traditional"  # Traditional LLM screening
    AGENTIC = "agentic"  # Agentic LLM screening
    HYBRID = "hybrid"  # Hybrid screening approach


class UnifiedPipelineOrchestrator:
    """
    Unified pipeline orchestrator that coordinates all pipeline operations.
    
    Features:
    - Mode-based execution (research, screening, traditional, agentic, hybrid)
    - Foundation layer integration (config, logging, retry)
    - Infrastructure layer integration (database, monitoring)
    - Type-safe operation with enhanced error handling
    - Comprehensive metrics collection and reporting
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        logger: Optional[PipelineLogger] = None,
        db_manager: Optional[DatabaseManager] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        """Initialize the unified pipeline orchestrator.
        
        Args:
            config: Configuration manager (creates default if None)
            logger: Pipeline logger (creates default if None)
            db_manager: Database manager (creates default if None)
            metrics: Metrics collector (creates default if None)
        """
        # Initialize foundation components
        self.config = config or ConfigManager()
        self.logger = logger or PipelineLogger(__name__)
        
        # Initialize infrastructure components
        self.db = db_manager or DatabaseManager(
            connection_string=self.config.get("mongodb.uri"),
            database_name=self.config.get("mongodb.database")
        )
        self.metrics = metrics or MetricsCollector()
        
        # Pipeline state
        self.pipeline_run_id: Optional[str] = None
        self.current_stage: Optional[PipelineStage] = None
        self.stage_statistics: Dict[str, Any] = {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60
        )
        
        # Component registry
        self._components: Dict[str, Any] = {}
        
        self.logger.info("UnifiedPipelineOrchestrator initialized", extra={
            "config_source": self.config.source,
            "database": self.config.get("mongodb.database")
        })

    async def initialize(self) -> None:
        """Initialize all pipeline components and validate configuration."""
        try:
            with self.metrics.timer("initialization"):
                self.logger.info("Initializing pipeline orchestrator")
                
                # Initialize database connection
                await self._initialize_database()
                
                # Validate configuration
                await self._validate_configuration()
                
                # Initialize pipeline components based on mode
                await self._initialize_components()
                
                # Create pipeline run record
                self.pipeline_run_id = await self._create_pipeline_run()
                
                self.logger.info("Pipeline orchestrator initialized successfully", extra={
                    "pipeline_run_id": self.pipeline_run_id
                })
                
        except Exception as e:
            self.logger.error("Failed to initialize pipeline orchestrator", extra={
                "error": str(e)
            })
            raise

    @with_retry(max_attempts=3, delay=1.0)
    async def _initialize_database(self) -> None:
        """Initialize database connection with retry logic."""
        await self.db.connect()
        
        # Validate database health
        health = await self.db.health_check()
        if not health.get("healthy", False):
            raise RuntimeError(f"Database health check failed: {health}")
        
        self.logger.info("Database connection initialized", extra={
            "database": self.config.get("mongodb.database"),
            "health_status": health
        })

    async def _validate_configuration(self) -> None:
        """Validate pipeline configuration and API connections."""
        self.logger.info("Validating pipeline configuration")
        
        # Validate required configuration
        required_config = [
            "mongodb.uri",
            "mongodb.database"
        ]
        
        for key in required_config:
            if not self.config.get(key):
                raise ValueError(f"Required configuration missing: {key}")
        
        # Validate API keys based on enabled features
        if self.config.get("screening.traditional.enabled", False):
            if not self.config.get("openai.api_key"):
                raise ValueError("OpenAI API key required for traditional screening")
        
        if self.config.get("screening.agentic.enabled", False):
            if not self.config.get("openai.api_key"):
                raise ValueError("OpenAI API key required for agentic screening")
        
        self.logger.info("Configuration validation completed")

    async def _initialize_components(self) -> None:
        """Initialize pipeline components based on configuration."""
        self.logger.info("Initializing pipeline components")
        
        # Data acquisition components
        if self.config.get("acquisition.github.enabled", False):
            await self._initialize_github_acquisition()
        
        if self.config.get("acquisition.stackoverflow.enabled", False):
            await self._initialize_stackoverflow_acquisition()
        
        # Preprocessing components
        await self._initialize_keyword_filter()
        
        # Screening components
        await self._initialize_screening_components()
        
        # Analysis components
        await self._initialize_analysis_components()
        
        self.logger.info("Pipeline components initialized", extra={
            "components": list(self._components.keys())
        })

    async def _initialize_github_acquisition(self) -> None:
        """Initialize GitHub data acquisition component."""
        from ..data_acquisition.github import GitHubAcquisition
        
        self._components["github_acquisition"] = GitHubAcquisition(
            github_token=self.config.get("github.token"),
            db_manager=self.db
        )
        self.logger.info("GitHub acquisition component initialized")

    async def _initialize_stackoverflow_acquisition(self) -> None:
        """Initialize Stack Overflow data acquisition component."""
        from ..data_acquisition.stackoverflow import StackOverflowAcquisition
        
        self._components["stackoverflow_acquisition"] = StackOverflowAcquisition(
            db_manager=self.db,
            api_key=self.config.get("stackoverflow.api_key")
        )
        self.logger.info("Stack Overflow acquisition component initialized")

    async def _initialize_keyword_filter(self) -> None:
        """Initialize keyword filtering component."""
        from ..preprocessing.keyword_filter import KeywordPreFilter
        
        self._components["keyword_filter"] = KeywordPreFilter(self.db)
        self.logger.info("Keyword filter component initialized")

    async def _initialize_screening_components(self) -> None:
        """Initialize LLM screening components."""
        from ..llm_screening.screening_orchestrator import ScreeningOrchestrator
        
        # Import old config system for compatibility
        from ..common.config import PipelineConfig
        
        # Create legacy config for screening orchestrator
        legacy_config = PipelineConfig(
            mongodb_connection_string=self.config.get("mongodb.uri"),
            database_name=self.config.get("mongodb.database"),
            openai_api_key=self.config.get("openai.api_key"),
            deepseek_api_key=self.config.get("deepseek.api_key"),
            github_token=self.config.get("github.token"),
            stackoverflow_api_key=self.config.get("stackoverflow.api_key")
        )
        
        self._components["screening_orchestrator"] = ScreeningOrchestrator(
            config=legacy_config,
            db_manager=self.db
        )
        self.logger.info("Screening orchestrator component initialized")

    async def _initialize_analysis_components(self) -> None:
        """Initialize analysis components."""
        from ..reliability.fleiss_kappa import FleissKappaValidator
        
        self._components["kappa_validator"] = FleissKappaValidator(self.db)
        self.logger.info("Analysis components initialized")

    async def execute_pipeline(
        self,
        mode: PipelineMode = PipelineMode.RESEARCH,
        stages: Optional[List[PipelineStage]] = None,
        max_posts_per_stage: int = 1000,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """Execute the pipeline in the specified mode.
        
        Args:
            mode: Pipeline execution mode
            stages: Specific stages to execute (all if None)
            max_posts_per_stage: Maximum posts to process per stage
            skip_validation: Skip API validation checks
            
        Returns:
            Pipeline execution results and statistics
        """
        execution_start = datetime.utcnow()
        
        try:
            self.logger.info("Starting pipeline execution", extra={
                "mode": mode.value,
                "stages": [s.value for s in stages] if stages else "all",
                "max_posts_per_stage": max_posts_per_stage
            })
            
            with self.metrics.timer("pipeline_execution"):
                # Update pipeline run status
                await self._update_pipeline_run(status="running")
                
                # Execute based on mode
                if mode == PipelineMode.RESEARCH:
                    results = await self._execute_research_pipeline(
                        stages, max_posts_per_stage, skip_validation
                    )
                elif mode == PipelineMode.SCREENING_ONLY:
                    results = await self._execute_screening_only(
                        max_posts_per_stage, skip_validation
                    )
                elif mode in [PipelineMode.TRADITIONAL, PipelineMode.AGENTIC, PipelineMode.HYBRID]:
                    results = await self._execute_screening_mode(
                        mode, max_posts_per_stage, skip_validation
                    )
                else:
                    raise ValueError(f"Unknown pipeline mode: {mode}")
                
                # Calculate execution metrics
                execution_time = (datetime.utcnow() - execution_start).total_seconds()
                results.update({
                    "pipeline_run_id": self.pipeline_run_id,
                    "execution_time_seconds": execution_time,
                    "mode": mode.value,
                    "stage_statistics": self.stage_statistics
                })
                
                # Update pipeline run with success
                await self._update_pipeline_run(
                    status="completed",
                    end_time=datetime.utcnow(),
                    statistics=results
                )
                
                self.logger.info("Pipeline execution completed successfully", extra={
                    "execution_time": execution_time,
                    "posts_processed": results.get("total_posts_processed", 0)
                })
                
                return results
                
        except Exception as e:
            execution_time = (datetime.utcnow() - execution_start).total_seconds()
            
            # Update pipeline run with error
            await self._update_pipeline_run(
                status="failed",
                end_time=datetime.utcnow(),
                error=str(e)
            )
            
            self.logger.error("Pipeline execution failed", extra={
                "error": str(e),
                "execution_time": execution_time
            })
            
            raise

    async def _execute_research_pipeline(
        self,
        stages: Optional[List[PipelineStage]],
        max_posts_per_stage: int,
        skip_validation: bool
    ) -> Dict[str, Any]:
        """Execute the full research pipeline."""
        results = {"total_posts_processed": 0, "stage_results": {}}
        
        # Default stages for research pipeline
        if stages is None:
            stages = [
                PipelineStage.DATA_ACQUISITION,
                PipelineStage.DATA_PREPROCESSING,
                PipelineStage.LLM_SCREENING,
                PipelineStage.LABELLING_PREP,
                PipelineStage.ANALYSIS
            ]
        
        for stage in stages:
            self.current_stage = stage
            stage_result = await self._execute_stage(stage, max_posts_per_stage, skip_validation)
            results["stage_results"][stage.value] = stage_result
            results["total_posts_processed"] += stage_result.get("posts_processed", 0)
        
        return results

    async def _execute_screening_only(
        self,
        max_posts: int,
        skip_validation: bool
    ) -> Dict[str, Any]:
        """Execute LLM screening only."""
        self.current_stage = PipelineStage.LLM_SCREENING
        
        screening_orchestrator = self._components.get("screening_orchestrator")
        if not screening_orchestrator:
            raise RuntimeError("Screening orchestrator not initialized")
        
        return await screening_orchestrator.run_screening_pipeline(
            max_posts=max_posts,
            skip_validation=skip_validation
        )

    async def _execute_screening_mode(
        self,
        mode: PipelineMode,
        max_posts: int,
        skip_validation: bool
    ) -> Dict[str, Any]:
        """Execute specific screening mode."""
        # Map pipeline mode to screening mode
        screening_mode_map = {
            PipelineMode.TRADITIONAL: "traditional",
            PipelineMode.AGENTIC: "agentic", 
            PipelineMode.HYBRID: "hybrid"
        }
        
        # Update screening configuration
        original_mode = self.config.get("screening.mode")
        self.config.set("screening.mode", screening_mode_map[mode])
        
        try:
            result = await self._execute_screening_only(max_posts, skip_validation)
            result["mode"] = mode.value
            return result
        finally:
            # Restore original mode
            if original_mode:
                self.config.set("screening.mode", original_mode)

    async def _execute_stage(
        self,
        stage: PipelineStage,
        max_posts: int,
        skip_validation: bool
    ) -> Dict[str, Any]:
        """Execute a specific pipeline stage."""
        stage_start = datetime.utcnow()
        
        self.logger.info(f"Executing stage: {stage.value}")
        
        try:
            with self.metrics.timer(f"stage_{stage.value}"):
                if stage == PipelineStage.DATA_ACQUISITION:
                    result = await self._execute_data_acquisition(max_posts)
                elif stage == PipelineStage.DATA_PREPROCESSING:
                    result = await self._execute_preprocessing(max_posts)
                elif stage == PipelineStage.LLM_SCREENING:
                    result = await self._execute_screening_only(max_posts, skip_validation)
                elif stage == PipelineStage.LABELLING_PREP:
                    result = await self._execute_labelling_prep()
                elif stage == PipelineStage.ANALYSIS:
                    result = await self._execute_analysis()
                else:
                    raise ValueError(f"Unknown stage: {stage}")
                
                # Record stage statistics
                stage_time = (datetime.utcnow() - stage_start).total_seconds()
                result["duration_seconds"] = stage_time
                self.stage_statistics[stage.value] = result
                
                self.logger.info(f"Stage {stage.value} completed", extra={
                    "duration": stage_time,
                    "posts_processed": result.get("posts_processed", 0)
                })
                
                return result
                
        except Exception as e:
            stage_time = (datetime.utcnow() - stage_start).total_seconds()
            self.logger.error(f"Stage {stage.value} failed", extra={
                "error": str(e),
                "duration": stage_time
            })
            raise

    async def _execute_data_acquisition(self, max_posts: int) -> Dict[str, Any]:
        """Execute data acquisition stage."""
        results = {"posts_processed": 0, "sources": {}}
        
        # GitHub acquisition
        if "github_acquisition" in self._components:
            github_acquisition = self._components["github_acquisition"]
            github_count = 0
            
            async for raw_post in github_acquisition.acquire_all_repositories(
                since_days=self.config.get("acquisition.since_days", 30),
                max_issues_per_repo=max_posts // 4,
                include_discussions=True
            ):
                await github_acquisition.save_to_database(raw_post)
                github_count += 1
                
                if github_count >= max_posts // 2:
                    break
            
            results["sources"]["github"] = {"posts_acquired": github_count}
            results["posts_processed"] += github_count
        
        # Stack Overflow acquisition
        if "stackoverflow_acquisition" in self._components:
            so_acquisition = self._components["stackoverflow_acquisition"]
            so_count = 0
            
            async for raw_post in so_acquisition.acquire_tagged_questions(
                since_days=self.config.get("acquisition.since_days", 30),
                max_questions=max_posts // 2,
                include_answers=True
            ):
                await so_acquisition.save_to_database(raw_post)
                so_count += 1
                
                if so_count >= max_posts // 2:
                    break
            
            results["sources"]["stackoverflow"] = {"posts_acquired": so_count}
            results["posts_processed"] += so_count
        
        return results

    async def _execute_preprocessing(self, max_posts: int) -> Dict[str, Any]:
        """Execute preprocessing stage."""
        keyword_filter = self._components.get("keyword_filter")
        if not keyword_filter:
            raise RuntimeError("Keyword filter not initialized")
        
        return await keyword_filter.filter_batch(
            batch_size=min(1000, max_posts),
            confidence_threshold=0.3
        )

    async def _execute_labelling_prep(self) -> Dict[str, Any]:
        """Execute labelling preparation stage."""
        session_data = {
            "session_date": datetime.utcnow(),
            "raters": ["R1", "R2", "R3"],
            "posts_assigned": 0,
            "posts_completed": 0,
            "moderator": "auto_generated",
            "status": "active"
        }
        
        session_id = await self.db.save_labelling_session(session_data)
        
        # Count posts ready for labelling
        posts_count = 0
        async for post in self.db.get_posts_for_labelling(session_id, batch_size=1000):
            posts_count += 1
        
        # Update session with assignment count
        await self.db.update_one(
            "labelling_sessions",
            {"_id": session_id},
            {"$set": {"posts_assigned": posts_count}}
        )
        
        return {
            "session_id": str(session_id),
            "posts_ready_for_labelling": posts_count,
            "posts_processed": posts_count
        }

    async def _execute_analysis(self) -> Dict[str, Any]:
        """Execute analysis stage."""
        kappa_validator = self._components.get("kappa_validator")
        if not kappa_validator:
            raise RuntimeError("Kappa validator not initialized")
        
        # Get pipeline statistics
        pipeline_stats = await self.db.get_pipeline_statistics()
        
        # Run reliability analysis on completed sessions
        reliability_results = []
        async for session in self.db.find_many("labelling_sessions", {"status": "completed"}):
            try:
                session_id = str(session["_id"])
                validation_result = await kappa_validator.validate_session_quality(session_id)
                reliability_results.append(validation_result)
            except Exception as e:
                self.logger.warning(f"Failed to validate session {session.get('_id')}", extra={
                    "error": str(e)
                })
        
        return {
            "sessions_analyzed": len(reliability_results),
            "sessions_passing_kappa": sum(1 for r in reliability_results if r.get("meets_threshold", False)),
            "average_kappa": sum(r.get("fleiss_kappa", 0) for r in reliability_results) / max(len(reliability_results), 1),
            "reliability_results": reliability_results[:5],
            "posts_processed": pipeline_stats.get("labelled_posts", {}).get("total", 0)
        }

    async def _create_pipeline_run(self) -> str:
        """Create a new pipeline run record."""
        run_data = {
            "start_time": datetime.utcnow(),
            "status": "initializing",
            "config": {
                "source": self.config.source,
                "pipeline_version": "2.0.0"
            },
            "stages_completed": [],
            "current_stage": self.current_stage.value if self.current_stage else None,
            "statistics": {}
        }
        
        result = await self.db.insert_one("pipeline_runs", run_data)
        return str(result.inserted_id)

    async def _update_pipeline_run(
        self,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        statistics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Update pipeline run record."""
        update_data = {
            "last_updated": datetime.utcnow(),
            "current_stage": self.current_stage.value if self.current_stage else None
        }
        
        if status:
            update_data["status"] = status
        if end_time:
            update_data["end_time"] = end_time
        if statistics:
            update_data["statistics"] = statistics
        if error:
            update_data["error"] = error
        
        await self.db.update_one(
            "pipeline_runs",
            {"_id": self.pipeline_run_id},
            {"$set": update_data}
        )

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        # Get current run status
        current_run = None
        if self.pipeline_run_id:
            current_run = await self.db.find_one("pipeline_runs", {"_id": self.pipeline_run_id})
        
        # Get overall statistics
        pipeline_stats = await self.db.get_pipeline_statistics()
        
        # Get component status
        component_status = {}
        for name, component in self._components.items():
            if hasattr(component, "get_status"):
                component_status[name] = await component.get_status()
            else:
                component_status[name] = "initialized"
        
        return {
            "current_run": {
                "id": self.pipeline_run_id,
                "status": current_run.get("status") if current_run else "not_started",
                "current_stage": self.current_stage.value if self.current_stage else None,
                "start_time": current_run.get("start_time") if current_run else None
            },
            "pipeline_statistics": pipeline_stats,
            "component_status": component_status,
            "stage_statistics": self.stage_statistics
        }

    async def cleanup(self) -> None:
        """Cleanup resources and shutdown components."""
        self.logger.info("Cleaning up pipeline orchestrator")
        
        # Cleanup components
        for name, component in self._components.items():
            if hasattr(component, "cleanup"):
                try:
                    await component.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up component {name}", extra={
                        "error": str(e)
                    })
        
        # Disconnect database
        if self.db:
            await self.db.disconnect()
        
        self.logger.info("Pipeline orchestrator cleanup completed")