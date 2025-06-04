"""
Main Pipeline Orchestrator for LLM Contracts Research.

This module provides backward compatibility while leveraging the new
UnifiedPipelineOrchestrator architecture. It maintains the same API
but delegates to the new foundation and infrastructure layers.
"""

import logging
from typing import Dict, Any

from .foundation.config import ConfigManager
from .orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator, PipelineMode
from .domain.models import PipelineStage
from .common.config import PipelineConfig

logger = logging.getLogger(__name__)


class ResearchPipelineOrchestrator:
    """
    Main orchestrator for the LLM Contracts Research Pipeline.
    
    This class now serves as a compatibility wrapper around the new
    UnifiedPipelineOrchestrator while maintaining the same API for
    existing code that depends on this interface.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline orchestrator.

        Args:
            config: Legacy PipelineConfig object
        """
        # Convert legacy config to new ConfigManager
        self.legacy_config = config
        self.config = self._convert_legacy_config(config)
        
        # Initialize new unified orchestrator
        self.unified_orchestrator = UnifiedPipelineOrchestrator(
            config=self.config
        )
        
        # Legacy compatibility properties
        self.pipeline_run_id = None
        self.current_stage = "initialized"
        self.stage_statistics = {}

    def _convert_legacy_config(self, legacy_config: PipelineConfig) -> ConfigManager:
        """Convert legacy PipelineConfig to new ConfigManager format."""
        config_manager = ConfigManager()
        
        # Database configuration
        if hasattr(legacy_config, 'mongodb_connection_string'):
            config_manager.set("mongodb.uri", legacy_config.mongodb_connection_string)
        if hasattr(legacy_config, 'database_name'):
            config_manager.set("mongodb.database", legacy_config.database_name)
            
        # API keys
        if hasattr(legacy_config, 'github_token'):
            config_manager.set("github.token", legacy_config.github_token)
        if hasattr(legacy_config, 'openai_api_key'):
            config_manager.set("openai.api_key", legacy_config.openai_api_key)
        if hasattr(legacy_config, 'deepseek_api_key'):
            config_manager.set("deepseek.api_key", legacy_config.deepseek_api_key)
        if hasattr(legacy_config, 'stackoverflow_api_key'):
            config_manager.set("stackoverflow.api_key", legacy_config.stackoverflow_api_key)
            
        # Pipeline settings
        if hasattr(legacy_config, 'acquisition_since_days'):
            config_manager.set("acquisition.since_days", legacy_config.acquisition_since_days)
            
        # Enable components
        config_manager.set("acquisition.github.enabled", True)
        config_manager.set("acquisition.stackoverflow.enabled", True)
        config_manager.set("screening.traditional.enabled", True)
        
        return config_manager
        
    @property
    def db(self):
        """Legacy database access."""
        return self.unified_orchestrator.db

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize the unified orchestrator
            await self.unified_orchestrator.initialize()
            
            # Set legacy compatibility properties
            self.pipeline_run_id = self.unified_orchestrator.pipeline_run_id
            self.current_stage = str(self.unified_orchestrator.current_stage.value) if self.unified_orchestrator.current_stage else "initialized"
            
            logger.info(f"Pipeline initialized with run ID: {self.pipeline_run_id}")

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
        # Determine which stages to run based on skip flags
        stages = []
        if not skip_acquisition:
            stages.append(PipelineStage.DATA_ACQUISITION)
        if not skip_filtering:
            stages.append(PipelineStage.DATA_PREPROCESSING)
        if not skip_screening:
            stages.append(PipelineStage.LLM_SCREENING)
        
        # Always include these stages for completeness
        stages.extend([
            PipelineStage.LABELLING_PREP,
            PipelineStage.ANALYSIS
        ])
        
        # Execute the pipeline using the unified orchestrator
        results = await self.unified_orchestrator.execute_pipeline(
            mode=PipelineMode.RESEARCH,
            stages=stages,
            max_posts_per_stage=max_posts_per_stage,
            skip_validation=False
        )
        
        # Update legacy properties for compatibility
        self.pipeline_run_id = results.get('pipeline_run_id', self.pipeline_run_id)
        self.stage_statistics = results.get('stage_statistics', {})
        
        # Convert results to legacy format
        return {
            'pipeline_run_id': self.pipeline_run_id,
            'status': 'completed',
            'execution_time_seconds': results.get('execution_time_seconds', 0),
            'stage_statistics': self.stage_statistics,
            'overall_statistics': results
        }

    async def validate_kappa_for_session(self, session_id: str) -> Dict[str, Any]:
        """Validate Fleiss Kappa for a specific session."""
        return await self.unified_orchestrator._components.get("kappa_validator").validate_session_quality(session_id)

    async def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        return await self.unified_orchestrator.get_pipeline_status()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.unified_orchestrator.cleanup()


# Legacy method implementations have been migrated to UnifiedPipelineOrchestrator
# This class now serves as a compatibility wrapper
