#!/usr/bin/env python3
"""
Multi-Source Data Pipeline Runner

Runs the complete LLM contracts research pipeline using the new unified architecture.
Supports both legacy YAML configuration and new ConfigManager approach.
"""

from pipeline.foundation.config import ConfigManager
from pipeline.orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator, PipelineMode
from pipeline.domain.models import PipelineStage
import asyncio
import logging
import os
import yaml
from typing import Dict, Any
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


class ModernPipelineRunner:
    """
    Modern pipeline runner using the new unified architecture.
    
    Supports YAML configuration files for backward compatibility while
    leveraging the new foundation and infrastructure layers.
    """

    def __init__(self, config_file: str = "pipeline_config.yaml"):
        self.config_file = config_file
        self.yaml_config = self._load_yaml_config()
        
        # Convert YAML config to new ConfigManager
        self.config = self._convert_yaml_to_config_manager()
        
        # Initialize unified orchestrator
        self.orchestrator = UnifiedPipelineOrchestrator(config=self.config)

    def _load_yaml_config(self) -> Dict[str, Any]:
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

    def _convert_yaml_to_config_manager(self) -> ConfigManager:
        """Convert YAML configuration to ConfigManager format."""
        config_manager = ConfigManager()
        
        # Apply LLM configuration from YAML
        self._apply_yaml_llm_config()
        
        # Set database configuration
        config_manager.set("mongodb.uri", os.getenv('MONGODB_URI'))
        config_manager.set("mongodb.database", os.getenv('DATABASE_NAME', 'llm_contracts_research'))
        
        # Set API keys from environment
        if os.getenv('GITHUB_TOKEN'):
            config_manager.set("github.token", os.getenv('GITHUB_TOKEN'))
        if os.getenv('OPENAI_API_KEY'):
            config_manager.set("openai.api_key", os.getenv('OPENAI_API_KEY'))
        if os.getenv('STACKOVERFLOW_API_KEY'):
            config_manager.set("stackoverflow.api_key", os.getenv('STACKOVERFLOW_API_KEY'))
            
        # Set pipeline configuration from YAML
        sources = self.yaml_config.get('sources', {})
        if sources.get('github', {}).get('enabled', False):
            config_manager.set("acquisition.github.enabled", True)
        if sources.get('stackoverflow', {}).get('enabled', False):
            config_manager.set("acquisition.stackoverflow.enabled", True)
            
        # Set screening configuration
        llm_config = self.yaml_config.get('llm_screening', {})
        if llm_config.get('mode') == 'traditional':
            config_manager.set("screening.traditional.enabled", True)
        elif llm_config.get('mode') == 'agentic':
            config_manager.set("screening.agentic.enabled", True)
            
        return config_manager

    def _apply_yaml_llm_config(self):
        """Apply LLM configuration from pipeline config to environment variables."""
        llm_config = self.yaml_config.get('llm_screening', {})

        # Set screening mode
        screening_mode = llm_config.get('mode', 'traditional')
        os.environ['SCREENING_MODE'] = screening_mode
        logger.info(f"Set screening mode: {screening_mode}")

        # Set batch size and rate limiting from config
        if 'batch_size' in llm_config:
            batch_size = str(llm_config['batch_size'])
            os.environ['LLM_BATCH_SIZE'] = batch_size
            logger.info(f"Set batch size: {batch_size}")

        if 'rate_limit_delay' in llm_config:
            rate_delay = str(llm_config['rate_limit_delay'])
            os.environ['LLM_RATE_LIMIT_DELAY'] = rate_delay
            logger.info(f"Set rate limit delay: {rate_delay}")

        if 'max_concurrent_requests' in llm_config:
            max_concurrent = str(llm_config['max_concurrent_requests'])
            os.environ['LLM_MAX_CONCURRENT_REQUESTS'] = max_concurrent
            logger.info(f"Set max concurrent requests: {max_concurrent}")

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
        """Initialize the unified pipeline orchestrator."""
        await self.orchestrator.initialize()
        logger.info("Modern pipeline initialized successfully")

    async def step_data_acquisition(self, max_posts: int = 1000) -> Dict[str, Any]:
        """Step 1: Acquire data using unified orchestrator."""
        logger.info("=== STEP 1: Data Acquisition ===")
        return await self.orchestrator._execute_stage(
            PipelineStage.DATA_ACQUISITION, max_posts, False
        )

    async def step_keyword_filtering(self, max_posts: int = 1000) -> Dict[str, Any]:
        """Step 2: Apply keyword filtering using unified orchestrator."""
        logger.info("=== STEP 2: Keyword Filtering ===")
        return await self.orchestrator._execute_stage(
            PipelineStage.DATA_PREPROCESSING, max_posts, False
        )

    async def step_llm_screening(self, max_posts: int = 1000) -> Dict[str, Any]:
        """Step 3: Run LLM screening using unified orchestrator."""
        logger.info("=== STEP 3: LLM Screening ===")
        return await self.orchestrator._execute_stage(
            PipelineStage.LLM_SCREENING, max_posts, False
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return await self.orchestrator.get_pipeline_status()

    async def run_full_pipeline(self):
        """Run the complete pipeline end-to-end."""
        logger.info("=== RUNNING FULL PIPELINE ===")
        
        # Determine which stages to run based on YAML config
        stages = []
        pipeline_steps = self.yaml_config.get('pipeline_steps', {})
        
        if pipeline_steps.get('data_acquisition', True):
            stages.append(PipelineStage.DATA_ACQUISITION)
        if pipeline_steps.get('keyword_filtering', True):
            stages.append(PipelineStage.DATA_PREPROCESSING)
        if pipeline_steps.get('llm_screening', True):
            stages.append(PipelineStage.LLM_SCREENING)
            
        # Run the pipeline
        results = await self.orchestrator.execute_pipeline(
            mode=PipelineMode.RESEARCH,
            stages=stages,
            max_posts_per_stage=1000,
            skip_validation=False
        )
        
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Results: {results}")
        return results

    async def shutdown(self):
        """Cleanup and shutdown."""
        await self.orchestrator.cleanup()


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

    runner = ModernPipelineRunner(config_file=args.config)

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
