#!/usr/bin/env python3
"""
Enhanced Multi-Source Data Pipeline Runner with Contract Classification

Runs the complete LLM contracts research pipeline with integrated contract
classification and analysis capabilities.
"""

from pipeline.foundation.config import ConfigManager
from pipeline.orchestration.pipeline_orchestrator import UnifiedPipelineOrchestrator, PipelineMode
from pipeline.domain.models import PipelineStage, FilteredPost, LLMScreeningResult
from pipeline.llm_screening.contract_analysis import ContractAnalyzer, analyze_post_batch
from pipeline.storage.repositories import FilteredPostRepository, LabelledPostRepository
import asyncio
import logging
import os
import yaml
from typing import Dict, Any, List, Optional
import argparse
import json
from datetime import datetime

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


class EnhancedPipelineRunner:
    """
    Enhanced pipeline runner with contract classification capabilities.
    
    Extends the modern pipeline runner with:
    - Contract violation detection and classification
    - Novel pattern discovery
    - Research value assessment
    - Detailed analytics and reporting
    """

    def __init__(self, config_file: str = "pipeline_config.yaml"):
        self.config_file = config_file
        self.yaml_config = self._load_yaml_config()
        
        # Convert YAML config to new ConfigManager
        self.config = self._convert_yaml_to_config_manager()
        
        # Initialize unified orchestrator
        self.orchestrator = UnifiedPipelineOrchestrator(config=self.config)
        
        # Initialize contract analyzer
        self.contract_analyzer = ContractAnalyzer()
        
        # Initialize repositories for data access
        self.filtered_repo = FilteredPostRepository()
        self.labelled_repo = LabelledPostRepository()
        
        # Track analysis results
        self.analysis_results = []

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
        """Create default pipeline configuration with contract analysis."""
        config = {
            'sources': {
                'github': {
                    'enabled': True,
                    'repositories': [
                        {'owner': 'openai', 'repo': 'openai-python'},
                        {'owner': 'anthropics', 'repo': 'anthropic-sdk-python'},
                        {'owner': 'google', 'repo': 'generative-ai-python'},
                        {'owner': 'langchain-ai', 'repo': 'langchain'},
                        {'owner': 'microsoft', 'repo': 'semantic-kernel'}
                    ],
                    'max_issues_per_repo': 100,
                    'days_back': 30
                },
                'stackoverflow': {
                    'enabled': True,
                    'tags': [
                        'openai-api',
                        'langchain',
                        'llm',
                        'gpt-4',
                        'claude-ai',
                        'anthropic'
                    ],
                    'max_questions_per_tag': 200,
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
                'llm_screening': True,
                'contract_analysis': True  # New step
            },
            'llm_screening': {
                'mode': 'agentic',  # Use agentic for better classification
                'model': 'gpt-4-turbo-2024-04-09',
                'temperature': 0.1,
                'max_tokens': 2000,
                'provider': 'openai'
            },
            'contract_analysis': {
                'enable_novel_discovery': True,
                'min_confidence': 0.3,
                'save_results': True,
                'export_format': 'json'
            }
        }
        return config

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
            
        # Set contract analysis configuration
        contract_config = self.yaml_config.get('contract_analysis', {})
        config_manager.set("analysis.novel_discovery", 
                          contract_config.get('enable_novel_discovery', True))
        config_manager.set("analysis.min_confidence", 
                          contract_config.get('min_confidence', 0.3))
            
        return config_manager

    def _apply_yaml_llm_config(self):
        """Apply LLM configuration from pipeline config to environment variables."""
        llm_config = self.yaml_config.get('llm_screening', {})

        # Set screening mode
        screening_mode = llm_config.get('mode', 'agentic')
        os.environ['SCREENING_MODE'] = screening_mode
        logger.info(f"Set screening mode: {screening_mode}")

        # Set batch size and rate limiting from config
        if 'batch_size' in llm_config:
            batch_size = str(llm_config.get('batch_size', 10))
            os.environ['LLM_BATCH_SIZE'] = batch_size
            logger.info(f"Set batch size: {batch_size}")

        if 'rate_limit_delay' in llm_config:
            rate_delay = str(llm_config.get('rate_limit_delay', 1.0))
            os.environ['LLM_RATE_LIMIT_DELAY'] = rate_delay
            logger.info(f"Set rate limit delay: {rate_delay}")

        if 'max_concurrent_requests' in llm_config:
            max_concurrent = str(llm_config.get('max_concurrent_requests', 5))
            os.environ['LLM_MAX_CONCURRENT_REQUESTS'] = max_concurrent
            logger.info(f"Set max concurrent requests: {max_concurrent}")

        # Set OpenAI configuration
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
        logger.info("Enhanced pipeline initialized successfully")

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

    async def step_contract_analysis(self, max_posts: int = 1000) -> Dict[str, Any]:
        """Step 4: Analyze posts for contract violations."""
        logger.info("=== STEP 4: Contract Analysis ===")
        
        # Get screened posts
        screened_posts = await self.labelled_repo.get_all(limit=max_posts)
        
        if not screened_posts:
            logger.warning("No screened posts found for analysis")
            return {
                'analyzed': 0,
                'violations_found': 0,
                'novel_discoveries': 0
            }
        
        # Prepare data for analysis
        filtered_posts = []
        llm_results = {}
        
        for labelled_post in screened_posts:
            # Get the filtered post
            filtered_post = await self.filtered_repo.get_by_id(labelled_post.filtered_post_id)
            if filtered_post:
                filtered_posts.append(filtered_post)
                
                # Use the best available screening result
                llm_result = (
                    labelled_post.agentic_screening or
                    labelled_post.borderline_screening or
                    labelled_post.bulk_screening
                )
                if llm_result:
                    llm_results[filtered_post.id] = llm_result
        
        # Run contract analysis
        logger.info(f"Analyzing {len(filtered_posts)} posts for contract violations...")
        self.analysis_results = analyze_post_batch(filtered_posts, llm_results)
        
        # Calculate statistics
        total_violations = sum(r.total_violations for r in self.analysis_results)
        novel_violations = sum(r.novel_violations for r in self.analysis_results)
        posts_with_violations = sum(1 for r in self.analysis_results if r.has_violations)
        
        # Get novel contracts summary
        novel_summary = self.contract_analyzer.get_novel_contracts_summary()
        
        # Save results if configured
        if self.yaml_config.get('contract_analysis', {}).get('save_results', True):
            await self._save_analysis_results()
        
        results = {
            'analyzed': len(self.analysis_results),
            'posts_with_violations': posts_with_violations,
            'total_violations': total_violations,
            'novel_violations': novel_violations,
            'novel_categories': len(novel_summary['categories']),
            'average_violations_per_post': total_violations / len(self.analysis_results) if self.analysis_results else 0
        }
        
        logger.info(f"Contract analysis complete: {results}")
        return results

    async def _save_analysis_results(self):
        """Save contract analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"contract_analysis_results_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            'timestamp': timestamp,
            'summary': {
                'total_posts': len(self.analysis_results),
                'posts_with_violations': sum(1 for r in self.analysis_results if r.has_violations),
                'total_violations': sum(r.total_violations for r in self.analysis_results),
                'novel_violations': sum(r.novel_violations for r in self.analysis_results)
            },
            'novel_contracts': self.contract_analyzer.get_novel_contracts_summary(),
            'posts': []
        }
        
        # Add individual post results
        for result in self.analysis_results:
            if result.has_violations:
                post_data = {
                    'post_id': result.post_id,
                    'total_violations': result.total_violations,
                    'novel_violations': result.novel_violations,
                    'pattern': result.violation_pattern,
                    'research_value': result.research_value_score,
                    'violations': []
                }
                
                for violation in result.violations:
                    violation_data = {
                        'is_novel': violation.is_novel,
                        'contract_type': violation.contract_type.value if violation.contract_type else None,
                        'category': violation.contract_category,
                        'confidence': violation.confidence,
                        'severity': violation.severity.value if violation.severity else None
                    }
                    
                    if violation.is_novel:
                        violation_data['novel_name'] = violation.novel_name
                        violation_data['novel_description'] = violation.novel_description
                    
                    post_data['violations'].append(violation_data)
                
                export_data['posts'].append(post_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {filename}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics including contract analysis."""
        base_stats = await self.orchestrator.get_pipeline_status()
        
        # Add contract analysis statistics
        if self.analysis_results:
            contract_stats = {
                'contract_analysis': {
                    'posts_analyzed': len(self.analysis_results),
                    'total_violations': sum(r.total_violations for r in self.analysis_results),
                    'novel_violations': sum(r.novel_violations for r in self.analysis_results),
                    'high_value_posts': sum(1 for r in self.analysis_results if r.research_value_score > 0.7),
                    'novel_categories': len(self.contract_analyzer.get_novel_contracts_summary()['categories'])
                }
            }
            base_stats.update(contract_stats)
        
        return base_stats

    async def run_full_pipeline(self):
        """Run the complete pipeline end-to-end with contract analysis."""
        logger.info("=== RUNNING ENHANCED FULL PIPELINE ===")
        
        # Determine which stages to run based on YAML config
        stages = []
        pipeline_steps = self.yaml_config.get('pipeline_steps', {})
        
        if pipeline_steps.get('data_acquisition', True):
            stages.append(PipelineStage.DATA_ACQUISITION)
        if pipeline_steps.get('keyword_filtering', True):
            stages.append(PipelineStage.DATA_PREPROCESSING)
        if pipeline_steps.get('llm_screening', True):
            stages.append(PipelineStage.LLM_SCREENING)
            
        # Run the base pipeline
        results = await self.orchestrator.execute_pipeline(
            mode=PipelineMode.RESEARCH,
            stages=stages,
            max_posts_per_stage=1000,
            skip_validation=False
        )
        
        # Run contract analysis if enabled
        if pipeline_steps.get('contract_analysis', True):
            analysis_results = await self.step_contract_analysis()
            results['contract_analysis'] = analysis_results
        
        logger.info("=== ENHANCED PIPELINE COMPLETE ===")
        logger.info(f"Results: {results}")
        return results

    async def generate_report(self) -> str:
        """Generate a comprehensive report of pipeline results."""
        stats = await self.get_stats()
        
        report = ["# LLM Contract Research Pipeline Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Data acquisition stats
        if 'raw_posts' in stats:
            report.append("## Data Acquisition")
            report.append(f"- Total posts collected: {stats['raw_posts']}")
            report.append(f"- GitHub issues: {stats.get('github_posts', 0)}")
            report.append(f"- Stack Overflow questions: {stats.get('stackoverflow_posts', 0)}\n")
        
        # Filtering stats
        if 'filtered_posts' in stats:
            report.append("## Keyword Filtering")
            report.append(f"- Posts after filtering: {stats['filtered_posts']}")
            report.append(f"- Filter rate: {(1 - stats['filtered_posts']/stats.get('raw_posts', 1))*100:.1f}%\n")
        
        # Screening stats
        if 'screened_posts' in stats:
            report.append("## LLM Screening")
            report.append(f"- Posts screened: {stats['screened_posts']}")
            report.append(f"- Positive classifications: {stats.get('positive_posts', 0)}\n")
        
        # Contract analysis stats
        if 'contract_analysis' in stats:
            ca = stats['contract_analysis']
            report.append("## Contract Analysis")
            report.append(f"- Posts analyzed: {ca['posts_analyzed']}")
            report.append(f"- Total violations found: {ca['total_violations']}")
            report.append(f"- Novel violations: {ca['novel_violations']}")
            report.append(f"- Novel categories discovered: {ca['novel_categories']}")
            report.append(f"- High research value posts: {ca['high_value_posts']}\n")
        
        # Novel discoveries
        if self.contract_analyzer.novel_contracts_found:
            report.append("## Novel Contract Discoveries")
            novel_summary = self.contract_analyzer.get_novel_contracts_summary()
            for category, contracts in novel_summary['categories'].items():
                report.append(f"\n### {category}")
                for contract in contracts[:3]:  # Top 3 per category
                    report.append(f"- {contract['name']}: {contract['description'][:80]}...")
        
        return "\n".join(report)

    async def shutdown(self):
        """Cleanup and shutdown."""
        await self.orchestrator.cleanup()


async def main():
    """Main pipeline execution with CLI arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Source Data Pipeline with Contract Analysis")
    parser.add_argument('--step', choices=['acquisition', 'filtering', 'screening', 'analysis', 'full'],
                        default='full', help='Pipeline step to run')
    parser.add_argument('--config', default='pipeline_config.yaml',
                        help='Path to pipeline configuration file')
    parser.add_argument('--max-posts', type=int,
                        help='Maximum posts to process')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics')
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate comprehensive report')

    args = parser.parse_args()

    logger.info("Starting Enhanced Multi-Source Data Pipeline")

    runner = EnhancedPipelineRunner(config_file=args.config)

    try:
        await runner.initialize()

        if args.stats_only:
            stats = await runner.get_stats()
            logger.info("Current Pipeline Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            return

        if args.generate_report:
            report = await runner.generate_report()
            print(report)
            # Save report
            with open(f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", 'w') as f:
                f.write(report)
            return

        if args.step == 'acquisition':
            await runner.step_data_acquisition()
        elif args.step == 'filtering':
            await runner.step_keyword_filtering()
        elif args.step == 'screening':
            await runner.step_llm_screening(max_posts=args.max_posts)
        elif args.step == 'analysis':
            await runner.step_contract_analysis(max_posts=args.max_posts)
        elif args.step == 'full':
            await runner.run_full_pipeline()

        # Show final stats
        stats = await runner.get_stats()
        logger.info("\nFinal Pipeline Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"\n{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info("\nPipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())