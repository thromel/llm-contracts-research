"""
Pipeline Configuration for LLM Contracts Research.

Supports multiple screening modes:
- Traditional (DeepSeek-R1 bulk + GPT-4.1 borderline)
- Agentic (LangChain multi-agent pipeline)
- Hybrid (combination of both approaches)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging


class ScreeningMode(Enum):
    """LLM screening modes."""
    TRADITIONAL = "traditional"  # Bulk + Borderline screeners
    AGENTIC = "agentic"          # Multi-agent LangChain pipeline
    HYBRID = "hybrid"            # Both traditional and agentic


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM."""
    provider: LLMProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 60000  # Tokens per minute


@dataclass
class AgenticConfig:
    """Configuration for agentic screening."""
    # Agent-specific LLM configs
    contract_detector_llm: LLMConfig = None
    technical_analyst_llm: LLMConfig = None
    relevance_judge_llm: LLMConfig = None
    decision_synthesizer_llm: LLMConfig = None

    # Agent behavior settings
    parallel_agent_execution: bool = True
    agent_timeout: int = 60
    retry_attempts: int = 3
    fallback_on_failure: bool = True

    # Quality settings
    min_confidence_threshold: float = 0.4
    high_confidence_threshold: float = 0.8
    require_unanimous_negative: bool = False

    # Performance settings
    batch_size: int = 50
    concurrent_posts: int = 5
    save_detailed_results: bool = True


@dataclass
class TraditionalConfig:
    """Configuration for traditional screening."""
    # Bulk screener (DeepSeek-R1)
    bulk_screener_llm: LLMConfig = None

    # Borderline screener (GPT-4.1)
    borderline_screener_llm: LLMConfig = None

    # Thresholds
    borderline_confidence_min: float = 0.3
    borderline_confidence_max: float = 0.7
    high_confidence_threshold: float = 0.8

    # Performance settings
    bulk_batch_size: int = 100
    borderline_batch_size: int = 25
    concurrent_requests: int = 10


@dataclass
class DataAcquisitionConfig:
    """Configuration for data acquisition."""
    # GitHub settings
    github_token: str = ""
    github_repositories: List[str] = field(default_factory=lambda: [
        'openai/openai-python',
        'openai/openai-cookbook',
        'anthropics/anthropic-sdk-python',
        'langchain-ai/langchain',
        'huggingface/transformers'
    ])
    github_since_days: int = 30
    github_max_issues_per_repo: int = 1000
    github_include_discussions: bool = True

    # Stack Overflow settings
    stackoverflow_api_key: str = ""
    stackoverflow_tags: List[str] = field(default_factory=lambda: [
        'openai-api', 'gpt-3', 'gpt-4', 'chatgpt', 'claude',
        'langchain', 'transformers', 'huggingface'
    ])
    stackoverflow_since_days: int = 30
    stackoverflow_max_questions: int = 5000
    stackoverflow_include_answers: bool = True


@dataclass
class FilteringConfig:
    """Configuration for keyword pre-filtering."""
    confidence_threshold: float = 0.3
    batch_size: int = 1000
    target_noise_reduction: float = 0.70  # 70% noise reduction target
    target_recall: float = 0.93  # 93%+ recall target

    # Keyword settings
    use_llm_contract_keywords: bool = True
    use_ml_root_cause_keywords: bool = True
    use_error_indicators: bool = True
    use_contract_patterns: bool = True

    # Context extraction
    snippet_context_size: int = 100
    max_snippets_per_post: int = 5
    max_potential_contracts: int = 3


@dataclass
class ReliabilityConfig:
    """Configuration for reliability validation."""
    fleiss_kappa_threshold: float = 0.80
    use_bootstrap_for_small_samples: bool = True
    bootstrap_iterations: int = 1000
    small_sample_threshold: int = 50
    confidence_level: float = 0.95

    # Quality gates
    min_sample_size: int = 30
    require_balanced_categories: bool = True
    max_category_dominance: float = 0.70
    min_category_usage: float = 0.05


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str = ""
    database_name: str = "llm_contracts_research"

    # Collection settings
    enable_indexes: bool = True
    enable_sharding: bool = False

    # Performance settings
    batch_insert_size: int = 1000
    connection_pool_size: int = 10
    query_timeout: int = 30


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Core settings
    screening_mode: ScreeningMode = ScreeningMode.AGENTIC
    project_name: str = "LLM Contracts Research"
    pipeline_version: str = "1.0.0"

    # Module configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_acquisition: DataAcquisitionConfig = field(
        default_factory=DataAcquisitionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    traditional_screening: TraditionalConfig = field(
        default_factory=TraditionalConfig)
    agentic_screening: AgenticConfig = field(default_factory=AgenticConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)

    # Global settings
    log_level: str = "INFO"
    enable_provenance_tracking: bool = True
    enable_detailed_logging: bool = True

    # Performance limits
    max_posts_per_run: int = 50000
    max_concurrent_operations: int = 10
    operation_timeout: int = 300

    @classmethod
    def from_environment(cls) -> 'PipelineConfig':
        """Create configuration from environment variables."""
        config = cls()

        # Database configuration
        config.database.connection_string = os.getenv(
            'MONGODB_URI',
            'mongodb://localhost:27017/'
        )
        config.database.database_name = os.getenv(
            'DATABASE_NAME',
            'llm_contracts_research'
        )

        # API keys
        config.data_acquisition.github_token = os.getenv('GITHUB_TOKEN', '')
        config.data_acquisition.stackoverflow_api_key = os.getenv(
            'STACKOVERFLOW_API_KEY', '')

        # Screening mode
        mode_str = os.getenv('SCREENING_MODE', 'agentic').lower()
        if mode_str == 'traditional':
            config.screening_mode = ScreeningMode.TRADITIONAL
        elif mode_str == 'hybrid':
            config.screening_mode = ScreeningMode.HYBRID
        else:
            config.screening_mode = ScreeningMode.AGENTIC

        # LLM configurations
        config._setup_llm_configs_from_env()

        return config

    def _setup_llm_configs_from_env(self):
        """Setup LLM configurations from environment variables."""

        logger = logging.getLogger(__name__)
        logger.info("🔧 Setting up LLM configurations from environment...")

        # Traditional screening LLMs
        if os.getenv('DEEPSEEK_API_KEY'):
            logger.info("✅ Setting up DeepSeek for traditional screening")
            self.traditional_screening.bulk_screener_llm = LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name=os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner'),
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url=os.getenv('DEEPSEEK_BASE_URL',
                                   'https://api.deepseek.com/v1'),
                temperature=float(os.getenv('DEEPSEEK_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('DEEPSEEK_MAX_TOKENS', '1000'))
            )

        if os.getenv('OPENAI_API_KEY'):
            logger.info("✅ Setting up OpenAI for traditional screening")
            self.traditional_screening.borderline_screener_llm = LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=os.getenv('OPENAI_MODEL', 'gpt-4-1106-preview'),
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1500'))
            )

        # Agentic screening LLMs (can use different models for different agents)
        base_llm_config = None

        logger.info(
            f"🔍 Checking agentic provider: {os.getenv('AGENTIC_PROVIDER', 'openai')}")
        logger.info(
            f"🔑 OpenAI API key present: {bool(os.getenv('OPENAI_API_KEY'))}")

        if os.getenv('AGENTIC_PROVIDER', 'openai').lower() == 'openai' and os.getenv('OPENAI_API_KEY'):
            logger.info("✅ Setting up OpenAI for agentic screening")
            base_llm_config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=os.getenv('AGENTIC_MODEL', 'gpt-4-1106-preview'),
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )
            logger.info(
                f"📋 Agentic config: {base_llm_config.model_name} with API key: {base_llm_config.api_key[:10]}...")
        elif os.getenv('AGENTIC_PROVIDER', '').lower() == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
            logger.info("✅ Setting up Anthropic for agentic screening")
            base_llm_config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=os.getenv(
                    'AGENTIC_MODEL', 'claude-3-sonnet-20240229'),
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                base_url=os.getenv('ANTHROPIC_BASE_URL',
                                   'https://api.anthropic.com'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )
        elif os.getenv('DEEPSEEK_API_KEY'):
            logger.info(
                "✅ Setting up DeepSeek for agentic screening (fallback)")
            # Fallback to DeepSeek for agentic
            base_llm_config = LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name=os.getenv('AGENTIC_MODEL', 'deepseek-reasoner'),
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url=os.getenv('DEEPSEEK_BASE_URL',
                                   'https://api.deepseek.com/v1'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )

        # Set up agentic agents (using same config for all agents, can be customized)
        if base_llm_config:
            logger.info("✅ Setting up all agentic agents with base config")
            self.agentic_screening.contract_detector_llm = base_llm_config
            self.agentic_screening.technical_analyst_llm = base_llm_config
            self.agentic_screening.relevance_judge_llm = base_llm_config
            self.agentic_screening.decision_synthesizer_llm = base_llm_config
        else:
            logger.warning(
                "❌ No valid LLM configuration found for agentic screening")

    def get_active_llm_configs(self) -> Dict[str, LLMConfig]:
        """Get LLM configurations for the active screening mode."""
        configs = {}

        if self.screening_mode in [ScreeningMode.TRADITIONAL, ScreeningMode.HYBRID]:
            if self.traditional_screening.bulk_screener_llm:
                configs['bulk_screener'] = self.traditional_screening.bulk_screener_llm
            if self.traditional_screening.borderline_screener_llm:
                configs['borderline_screener'] = self.traditional_screening.borderline_screener_llm

        if self.screening_mode in [ScreeningMode.AGENTIC, ScreeningMode.HYBRID]:
            if self.agentic_screening.contract_detector_llm:
                configs['contract_detector'] = self.agentic_screening.contract_detector_llm
            if self.agentic_screening.technical_analyst_llm:
                configs['technical_analyst'] = self.agentic_screening.technical_analyst_llm
            if self.agentic_screening.relevance_judge_llm:
                configs['relevance_judge'] = self.agentic_screening.relevance_judge_llm
            if self.agentic_screening.decision_synthesizer_llm:
                configs['decision_synthesizer'] = self.agentic_screening.decision_synthesizer_llm

        return configs

    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Database validation
        if not self.database.connection_string:
            issues.append("Database connection string is required")

        # API key validation
        if not self.data_acquisition.github_token:
            issues.append("GitHub token is required for data acquisition")

        # LLM configuration validation
        active_configs = self.get_active_llm_configs()
        if not active_configs:
            issues.append(
                f"No LLM configurations found for screening mode: {self.screening_mode.value}")

        for name, config in active_configs.items():
            if not config.api_key:
                issues.append(f"API key missing for {name} LLM configuration")

        # Threshold validation
        if not 0.0 <= self.filtering.confidence_threshold <= 1.0:
            issues.append(
                "Filtering confidence threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.reliability.fleiss_kappa_threshold <= 1.0:
            issues.append("Fleiss kappa threshold must be between 0.0 and 1.0")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # This would implement a full serialization method
        # For now, return a summary
        return {
            'screening_mode': self.screening_mode.value,
            'project_name': self.project_name,
            'pipeline_version': self.pipeline_version,
            'database_name': self.database.database_name,
            'active_llm_configs': list(self.get_active_llm_configs().keys()),
            'validation_issues': self.validate()
        }


# Predefined configurations for common scenarios
def get_development_config() -> PipelineConfig:
    """Get configuration for development/testing."""
    config = PipelineConfig.from_environment()

    # Override for development
    config.data_acquisition.github_since_days = 7
    config.data_acquisition.github_max_issues_per_repo = 100
    config.data_acquisition.stackoverflow_max_questions = 500
    config.filtering.batch_size = 100
    config.agentic_screening.batch_size = 10
    config.traditional_screening.bulk_batch_size = 50
    config.max_posts_per_run = 1000

    return config


def get_production_config() -> PipelineConfig:
    """Get configuration for production."""
    config = PipelineConfig.from_environment()

    # Production optimizations
    config.agentic_screening.concurrent_posts = 10
    config.traditional_screening.concurrent_requests = 20
    config.database.connection_pool_size = 20
    config.enable_detailed_logging = False

    return config


def get_research_config() -> PipelineConfig:
    """Get configuration optimized for research quality."""
    config = PipelineConfig.from_environment()

    # Research-focused settings
    config.screening_mode = ScreeningMode.AGENTIC
    config.agentic_screening.save_detailed_results = True
    config.agentic_screening.fallback_on_failure = True
    config.reliability.fleiss_kappa_threshold = 0.85  # Higher threshold
    config.filtering.target_recall = 0.95  # Higher recall

    return config
