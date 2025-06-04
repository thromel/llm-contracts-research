"""Unified configuration management system."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from pydantic import BaseModel, Field, field_validator, ValidationError

from .types import ScreeningMode, LLMProvider, LogLevel, ConfigValidationError


class LLMConfig(BaseModel):
    """Configuration for a specific LLM."""
    provider: LLMProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    timeout: int = Field(default=30, gt=0)
    rate_limit_rpm: int = Field(default=60, gt=0)
    rate_limit_tpm: int = Field(default=60000, gt=0)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if not v or v.startswith('your_') or v == 'placeholder':
            raise ValueError("Valid API key is required")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""
    connection_string: str = Field(default="mongodb://localhost:27017/")
    database_name: str = Field(default="llm_contracts_research")
    enable_indexes: bool = True
    enable_sharding: bool = False
    batch_insert_size: int = Field(default=1000, gt=0)
    connection_pool_size: int = Field(default=10, gt=0)
    query_timeout: int = Field(default=30, gt=0)

    @field_validator('connection_string')
    @classmethod
    def validate_connection_string(cls, v):
        if not v or not v.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError("Valid MongoDB connection string is required")
        return v


class DataAcquisitionConfig(BaseModel):
    """Configuration for data acquisition."""
    github_token: str = ""
    github_repositories: List[str] = Field(default_factory=list)
    github_since_days: int = Field(default=30, gt=0)
    github_max_issues_per_repo: int = Field(default=1000, gt=0)
    github_include_discussions: bool = True
    
    stackoverflow_api_key: str = ""
    stackoverflow_tags: List[str] = Field(default_factory=list)
    stackoverflow_since_days: int = Field(default=30, gt=0)
    stackoverflow_max_questions: int = Field(default=5000, gt=0)
    stackoverflow_include_answers: bool = True


class FilteringConfig(BaseModel):
    """Configuration for keyword pre-filtering."""
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    batch_size: int = Field(default=1000, gt=0)
    target_noise_reduction: float = Field(default=0.70, ge=0.0, le=1.0)
    target_recall: float = Field(default=0.93, ge=0.0, le=1.0)
    
    use_llm_contract_keywords: bool = True
    use_ml_root_cause_keywords: bool = True
    use_error_indicators: bool = True
    use_contract_patterns: bool = True
    
    snippet_context_size: int = Field(default=100, gt=0)
    max_snippets_per_post: int = Field(default=5, gt=0)
    max_potential_contracts: int = Field(default=3, gt=0)


class TraditionalScreeningConfig(BaseModel):
    """Configuration for traditional screening."""
    bulk_screener_llm: Optional[LLMConfig] = None
    borderline_screener_llm: Optional[LLMConfig] = None
    
    borderline_confidence_min: float = Field(default=0.3, ge=0.0, le=1.0)
    borderline_confidence_max: float = Field(default=0.7, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    bulk_batch_size: int = Field(default=100, gt=0)
    borderline_batch_size: int = Field(default=25, gt=0)
    concurrent_requests: int = Field(default=10, gt=0)
    rate_limit_delay: float = Field(default=2.0, ge=0.0)
    max_concurrent_requests: int = Field(default=10, gt=0)


class AgenticScreeningConfig(BaseModel):
    """Configuration for agentic screening."""
    contract_detector_llm: Optional[LLMConfig] = None
    technical_analyst_llm: Optional[LLMConfig] = None
    relevance_judge_llm: Optional[LLMConfig] = None
    decision_synthesizer_llm: Optional[LLMConfig] = None
    
    parallel_agent_execution: bool = True
    agent_timeout: int = Field(default=60, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    fallback_on_failure: bool = True
    
    min_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    require_unanimous_negative: bool = False
    
    batch_size: int = Field(default=50, gt=0)
    concurrent_posts: int = Field(default=5, gt=0)
    save_detailed_results: bool = True


class ReliabilityConfig(BaseModel):
    """Configuration for reliability validation."""
    fleiss_kappa_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    use_bootstrap_for_small_samples: bool = True
    bootstrap_iterations: int = Field(default=1000, gt=0)
    small_sample_threshold: int = Field(default=50, gt=0)
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    
    min_sample_size: int = Field(default=30, gt=0)
    require_balanced_categories: bool = True
    max_category_dominance: float = Field(default=0.70, ge=0.0, le=1.0)
    min_category_usage: float = Field(default=0.05, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    
    # Core settings
    screening_mode: ScreeningMode = ScreeningMode.AGENTIC
    project_name: str = "LLM Contracts Research"
    pipeline_version: str = "1.0.0"
    
    # Module configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_acquisition: DataAcquisitionConfig = Field(default_factory=DataAcquisitionConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    traditional_screening: TraditionalScreeningConfig = Field(default_factory=TraditionalScreeningConfig)
    agentic_screening: AgenticScreeningConfig = Field(default_factory=AgenticScreeningConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    
    # Global settings
    log_level: LogLevel = LogLevel.INFO
    enable_provenance_tracking: bool = True
    enable_detailed_logging: bool = True
    
    # Performance limits
    max_posts_per_run: int = Field(default=50000, gt=0)
    max_concurrent_operations: int = Field(default=10, gt=0)
    operation_timeout: int = Field(default=300, gt=0)

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

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Validate database configuration
        try:
            self.database.model_dump()
        except ValidationError as e:
            issues.extend([f"Database config: {error['msg']}" for error in e.errors()])

        # Validate LLM configurations
        active_configs = self.get_active_llm_configs()
        if not active_configs:
            issues.append(f"No LLM configurations found for screening mode: {self.screening_mode.value}")

        for name, config in active_configs.items():
            try:
                config.model_dump()
            except ValidationError as e:
                issues.extend([f"{name} LLM config: {error['msg']}" for error in e.errors()])

        return issues


class ConfigManager:
    """Unified configuration manager."""
    
    def __init__(self):
        self.config: Optional[PipelineConfig] = None
        self.logger = logging.getLogger(__name__)
        self._env_loaded = False
        self._yaml_loaded = False

    def load_from_env(self, env_file: Optional[str] = None) -> None:
        """Load configuration from environment variables."""
        if env_file and Path(env_file).exists():
            self._load_env_file(env_file)
        
        self.config = self._create_config_from_env()
        self._env_loaded = True
        self.logger.info("Configuration loaded from environment variables")

    def load_from_yaml(self, yaml_file: str) -> None:
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_file}")
        
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Merge with existing config if available
        if self.config:
            self._merge_yaml_config(yaml_data)
        else:
            self.config = self._create_config_from_yaml(yaml_data)
        
        self._yaml_loaded = True
        self.logger.info(f"Configuration loaded from YAML file: {yaml_file}")

    def validate(self) -> None:
        """Validate the current configuration."""
        if not self.config:
            raise ConfigValidationError("config", "No configuration loaded")
        
        issues = self.config.validate_configuration()
        if issues:
            raise ConfigValidationError("validation", f"Configuration validation failed: {'; '.join(issues)}")
        
        self.logger.info("Configuration validation passed")

    def get_config(self) -> PipelineConfig:
        """Get the current configuration."""
        if not self.config:
            raise ConfigValidationError("config", "No configuration loaded")
        return self.config

    def _load_env_file(self, env_file: str) -> None:
        """Load environment variables from file."""
        env_path = Path(env_file)
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    def _create_config_from_env(self) -> PipelineConfig:
        """Create configuration from environment variables."""
        # Basic config
        config_data = {
            'project_name': os.getenv('PROJECT_NAME', 'LLM Contracts Research'),
            'pipeline_version': os.getenv('PIPELINE_VERSION', '1.0.0'),
            'log_level': LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            'enable_detailed_logging': os.getenv('ENABLE_DETAILED_LOGGING', 'true').lower() == 'true',
            'max_posts_per_run': int(os.getenv('MAX_POSTS_PER_RUN', '50000')),
        }

        # Screening mode
        mode_str = os.getenv('SCREENING_MODE', 'agentic').lower()
        config_data['screening_mode'] = ScreeningMode(mode_str)

        # Database config - allow basic creation without strict validation
        config_data['database'] = DatabaseConfig(
            connection_string=os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/'),
            database_name=os.getenv('DATABASE_NAME', 'llm_contracts_research'),
            connection_pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
        )

        # Data acquisition config
        config_data['data_acquisition'] = DataAcquisitionConfig(
            github_token=os.getenv('GITHUB_TOKEN', ''),
            stackoverflow_api_key=os.getenv('STACKOVERFLOW_API_KEY', ''),
        )

        # Setup LLM configs
        config = PipelineConfig(**config_data)
        self._setup_llm_configs_from_env(config)
        
        return config

    def _setup_llm_configs_from_env(self, config: PipelineConfig) -> None:
        """Setup LLM configurations from environment variables."""
        
        # Traditional screening LLMs
        if os.getenv('DEEPSEEK_API_KEY'):
            config.traditional_screening.bulk_screener_llm = LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name=os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner'),
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
                temperature=float(os.getenv('DEEPSEEK_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('DEEPSEEK_MAX_TOKENS', '1000'))
            )

        if os.getenv('OPENAI_API_KEY'):
            config.traditional_screening.borderline_screener_llm = LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=os.getenv('OPENAI_MODEL', 'gpt-4-1106-preview'),
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1500'))
            )

        # Agentic screening LLMs
        if config.screening_mode in [ScreeningMode.AGENTIC, ScreeningMode.HYBRID]:
            base_llm_config = self._get_agentic_llm_config()
            
            if base_llm_config:
                config.agentic_screening.contract_detector_llm = base_llm_config
                config.agentic_screening.technical_analyst_llm = base_llm_config
                config.agentic_screening.relevance_judge_llm = base_llm_config
                config.agentic_screening.decision_synthesizer_llm = base_llm_config

    def _get_agentic_llm_config(self) -> Optional[LLMConfig]:
        """Get LLM config for agentic screening based on provider preference."""
        provider = os.getenv('AGENTIC_PROVIDER', 'openai').lower()
        
        if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=os.getenv('AGENTIC_MODEL', 'gpt-4-1106-preview'),
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )
        elif provider == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=os.getenv('AGENTIC_MODEL', 'claude-3-sonnet-20240229'),
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                base_url=os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )
        elif os.getenv('DEEPSEEK_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name=os.getenv('AGENTIC_MODEL', 'deepseek-reasoner'),
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
                temperature=float(os.getenv('AGENTIC_TEMPERATURE', '0.1')),
                max_tokens=int(os.getenv('AGENTIC_MAX_TOKENS', '2000'))
            )
        
        return None

    def _create_config_from_yaml(self, yaml_data: Dict[str, Any]) -> PipelineConfig:
        """Create configuration from YAML data."""
        # Transform YAML structure to match PipelineConfig
        config_data = {}
        
        # Map YAML fields to config structure
        if 'llm_screening' in yaml_data:
            llm_config = yaml_data['llm_screening']
            mode = llm_config.get('mode', 'traditional')
            config_data['screening_mode'] = ScreeningMode(mode)
        
        # Create base config and populate from YAML
        config = PipelineConfig(**config_data)
        
        # Load data acquisition settings from YAML sources
        if 'sources' in yaml_data:
            sources = yaml_data['sources']
            if 'github' in sources and sources['github'].get('enabled'):
                github_repos = [f"{repo['owner']}/{repo['repo']}" 
                              for repo in sources['github'].get('repositories', [])]
                config.data_acquisition.github_repositories = github_repos
                config.data_acquisition.github_max_issues_per_repo = sources['github'].get('max_issues_per_repo', 1000)
                config.data_acquisition.github_since_days = sources['github'].get('days_back', 30)
            
            if 'stackoverflow' in sources and sources['stackoverflow'].get('enabled'):
                config.data_acquisition.stackoverflow_tags = sources['stackoverflow'].get('tags', [])
                config.data_acquisition.stackoverflow_max_questions = sources['stackoverflow'].get('max_questions_per_tag', 5000)
                config.data_acquisition.stackoverflow_since_days = sources['stackoverflow'].get('days_back', 30)

        # Load keyword filtering settings
        if 'keyword_filtering' in yaml_data:
            kf_config = yaml_data['keyword_filtering']
            config.filtering.confidence_threshold = kf_config.get('confidence_threshold', 0.3)
            config.filtering.batch_size = kf_config.get('batch_size', 1000)

        return config

    def _merge_yaml_config(self, yaml_data: Dict[str, Any]) -> None:
        """Merge YAML configuration with existing config."""
        # Update screening mode if specified in YAML
        if 'llm_screening' in yaml_data:
            llm_config = yaml_data['llm_screening']
            mode = llm_config.get('mode', 'traditional')
            self.config.screening_mode = ScreeningMode(mode)
        
        # Update specific fields from YAML without replacing entire config
        if 'sources' in yaml_data:
            sources = yaml_data['sources']
            if 'github' in sources and sources['github'].get('enabled'):
                github_repos = [f"{repo['owner']}/{repo['repo']}" 
                              for repo in sources['github'].get('repositories', [])]
                self.config.data_acquisition.github_repositories = github_repos
                self.config.data_acquisition.github_max_issues_per_repo = sources['github'].get('max_issues_per_repo', self.config.data_acquisition.github_max_issues_per_repo)
                self.config.data_acquisition.github_since_days = sources['github'].get('days_back', self.config.data_acquisition.github_since_days)
            
            if 'stackoverflow' in sources and sources['stackoverflow'].get('enabled'):
                self.config.data_acquisition.stackoverflow_tags = sources['stackoverflow'].get('tags', self.config.data_acquisition.stackoverflow_tags)
                self.config.data_acquisition.stackoverflow_max_questions = sources['stackoverflow'].get('max_questions_per_tag', self.config.data_acquisition.stackoverflow_max_questions)
                self.config.data_acquisition.stackoverflow_since_days = sources['stackoverflow'].get('days_back', self.config.data_acquisition.stackoverflow_since_days)
        
        # Update filtering config
        if 'keyword_filtering' in yaml_data:
            kf_config = yaml_data['keyword_filtering']
            self.config.filtering.confidence_threshold = kf_config.get('confidence_threshold', self.config.filtering.confidence_threshold)
            self.config.filtering.batch_size = kf_config.get('batch_size', self.config.filtering.batch_size)


# Global configuration instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return _config_manager


def load_config(env_file: Optional[str] = None, yaml_file: Optional[str] = None) -> PipelineConfig:
    """Load configuration from files and return the config."""
    manager = get_config_manager()
    
    if env_file or not manager._env_loaded:
        manager.load_from_env(env_file)
    
    if yaml_file:
        manager.load_from_yaml(yaml_file)
    
    manager.validate()
    return manager.get_config()


# Convenience functions for common configuration scenarios
def get_development_config() -> PipelineConfig:
    """Get configuration optimized for development."""
    config = load_config()
    
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
    """Get configuration optimized for production."""
    config = load_config()
    
    # Production optimizations
    config.agentic_screening.concurrent_posts = 10
    config.traditional_screening.concurrent_requests = 20
    config.database.connection_pool_size = 20
    config.enable_detailed_logging = False
    
    return config


def get_research_config() -> PipelineConfig:
    """Get configuration optimized for research quality."""
    config = load_config()
    
    # Research-focused settings
    config.screening_mode = ScreeningMode.AGENTIC
    config.agentic_screening.save_detailed_results = True
    config.agentic_screening.fallback_on_failure = True
    config.reliability.fleiss_kappa_threshold = 0.85
    config.filtering.target_recall = 0.95
    
    return config