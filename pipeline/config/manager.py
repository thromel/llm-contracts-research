"""
Configuration manager for centralized configuration handling.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, TypeVar
import logging

from ..core.interfaces import ConfigProvider
from ..core.exceptions import ConfigurationError
from .providers import ChainedConfigProvider, EnvironmentConfigProvider, YAMLConfigProvider
from .schemas import PipelineConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigurationManager:
    """
    Central configuration manager for the pipeline.
    
    Manages configuration loading, validation, and access with support
    for multiple configuration sources and hierarchical overrides.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("config")
        self._providers: List[ConfigProvider] = []
        self._config_cache: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize(
        self,
        config_file: Optional[str] = None,
        use_env_vars: bool = True,
        env_prefix: str = "LLM_PIPELINE_"
    ) -> None:
        """Initialize configuration from various sources.
        
        Args:
            config_file: Primary configuration file
            use_env_vars: Whether to use environment variables
            env_prefix: Prefix for environment variables
        """
        self._providers.clear()
        
        # Add environment variable provider (highest priority)
        if use_env_vars:
            env_provider = EnvironmentConfigProvider(prefix=env_prefix)
            self._providers.append(env_provider)
            logger.info(f"Added environment config provider with prefix: {env_prefix}")
        
        # Add YAML file providers
        if config_file:
            yaml_provider = YAMLConfigProvider(config_file)
            self._providers.append(yaml_provider)
            logger.info(f"Added YAML config provider: {config_file}")
        
        # Add default configuration files
        default_files = [
            self.config_dir / "defaults.yaml",
            self.config_dir / "pipeline.yaml",
            Path("pipeline_config.yaml")
        ]
        
        for file_path in default_files:
            if file_path.exists():
                yaml_provider = YAMLConfigProvider(str(file_path))
                self._providers.append(yaml_provider)
                logger.info(f"Added default config file: {file_path}")
        
        # Create chained provider
        if self._providers:
            self._chained_provider = ChainedConfigProvider(self._providers)
        else:
            raise ConfigurationError("No configuration sources available")
        
        # Load and validate configuration
        self._load_configuration()
        self._initialized = True
        
    def _load_configuration(self) -> None:
        """Load configuration from all providers."""
        try:
            # Load base configuration
            self._config_cache = {}
            
            # Get all configuration keys
            all_keys = set()
            for provider in self._providers:
                if hasattr(provider, 'get_all_keys'):
                    all_keys.update(provider.get_all_keys())
            
            # Load each key
            for key in all_keys:
                value = self._chained_provider.get(key)
                if value is not None:
                    self._set_nested_value(self._config_cache, key, value)
            
            logger.info(f"Loaded configuration with {len(all_keys)} keys")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary.
        
        Args:
            config: Configuration dictionary
            key: Dot-separated key path
            value: Value to set
        """
        parts = key.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Dot-separated configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self._initialized:
            raise ConfigurationError("Configuration manager not initialized")
        
        # Try chained provider first
        value = self._chained_provider.get(key, default)
        if value is not None:
            return value
        
        # Try nested lookup in cache
        parts = key.split('.')
        current = self._config_cache
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_typed(self, key: str, type_class: Type[T], default: Optional[T] = None) -> T:
        """Get typed configuration value.
        
        Args:
            key: Configuration key
            type_class: Expected type
            default: Default value
            
        Returns:
            Typed configuration value
        """
        value = self.get(key, default)
        
        if value is None:
            return default
        
        if isinstance(value, type_class):
            return value
        
        # Try type conversion
        try:
            return type_class(value)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Cannot convert config value '{key}' to {type_class.__name__}: {str(e)}"
            )
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (runtime only).
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self._set_nested_value(self._config_cache, key, value)
        
        # Also set in the first provider that supports setting
        for provider in self._providers:
            try:
                provider.set(key, value)
                break
            except (AttributeError, NotImplementedError):
                continue
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get complete pipeline configuration.
        
        Returns:
            PipelineConfig instance
        """
        config_dict = {
            'screening_mode': self.get('screening.mode', 'traditional'),
            'project_name': self.get('project.name', 'LLM Contracts Research'),
            'pipeline_version': self.get('pipeline.version', '1.0.0'),
            
            # Data acquisition
            'data_acquisition': {
                'github_token': self.get('github.token', os.getenv('GITHUB_TOKEN')),
                'github_repositories': self.get('github.repositories', []),
                'github_since_days': self.get('github.since_days', 30),
                'stackoverflow_api_key': self.get('stackoverflow.api_key', os.getenv('STACKOVERFLOW_API_KEY')),
                'stackoverflow_tags': self.get('stackoverflow.tags', []),
                'stackoverflow_since_days': self.get('stackoverflow.since_days', 30),
            },
            
            # Filtering
            'filtering': {
                'confidence_threshold': self.get('filtering.confidence_threshold', 0.3),
                'batch_size': self.get('filtering.batch_size', 1000),
            },
            
            # Screening
            'screening': {
                'traditional': {
                    'bulk_batch_size': self.get('screening.traditional.bulk_batch_size', 100),
                    'borderline_batch_size': self.get('screening.traditional.borderline_batch_size', 25),
                },
                'agentic': {
                    'batch_size': self.get('screening.agentic.batch_size', 50),
                    'concurrent_posts': self.get('screening.agentic.concurrent_posts', 5),
                }
            },
            
            # Storage
            'storage': {
                'mongodb_uri': self.get('mongodb.uri', os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')),
                'database_name': self.get('mongodb.database', 'llm_contracts_research'),
            }
        }
        
        return PipelineConfig.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = [
            ('mongodb.uri', 'MongoDB connection URI'),
            ('screening.mode', 'Screening mode'),
        ]
        
        for field, description in required_fields:
            if not self.get(field):
                errors.append(f"Missing required configuration: {description} ({field})")
        
        # Validate screening mode
        screening_mode = self.get('screening.mode')
        if screening_mode not in ['traditional', 'agentic', 'hybrid']:
            errors.append(f"Invalid screening mode: {screening_mode}")
        
        # Validate API keys based on screening mode
        if screening_mode in ['traditional', 'hybrid']:
            if not self.get('openai.api_key') and not os.getenv('OPENAI_API_KEY'):
                errors.append("OpenAI API key required for traditional screening")
        
        return errors
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        if self._initialized:
            self._load_configuration()
            logger.info("Configuration reloaded")
    
    def export(self, format: str = 'dict') -> Any:
        """Export current configuration.
        
        Args:
            format: Export format ('dict', 'yaml', 'json')
            
        Returns:
            Exported configuration
        """
        if format == 'dict':
            return self._config_cache.copy()
        elif format == 'yaml':
            import yaml
            return yaml.dump(self._config_cache, default_flow_style=False)
        elif format == 'json':
            import json
            return json.dumps(self._config_cache, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance.
    
    Returns:
        Global ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager