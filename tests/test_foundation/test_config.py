"""Tests for the unified configuration system."""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from pipeline.foundation.config import (
    ConfigManager, PipelineConfig, LLMConfig, 
    load_config, get_development_config, get_production_config
)
from pipeline.foundation.types import ScreeningMode, LLMProvider, ConfigValidationError


class TestLLMConfig:
    """Test LLMConfig validation."""
    
    def test_valid_config(self):
        """Test valid LLM configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="sk-test-key",
            temperature=0.1,
            max_tokens=1000
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.1
    
    def test_invalid_api_key(self):
        """Test invalid API key validation."""
        with pytest.raises(ValueError, match="Valid API key is required"):
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="your_api_key_here"
            )
    
    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        with pytest.raises(ValueError):
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="sk-test-key",
                temperature=3.0  # Invalid: > 2.0
            )


class TestPipelineConfig:
    """Test PipelineConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PipelineConfig()
        assert config.screening_mode == ScreeningMode.AGENTIC
        assert config.project_name == "LLM Contracts Research"
        assert config.log_level.value == "INFO"
    
    def test_get_active_llm_configs_traditional(self):
        """Test getting active configs for traditional mode."""
        config = PipelineConfig(screening_mode=ScreeningMode.TRADITIONAL)
        config.traditional_screening.bulk_screener_llm = LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-reasoner",
            api_key="sk-test-key"
        )
        
        active_configs = config.get_active_llm_configs()
        assert 'bulk_screener' in active_configs
        assert active_configs['bulk_screener'].provider == LLMProvider.DEEPSEEK
    
    def test_get_active_llm_configs_agentic(self):
        """Test getting active configs for agentic mode."""
        config = PipelineConfig(screening_mode=ScreeningMode.AGENTIC)
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="sk-test-key"
        )
        
        config.agentic_screening.contract_detector_llm = llm_config
        config.agentic_screening.technical_analyst_llm = llm_config
        
        active_configs = config.get_active_llm_configs()
        assert 'contract_detector' in active_configs
        assert 'technical_analyst' in active_configs
    
    def test_validation_missing_llm_configs(self):
        """Test validation when LLM configs are missing."""
        config = PipelineConfig(screening_mode=ScreeningMode.TRADITIONAL)
        # No LLM configs set
        
        issues = config.validate_configuration()
        assert any("No LLM configurations found" in issue for issue in issues)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ConfigManager()
    
    def test_load_from_env_basic(self):
        """Test loading basic configuration from environment."""
        with patch.dict(os.environ, {
            'SCREENING_MODE': 'traditional',
            'MONGODB_CONNECTION_STRING': 'mongodb://test:27017/',
            'PROJECT_NAME': 'Test Project',
            'OPENAI_API_KEY': 'sk-test-key'
        }):
            self.manager.load_from_env()
            config = self.manager.get_config()
            
            assert config.screening_mode == ScreeningMode.TRADITIONAL
            assert config.database.connection_string == 'mongodb://test:27017/'
            assert config.project_name == 'Test Project'
    
    def test_load_from_env_file(self):
        """Test loading from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("SCREENING_MODE=agentic\n")
            f.write("PROJECT_NAME=Test Project From File\n")
            f.write("OPENAI_API_KEY=sk-test-key\n")
            f.write("MONGODB_CONNECTION_STRING=mongodb://localhost:27017/\n")
            env_file = f.name
        
        try:
            self.manager.load_from_env(env_file)
            config = self.manager.get_config()
            
            assert config.screening_mode == ScreeningMode.AGENTIC
            assert config.project_name == "Test Project From File"
        finally:
            os.unlink(env_file)
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_data = {
            'sources': {
                'github': {
                    'enabled': True,
                    'repositories': [
                        {'owner': 'openai', 'repo': 'openai-python'},
                        {'owner': 'anthropics', 'repo': 'anthropic-sdk-python'}
                    ],
                    'max_issues_per_repo': 500
                },
                'stackoverflow': {
                    'enabled': True,
                    'tags': ['openai-api', 'gpt-4'],
                    'max_questions_per_tag': 1000
                }
            },
            'keyword_filtering': {
                'confidence_threshold': 0.4,
                'batch_size': 200
            },
            'llm_screening': {
                'mode': 'traditional'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_data, f)
            yaml_file = f.name
        
        try:
            # First load env config
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/'
            }):
                self.manager.load_from_env()
                self.manager.load_from_yaml(yaml_file)
            
            config = self.manager.get_config()
            
            assert config.screening_mode == ScreeningMode.TRADITIONAL
            assert 'openai/openai-python' in config.data_acquisition.github_repositories
            assert 'anthropics/anthropic-sdk-python' in config.data_acquisition.github_repositories
            assert config.data_acquisition.github_max_issues_per_repo == 500
            assert 'openai-api' in config.data_acquisition.stackoverflow_tags
            assert config.filtering.confidence_threshold == 0.4
            assert config.filtering.batch_size == 200
        finally:
            os.unlink(yaml_file)
    
    def test_validation_success(self):
        """Test successful validation."""
        with patch.dict(os.environ, {
            'SCREENING_MODE': 'traditional',
            'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/',
            'OPENAI_API_KEY': 'sk-test-key',
            'DEEPSEEK_API_KEY': 'sk-deepseek-key'
        }):
            self.manager.load_from_env()
            # Should not raise
            self.manager.validate()
    
    def test_validation_failure(self):
        """Test validation failure."""
        # Explicitly clear API keys and set required vars
        with patch.dict(os.environ, {}, clear=True):
            os.environ.update({
                'SCREENING_MODE': 'traditional',
                'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/'
            })
            self.manager.load_from_env()
            
            with pytest.raises(ConfigValidationError):
                self.manager.validate()


class TestConvenienceFunctions:
    """Test convenience configuration functions."""
    
    def test_load_config(self):
        """Test load_config function."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test-key',
            'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/'
        }):
            config = load_config()
            assert isinstance(config, PipelineConfig)
    
    def test_get_development_config(self):
        """Test development configuration."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test-key',
            'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/'
        }):
            config = get_development_config()
            
            # Check development-specific overrides
            assert config.data_acquisition.github_since_days == 7
            assert config.data_acquisition.github_max_issues_per_repo == 100
            assert config.max_posts_per_run == 1000
    
    def test_get_production_config(self):
        """Test production configuration."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test-key',
            'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017/'
        }):
            config = get_production_config()
            
            # Check production-specific optimizations
            assert config.agentic_screening.concurrent_posts == 10
            assert config.traditional_screening.concurrent_requests == 20
            assert config.database.connection_pool_size == 20
            assert config.enable_detailed_logging == False


if __name__ == "__main__":
    pytest.main([__file__])