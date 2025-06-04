"""Test configuration specific to foundation layer tests."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("SCREENING_MODE=traditional\n")
        f.write("MONGODB_CONNECTION_STRING=mongodb://localhost:27017/\n")
        f.write("OPENAI_API_KEY=sk-test-key\n")
        f.write("DEEPSEEK_API_KEY=sk-deepseek-key\n")
        env_file = f.name
    
    yield env_file
    os.unlink(env_file)


@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    # Store original values
    original_env = {}
    keys_to_clean = [
        'SCREENING_MODE', 'MONGODB_CONNECTION_STRING', 'OPENAI_API_KEY',
        'DEEPSEEK_API_KEY', 'PROJECT_NAME', 'LOG_LEVEL'
    ]
    
    for key in keys_to_clean:
        if key in os.environ:
            original_env[key] = os.environ[key]
            del os.environ[key]
    
    yield
    
    # Restore original values
    for key, value in original_env.items():
        os.environ[key] = value