"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
import tempfile
import os
from unittest.mock import Mock, AsyncMock

from pipeline.foundation.config import ConfigManager
from pipeline.storage.base import StorageAdapter
from pipeline.core.events import EventBus
from pipeline.common.models import RawPost, FilteredPost, Platform


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> ConfigManager:
    """Create a test configuration manager."""
    config_manager = ConfigManager()
    
    # Set test configuration values
    config_manager.set("screening.mode", "traditional")
    config_manager.set("mongodb.uri", "mongodb://localhost:27017/test")
    config_manager.set("mongodb.database", "llm_contracts_test")
    config_manager.set("filtering.confidence_threshold", 0.3)
    config_manager.set("filtering.batch_size", 100)
    
    return config_manager


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage adapter."""
    storage = AsyncMock(spec=StorageAdapter)
    
    # Configure default behaviors
    storage.connect = AsyncMock()
    storage.disconnect = AsyncMock()
    storage.save = AsyncMock(return_value="test_id_123")
    storage.find = AsyncMock(return_value=AsyncGenerator())
    storage.count = AsyncMock(return_value=0)
    storage.exists = AsyncMock(return_value=False)
    
    return storage


@pytest_asyncio.fixture
async def event_bus():
    """Create and start an event bus for testing."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def sample_raw_post() -> RawPost:
    """Create a sample raw post for testing."""
    return RawPost(
        platform=Platform.GITHUB,
        source_id="123",
        url="https://github.com/test/repo/issues/123",
        title="Error with OpenAI API rate limits",
        body_md="I'm getting rate limit errors when calling the API...",
        author="testuser",
        tags=["bug", "api"],
        labels=["help wanted"],
        state="closed",
        comments_count=5
    )


@pytest.fixture
def sample_filtered_post(sample_raw_post: RawPost) -> FilteredPost:
    """Create a sample filtered post for testing."""
    return FilteredPost(
        raw_post_id=sample_raw_post._id,
        passed_keyword_filter=True,
        matched_keywords=["rate limit", "API", "error"],
        filter_confidence=0.85,
        relevant_snippets=[
            "getting rate limit errors when calling the API"
        ],
        potential_contracts=["rate_limit"]
    )


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = Mock()
    client.get_repo = Mock()
    client.get_issues = Mock(return_value=[])
    return client


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    
    # Mock completion response
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content='{"decision": "Y", "confidence": 0.9}'))
    ]
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def env_vars(monkeypatch):
    """Set test environment variables."""
    test_vars = {
        "MONGODB_URI": "mongodb://localhost:27017/test",
        "GITHUB_TOKEN": "test_github_token",
        "OPENAI_API_KEY": "test_openai_key",
        "SCREENING_MODE": "traditional"
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars


@pytest.fixture
async def mock_mongodb_client():
    """Create a mock MongoDB client."""
    from motor.motor_asyncio import AsyncIOMotorClient
    
    client = AsyncMock(spec=AsyncIOMotorClient)
    db = AsyncMock()
    collection = AsyncMock()
    
    # Wire up the mock hierarchy
    client.__getitem__ = Mock(return_value=db)
    db.__getitem__ = Mock(return_value=collection)
    
    # Mock collection methods
    collection.insert_one = AsyncMock(
        return_value=Mock(inserted_id="test_id")
    )
    collection.find = Mock(
        return_value=AsyncMock(
            __aiter__=AsyncMock(return_value=iter([]))
        )
    )
    collection.count_documents = AsyncMock(return_value=0)
    collection.create_index = AsyncMock()
    
    return client


# Test data factories

def create_test_posts(count: int = 5) -> list[RawPost]:
    """Create test raw posts."""
    posts = []
    for i in range(count):
        post = RawPost(
            platform=Platform.GITHUB if i % 2 == 0 else Platform.STACKOVERFLOW,
            source_id=f"test_{i}",
            url=f"https://example.com/post/{i}",
            title=f"Test Post {i}",
            body_md=f"This is test post {i} with some content",
            author=f"user_{i}",
            score=i * 10
        )
        posts.append(post)
    return posts


# Async test helpers

async def async_list(async_gen):
    """Convert async generator to list."""
    return [item async for item in async_gen]


# Markers for different test types

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_mongodb: mark test as requiring MongoDB"
    )
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )