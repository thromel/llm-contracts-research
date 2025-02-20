from dataclasses import dataclass, field
from typing import Optional, Literal, List
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MongoConfig:
    """MongoDB configuration."""
    host: str = os.getenv('MONGO_HOST', 'localhost')
    port: int = int(os.getenv('MONGO_PORT', '27017'))
    user: str = os.getenv('MONGO_USER', '')
    password: str = os.getenv('MONGO_PASSWORD', '')
    database: str = os.getenv('MONGO_DB', 'github_issues')


@dataclass
class PostgresConfig:
    """PostgreSQL configuration."""
    host: str = os.getenv('POSTGRES_HOST', 'localhost')
    port: int = int(os.getenv('POSTGRES_PORT', '5432'))
    user: str = os.getenv('POSTGRES_USER', 'postgres')
    password: str = os.getenv('POSTGRES_PASSWORD', '')
    database: str = os.getenv('POSTGRES_DB', 'github_issues')


@dataclass
class GitHubConfig:
    """GitHub API configuration."""
    token: str = os.getenv('GITHUB_TOKEN', '')
    api_url: str = 'https://api.github.com'
    per_page: int = 100
    max_retries: int = 3
    retry_delay: int = 5  # seconds


@dataclass
class AppConfig:
    """Application configuration."""
    # Database settings
    db_provider: Literal['mongo', 'postgres'] = os.getenv(
        'DB_PROVIDER', 'mongo')
    mongo: MongoConfig = field(default_factory=MongoConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    # GitHub settings
    github: GitHubConfig = field(default_factory=GitHubConfig)

    # Logging and progress
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    show_progress: bool = os.getenv('SHOW_PROGRESS', 'true').lower() == 'true'

    # Checkpointing
    checkpoint_enabled: bool = os.getenv(
        'CHECKPOINT_ENABLED', 'true').lower() == 'true'
    checkpoint_interval: int = int(
        os.getenv('CHECKPOINT_INTERVAL', '100'))  # issues
    checkpoint_dir: str = os.getenv('CHECKPOINT_DIR', 'checkpoints')

    # Concurrency settings
    max_concurrent_requests: int = int(
        os.getenv('MAX_CONCURRENT_REQUESTS', '5'))

    # Issue fetching settings
    batch_size: int = int(os.getenv('BATCH_SIZE', '50'))
    max_issues_per_repo: Optional[int] = int(
        os.getenv('MAX_ISSUES_PER_REPO', '0')) or None
    since_days: Optional[int] = int(os.getenv('SINCE_DAYS', '0')) or None
    include_closed: bool = os.getenv(
        'INCLUDE_CLOSED', 'true').lower() == 'true'

    # Export settings
    export_format: List[str] = field(
        default_factory=lambda: os.getenv('EXPORT_FORMAT', 'json').split(','))
    export_dir: str = os.getenv('EXPORT_DIR', 'exports')


# Global config instance
config = AppConfig()
