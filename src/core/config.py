"""Configuration module."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Literal, List
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
    """GitHub configuration."""
    token: str = os.getenv('GITHUB_TOKEN', '')
    api_url: str = 'https://api.github.com'
    per_page: int = 100
    max_retries: int = 3


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

    # Repository list
    repositories: List[str] = field(default_factory=list)

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
    include_comments: bool = os.getenv(
        'INCLUDE_COMMENTS', 'true').lower() == 'true'
    max_comments_per_issue: Optional[int] = int(
        os.getenv('MAX_COMMENTS_PER_ISSUE', '0')) or None

    # Export settings
    export_format: List[str] = field(
        default_factory=lambda: os.getenv('EXPORT_FORMAT', 'json').split(','))
    export_dir: str = os.getenv('EXPORT_DIR', 'exports')


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from file and environment.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        AppConfig instance
    """
    config = AppConfig()

    if config_path:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

            # Load repositories list
            if isinstance(yaml_config, dict):
                if 'repositories' in yaml_config:
                    config.repositories = yaml_config['repositories']

                # Override other settings if specified
                if 'github' in yaml_config:
                    github_config = yaml_config['github']
                    if 'token' in github_config:
                        config.github.token = github_config['token']
                    if 'api_url' in github_config:
                        config.github.api_url = github_config['api_url']
                    if 'per_page' in github_config:
                        config.github.per_page = github_config['per_page']
                    if 'max_retries' in github_config:
                        config.github.max_retries = github_config['max_retries']

                if 'mongodb' in yaml_config:
                    mongo_config = yaml_config['mongodb']
                    if 'host' in mongo_config:
                        config.mongo.host = mongo_config['host']
                    if 'port' in mongo_config:
                        config.mongo.port = mongo_config['port']
                    if 'user' in mongo_config:
                        config.mongo.user = mongo_config['user']
                    if 'password' in mongo_config:
                        config.mongo.password = mongo_config['password']
                    if 'database' in mongo_config:
                        config.mongo.database = mongo_config['database']

                if 'settings' in yaml_config:
                    settings = yaml_config['settings']
                    if 'max_issues_per_repo' in settings:
                        config.max_issues_per_repo = settings['max_issues_per_repo']
                    if 'since_days' in settings:
                        config.since_days = settings['since_days']
                    if 'include_closed' in settings:
                        config.include_closed = settings['include_closed']
                    if 'include_comments' in settings:
                        config.include_comments = settings['include_comments']
                    if 'max_comments_per_issue' in settings:
                        config.max_comments_per_issue = settings['max_comments_per_issue']
                    if 'checkpoint_enabled' in settings:
                        config.checkpoint_enabled = settings['checkpoint_enabled']
                    if 'checkpoint_interval' in settings:
                        config.checkpoint_interval = settings['checkpoint_interval']
                    if 'checkpoint_dir' in settings:
                        config.checkpoint_dir = settings['checkpoint_dir']
                    if 'max_concurrent_requests' in settings:
                        config.max_concurrent_requests = settings['max_concurrent_requests']

    return config


# Global config instance
config = AppConfig()
