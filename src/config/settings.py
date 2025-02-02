"""Configuration settings for the application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
CONTEXT_DIR = BASE_DIR / 'context'

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'raw').mkdir(exist_ok=True)
(DATA_DIR / 'analyzed').mkdir(exist_ok=True)
CONTEXT_DIR.mkdir(exist_ok=True)

# API settings
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# OpenAI API settings
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.3'))
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
OPENAI_TOP_P = float(os.getenv('OPENAI_TOP_P', '1.0'))
OPENAI_FREQUENCY_PENALTY = float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0'))
OPENAI_PRESENCE_PENALTY = float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0'))

# GitHub API settings
GITHUB_API_VERSION = '2022-11-28'
GITHUB_PER_PAGE = 100
GITHUB_MAX_RETRIES = 3
GITHUB_RETRY_DELAY = 5  # seconds

# Analysis settings
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
MAX_COMMENTS_PER_ISSUE = int(os.getenv('MAX_COMMENTS_PER_ISSUE', '5'))
DEFAULT_LOOKBACK_DAYS = int(os.getenv('DEFAULT_LOOKBACK_DAYS', '180'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = DATA_DIR / 'app.log'

# Export settings
CSV_EXPORT = bool(os.getenv('CSV_EXPORT', 'true').lower() == 'true')
JSON_EXPORT = bool(os.getenv('JSON_EXPORT', 'true').lower() == 'true')
SAVE_INTERMEDIATE = bool(
    os.getenv('SAVE_INTERMEDIATE', 'true').lower() == 'true')
