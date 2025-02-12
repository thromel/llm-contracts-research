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
(DATA_DIR / 'exports').mkdir(exist_ok=True)
CONTEXT_DIR.mkdir(exist_ok=True)

# API settings
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# OpenAI API settings
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
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
MAX_COMMENTS_PER_ISSUE = 10
DEFAULT_LOOKBACK_DAYS = int(os.getenv('DEFAULT_LOOKBACK_DAYS', '180'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = DATA_DIR / 'app.log'

# Export settings
CSV_EXPORT = bool(os.getenv('CSV_EXPORT', 'true').lower() == 'true')
JSON_EXPORT = bool(os.getenv('JSON_EXPORT', 'true').lower() == 'true')
SAVE_INTERMEDIATE = True
EXPORT_DIR = DATA_DIR / 'exports'

# Create export directories
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
(EXPORT_DIR / 'json').mkdir(parents=True, exist_ok=True)
(EXPORT_DIR / 'csv').mkdir(parents=True, exist_ok=True)
if SAVE_INTERMEDIATE:
    (EXPORT_DIR / 'intermediate').mkdir(parents=True, exist_ok=True)

# MongoDB settings
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DB = os.getenv('MONGODB_DB', 'llm_contracts_analysis')
MONGODB_ENABLED = bool(os.getenv('MONGODB_ENABLED', 'false').lower() == 'true')

# Analysis version and metadata
ANALYSIS_VERSION = "1.0.0"
ANALYSIS_MODEL = "violation-detection-v1"
ANALYSIS_BATCH_ID = "batch_001"

# Checkpoint settings
CHECKPOINT_INTERVAL = 5  # Number of issues to process before creating a checkpoint
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Delay in seconds between retries

# Pipeline stages
PIPELINE_STAGES = [
    'preprocessing',
    'model_construction',
    'training',
    'inference',
    'post_processing',
    'output_processing',
    'output_formatting',
    'serialization',
    'input_generation',
    'analysis',
    'response_handling'
]

# Create required directories
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'analysis').mkdir(exist_ok=True)
(DATA_DIR / 'checkpoints').mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
(EXPORT_DIR / 'csv').mkdir(exist_ok=True)
(EXPORT_DIR / 'json').mkdir(exist_ok=True)
