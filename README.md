# GitHub Issues Contract Violation Analyzer

A tool for analyzing GitHub issues to identify and categorize API contract violations using state-of-the-art language models.

## Features

- **Advanced Contract Analysis**: Leverages LLMs to analyze GitHub issues for potential API contract violations
- **Multiple Storage Options**: 
  - JSON storage for detailed analysis results
  - CSV export functionality for data analysis
  - MongoDB integration for scalable data storage
- **Robust Data Processing**:
  - Support for both direct GitHub API fetching and CSV file input
  - Automatic checkpointing for long-running analyses
  - Intermediate results saving
  - Graceful shutdown handling
- **Modular Architecture**:
  - Pluggable storage backends
  - Extensible analyzer framework
  - Configurable LLM clients
- **Progress Tracking**: Real-time progress monitoring with customizable trackers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thromel/llm-contracts-research.git
cd llm-contracts-research
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
# API Keys
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_key

# OpenAI Settings
OPENAI_MODEL=your_model_name
OPENAI_BASE_URL=your_api_base_url
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000
OPENAI_TOP_P=1.0
OPENAI_FREQUENCY_PENALTY=0.0
OPENAI_PRESENCE_PENALTY=0.0

# MongoDB Settings (Optional)
MONGODB_URI=your_mongodb_uri
MONGODB_DB=your_database_name
MONGODB_ENABLED=true

# Analysis Settings
BATCH_SIZE=50
MAX_COMMENTS_PER_ISSUE=10
DEFAULT_LOOKBACK_DAYS=1000
SAVE_INTERMEDIATE=true
JSON_EXPORT=true
CSV_EXPORT=true
```

## Project Structure

```
src/
├── analysis/
│   ├── core/
│   │   ├── analyzers/
│   │   │   ├── contract_analyzer.py    # Core contract analysis logic
│   │   │   ├── github.py              # GitHub-specific analysis
│   │   │   └── orchestrator.py        # Analysis orchestration
│   │   ├── clients/
│   │   │   ├── github.py              # GitHub API client
│   │   │   └── openai.py             # OpenAI API client
│   │   ├── processors/
│   │   │   ├── cleaner.py            # Response cleaning
│   │   │   ├── validator.py          # Analysis validation
│   │   │   └── checkpoint.py         # Checkpoint management
│   │   ├── storage/
│   │   │   ├── json_storage.py       # JSON storage implementation
│   │   │   ├── csv_storage.py        # CSV storage implementation
│   │   │   └── mongodb/              # MongoDB integration
│   │   └── dto/                      # Data transfer objects
│   └── main.py                       # Main entry point
├── config/
│   └── settings.py                   # Configuration settings
└── utils/
    └── logger.py                     # Logging utilities
```

## Usage

### Basic Usage

1. Analyzing issues from a GitHub repository:
```bash
python -m src.analysis.main --repo owner/repo --issues 100
```

2. Analyzing issues from a CSV file:
```bash
python -m src.analysis.main --input-csv path/to/issues.csv
```

### Advanced Options

- `--resume`: Resume from the last checkpoint if available
- `--checkpoint-interval N`: Create checkpoints every N issues (default: 5)

### Storage Configuration

The analyzer supports multiple storage backends that can be configured in your `.env` file:

- **JSON Storage**: Enable with `JSON_EXPORT=true`
- **CSV Storage**: Enable with `CSV_EXPORT=true`
- **MongoDB Storage**: Enable with `MONGODB_ENABLED=true` and configure connection settings

### Examples

1. Analyze 50 issues with custom checkpoint interval:
```bash
python -m src.analysis.main --repo openai/openai-python --issues 50 --checkpoint-interval 10
```

2. Resume a previously interrupted analysis:
```bash
python -m src.analysis.main --repo openai/openai-python --issues 50 --resume
```

3. Analyze issues from a CSV file:
```bash
python -m src.analysis.main --input-csv data/raw/github_issues.csv
```

## Output Files

The analyzer generates several output files in the `data/analyzed` directory:

- **JSON Output**: 
  - `github_issues_analysis_TIMESTAMP_raw.json`: Raw analysis data
  - `github_issues_analysis_TIMESTAMP_final.json`: Final analysis results
  
- **CSV Output**:
  - `github_issues_analysis_TIMESTAMP_final.csv`: Tabular format of analysis results
  
- **Checkpoints**:
  - `analysis_checkpoint.json`: Temporary checkpoint file
  - `intermediate/`: Directory containing intermediate analysis results

## Architecture

### Core Components

1. **Analyzers**:
   - `ContractAnalyzer`: Core analysis logic for contract violations
   - `GitHubIssuesAnalyzer`: GitHub-specific implementation
   - `AnalysisOrchestrator`: Coordinates the analysis process

2. **Storage**:
   - Modular storage system with support for multiple backends
   - Factory pattern for storage creation
   - Adapter pattern for consistent interface

3. **Processors**:
   - Response cleaning and validation
   - Checkpoint management
   - Progress tracking

### Design Patterns

- **Factory Pattern**: Used for storage backend creation
- **Strategy Pattern**: Used for different analysis strategies
- **Adapter Pattern**: Used for storage implementations
- **Observer Pattern**: Used for progress tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write unit tests for new features
- Update documentation for significant changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## How to Contribute

We welcome contributions from the community! If you'd like to contribute improvements, fixes, or new features, please follow these guidelines:

1. Fork the repository and clone your fork.
2. Create a new branch for your changes (e.g., feature/your-feature or fix/issue-number).
3. Make your changes with clear, concise commit messages.
4. Ensure that your code adheres to the project's coding style (PEP 8).
5. Write tests for your changes where applicable.
6. Push your branch and open a pull request describing your changes.
7. Consult the issue tracker before making major changes to avoid duplicated efforts.

Thank you for your interest in contributing to GitHub Issues Contract Violation Analyzer!
