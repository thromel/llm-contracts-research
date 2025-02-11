# GitHub Issues Contract Violation Analyzer

A tool for analyzing GitHub issues to identify and categorize API contract violations.

## Features

- Analyze GitHub issues for potential API contract violations
- Support for both direct GitHub API fetching and CSV file input
- Automatic checkpointing for long-running analyses
- Graceful shutdown handling
- Detailed analysis results in both CSV and JSON formats

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
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=your_model_name
OPENAI_BASE_URL=your_api_base_url

# Analysis Settings
BATCH_SIZE=50
MAX_COMMENTS_PER_ISSUE=10
DEFAULT_LOOKBACK_DAYS=1000
```

## Usage

### Analyzing Issues from GitHub

To analyze issues directly from a GitHub repository:

```bash
python -m src.analysis.main --repo owner/repo --issues 100
```

### Analyzing Issues from CSV

To analyze issues from a previously saved CSV file:

```bash
python -m src.analysis.main --input-csv path/to/issues.csv
```

### Additional Options

- `--resume`: Resume from the last checkpoint if available
- `--checkpoint-interval N`: Create checkpoints every N issues (default: 5)

### Examples

1. Analyze 50 issues from the OpenAI Python client repository:
```bash
python -m src.analysis.main --repo openai/openai-python --issues 50
```

2. Resume a previously interrupted analysis:
```bash
python -m src.analysis.main --repo openai/openai-python --issues 50 --resume
```

3. Analyze issues from a CSV file:
```bash
python -m src.analysis.main --input-csv data/raw/github_issues.csv
```

## Project Structure

```
src/
├── analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py      # Core analysis functionality
│   │   ├── checkpoint.py    # Checkpoint management
│   │   └── data_loader.py   # Data loading utilities
│   └── main.py             # Main entry point
├── config/
│   └── settings.py         # Configuration settings
└── utils/
    └── logger.py          # Logging utilities
```

## Output

The analyzer generates several output files in the `data/analyzed` directory:

- `github_issues_analysis_TIMESTAMP_raw.csv`: Raw analysis data
- `github_issues_analysis_TIMESTAMP_final.csv`: Final analysis results
- `analysis_checkpoint.json`: Checkpoint file (temporary)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
