# LLM Contracts Research

This project analyzes LLM contracts by collecting and processing GitHub issues from relevant repositories. The analysis focuses on understanding patterns, challenges, and solutions in LLM-related issues.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-contracts-research.git
cd llm-contracts-research
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your GitHub token:
```
GITHUB_TOKEN=your_token_here
```

## Usage

1. Run the data collection script:
```bash
python src/fetch_issues.py
```

This will generate a CSV file containing the collected GitHub issues in the `data` directory.

## Project Structure

```
llm-contracts-research/
├── data/               # Directory for storing collected data
├── src/               # Source code
│   ├── fetch_issues.py    # Script to fetch GitHub issues
│   └── utils.py           # Utility functions
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Contributing

Feel free to open issues or submit pull requests with improvements. 