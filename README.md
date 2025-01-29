# LLM Contracts Research

This project analyzes LLM contracts by collecting and processing GitHub issues from relevant repositories. The analysis focuses on understanding patterns, challenges, and solutions in LLM-related issues, with a particular emphasis on identifying contract violations.

## Project Overview

### Data Collection
- Fetches closed issues from major LLM-related repositories
- Collects comprehensive issue data including:
  - Basic issue information (title, body, state)
  - Author information
  - Temporal data (creation, closure, resolution time)
  - Engagement metrics (comments, reactions)
  - Labels and categorization
  - First few comments for context

### Repositories Analyzed
- **Commercial LLM Providers**: OpenAI, Cohere
- **Open Source LLM Organizations**: Mistral AI, DeepSeek, NVIDIA
- **LLM Development Tools**: Hugging Face, LangChain
- **Vector Databases**: Chroma, Pinecone, Weaviate
- **Chinese LLM Companies**: Qwen, ChatGLM2
- **LLM Safety & Evaluation**: EleutherAI
- **Model Training & Deployment**: Microsoft DeepSpeed, Semantic Kernel, PyTorch Serve

### Features
1. **Robust Data Collection**:
   - Batch processing with periodic saves
   - Rate limit handling
   - Error recovery
   - Progress tracking

2. **Data Export**:
   - CSV format for tabular analysis
   - Detailed JSON with full context
   - Summary statistics

3. **Automated Analysis** (Planned):
   - LLM-based contract violation detection
   - Classification of issue types
   - Pattern identification
   - Trend analysis

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

4. Create a `.env` file with your tokens:
```
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_key_here  # For contract analysis
```

## Usage

1. Fetch GitHub Issues:
```bash
python src/fetch_issues.py
```

2. Analyze Contract Violations (Coming Soon):
```bash
python src/analyze_contracts.py
```

## Project Structure

```
llm-contracts-research/
├── data/               # Directory for storing collected data
│   ├── raw/           # Raw GitHub issues data
│   └── analyzed/      # Results of LLM analysis
├── src/               # Source code
│   ├── fetch_issues.py    # GitHub issues collection
│   ├── analyze_contracts.py    # Contract violation analysis (planned)
│   └── utils.py           # Utility functions
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Data Format

### Issue Collection
Each collected issue includes:
- Repository information
- Issue details (title, body, state)
- Temporal data
- Author information
- Engagement metrics
- Labels and categorization
- Comments (first 5 for context)

### Analysis Output (Planned)
The contract violation analysis will add:
- Contract violation classification
- Violation type categorization
- Confidence scores
- Supporting evidence
- Recommended resolutions

## Contributing

Feel free to open issues or submit pull requests with improvements.
