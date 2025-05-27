# Multi-Source Data Pipeline

A comprehensive pipeline for acquiring and processing LLM contract violation data from GitHub and Stack Overflow APIs.

## Features

- **Multi-source data acquisition**: Fetch data from both GitHub repositories and Stack Overflow tags
- **Configurable sources**: YAML configuration for repositories, tags, and limits
- **Step-by-step execution**: Run individual pipeline steps independently
- **Deduplication**: Avoid duplicate posts and LLM screening calls to save costs
- **Normalized data format**: Consistent data structure across different sources

## Quick Setup

1. **Copy environment file and add your API keys:**
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start MongoDB (if running locally):**
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo
   
   # Or using your local MongoDB installation
   mongod
   ```

## Usage

### Basic Usage

Run the full pipeline:
```bash
python run_pipeline.py
```

### Step-by-Step Execution

Run individual pipeline steps:

```bash
# 1. Data acquisition only
python run_pipeline.py --step acquisition

# 2. Keyword filtering only (processes unfiltered posts from DB)
python run_pipeline.py --step filtering

# 3. LLM screening only (processes filtered posts from DB)
python run_pipeline.py --step screening

# 4. LLM screening with limit
python run_pipeline.py --step screening --max-posts 50
```

### Configuration

Use a custom configuration file:
```bash
python run_pipeline.py --config my_config.yaml
```

### Statistics

View current pipeline statistics:
```bash
python run_pipeline.py --stats-only
```

## Configuration File

The pipeline uses `pipeline_config.yaml` for configuration. If it doesn't exist, a default one will be created.

Example configuration:

```yaml
sources:
  github:
    enabled: true
    repositories:
      - owner: openai
        repo: openai-python
      - owner: anthropics
        repo: anthropic-sdk-python
    max_issues_per_repo: 50
    days_back: 30
  stackoverflow:
    enabled: true
    tags:
      - openai-api
      - gpt-4
      - claude-api
    max_questions_per_tag: 100
    days_back: 30

deduplication:
  enabled: true
  similarity_threshold: 0.8

pipeline_steps:
  data_acquisition: true
  keyword_filtering: true
  llm_screening: true
```

## Environment Variables

Copy `env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
```

### Required Variables

```bash
MONGODB_URI=mongodb://localhost:27017/llm_contracts
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Optional Variables (for higher rate limits)

```bash
# GitHub API - Get from https://github.com/settings/tokens
GITHUB_TOKEN=ghp_your-github-token-here

# Stack Overflow API - Get from https://stackapps.com/apps/oauth/register
STACKOVERFLOW_API_KEY=your-stackoverflow-api-key-here

# Alternative LLM providers
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

### API Key Setup Instructions

1. **Stack Overflow API Key** (Optional but recommended):
   - Visit https://stackapps.com/apps/oauth/register
   - Register a new application with any name
   - Copy the generated key to `STACKOVERFLOW_API_KEY`
   - **Without key**: 300 requests per day
   - **With key**: 10,000 requests per day

2. **GitHub Token** (Optional but recommended):
   - Visit https://github.com/settings/tokens
   - Generate a new classic token with `public_repo` scope
   - **Without token**: 60 requests per hour
   - **With token**: 5,000 requests per hour

## Data Flow

1. **Data Acquisition**: 
   - Fetches issues from configured GitHub repositories
   - Fetches questions from configured Stack Overflow tags
   - Deduplicates based on content hash
   - Saves to `raw_posts` collection

2. **Keyword Filtering**:
   - Applies keyword filtering to identify relevant posts (containing LLM-related keywords)
   - Only posts that "pass" this filter proceed to expensive LLM screening
   - Saves results to `filtered_posts` collection
   - Marks raw posts as processed

3. **LLM Screening**:
   - Screens posts that passed keyword filtering
   - Avoids duplicate LLM calls using content hashes
   - Saves results to `llm_screening_results` collection

## Database Collections

- `raw_posts`: Original posts from GitHub/Stack Overflow
- `filtered_posts`: Posts after keyword filtering
- `llm_screening_results`: Results from LLM screening

## Pipeline Statistics Explained

When you run `--stats-only`, here's what each number means:

- `raw_posts`: Total posts collected from APIs
- `raw_posts_github`: Posts from GitHub specifically  
- `raw_posts_stackoverflow`: Posts from Stack Overflow specifically
- `filtered_posts`: Posts that went through keyword filtering
- `passed_filter`: **Posts that passed keyword filtering** (contain relevant LLM keywords)
- `screened`: Posts that have been processed by LLM screening
- `screening_results`: Total LLM screening results generated
- `duplicates_by_hash`: Posts skipped due to duplicate content

## Deduplication Strategy

The pipeline uses content-based deduplication to:

1. **Avoid storing duplicate posts**: Uses SHA256 hash of normalized title + body
2. **Avoid duplicate LLM calls**: Checks if content hash was already screened
3. **Save token costs**: Skips LLM screening for previously processed content

## Independent Step Execution

Each step can be run independently:

- **Data acquisition**: Always fetches new data from APIs
- **Keyword filtering**: Processes unfiltered posts from database
- **LLM screening**: Processes filtered posts that haven't been screened

This allows for:
- Resuming interrupted pipelines
- Testing individual components
- Cost control (run acquisition multiple times, then batch LLM screening)
- Debugging specific pipeline stages 