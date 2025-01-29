import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from github import Github
from github.Issue import Issue


def load_github_token() -> str:
    """Load GitHub token from environment variables."""
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError(
            "GitHub token not found. Please set GITHUB_TOKEN in .env file")
    return token


def get_github_client() -> Github:
    """Create and return an authenticated GitHub client."""
    token = load_github_token()
    return Github(token)


def is_relevant_issue(issue: Issue) -> bool:
    """
    Determine if an issue is relevant for analysis based on specific criteria.
    """
    # Skip pull requests
    if issue.pull_request:
        return False

    # Keywords related to LLM contracts and API usage
    relevant_keywords = {
        'api', 'token', 'rate limit', 'quota', 'billing', 'cost',
        'pricing', 'usage', 'limit', 'throttle', 'error', 'exception',
        'authentication', 'key', 'credit', 'subscription', 'plan',
        'payment', 'charge', 'free tier', 'enterprise',
        'performance', 'latency', 'timeout', 'concurrent',
        'model', 'prompt', 'completion', 'embedding'
    }

    # Check if any relevant keywords are in the title or body
    title = issue.title.lower()
    body = issue.body.lower() if issue.body else ''
    text_to_check = (
        f"{title} {body}"
    )
    return any(keyword in text_to_check for keyword in relevant_keywords)


def get_issue_comments(issue: Issue, max_comments: int = 5) -> List[Dict[str, Any]]:
    """Get the first few comments of an issue."""
    comments = []
    try:
        for idx, comment in enumerate(issue.get_comments()):
            if idx >= max_comments:
                break
            comments.append({
                'author': comment.user.login if comment.user else 'unknown',
                'created_at': comment.created_at.isoformat(),
                'body': comment.body
            })
    except Exception as e:
        print(f"Error fetching comments for issue {issue.number}: {str(e)}")
    return comments


def format_issue_data(issue: Issue) -> Dict[str, Any]:
    """Format a GitHub issue into a dictionary for CSV export with enhanced fields."""
    comments = get_issue_comments(issue)

    return {
        # Basic Information
        'repository': issue.repository.full_name,
        'issue_number': issue.number,
        'title': issue.title,
        'body': issue.body,

        # Status Information
        'state': issue.state,
        'created_at': issue.created_at.isoformat(),
        'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
        'updated_at': issue.updated_at.isoformat(),

        # Author Information
        'author': issue.user.login if issue.user else 'unknown',
        'author_type': issue.user.type if issue.user else 'unknown',

        # Engagement Metrics
        'comments_count': issue.comments,
        'reactions_count': sum(reaction.total_count for reaction in issue.get_reactions()),

        # Labels and Categorization
        'labels': ','.join([label.name for label in issue.labels]),
        'milestone': issue.milestone.title if issue.milestone else None,

        # Resolution Information
        'closed_by': issue.closed_by.login if issue.closed_by else None,
        'resolution_time_hours': (issue.closed_at - issue.created_at).total_seconds() / 3600 if issue.closed_at else None,

        # First Few Comments
        'first_comments': comments,

        # Reference
        'url': issue.html_url
    }


def get_relevant_repositories() -> List[str]:
    """Return a list of relevant repositories to analyze, organized by category."""
    return [
        # Major Commercial LLM Providers
        'openai/openai-python',
        'anthropic/anthropic-python',
        'cohere-ai/cohere-python',

        # Open Source LLM Organizations
        'mistralai/mistralai-python',
        'deepseek-ai/deepseek-coder',
        'nvidia/NeMo',

        # LLM Development Tools
        'huggingface/transformers',
        'langchain-ai/langchain-python',

        # Vector Databases & Embeddings
        'chroma-core/chroma',
        'pinecone-io/pinecone-python-client',
        'weaviate/weaviate-python-client',

        # Chinese LLM Companies
        'QwenLM/Qwen-7B',
        'THUDM/ChatGLM-6B',

        # LLM Safety and Evaluation
        'EleutherAI/lm-evaluation-harness',

        # Model Training & Deployment
        'microsoft/DeepSpeed',
        'microsoft/semantic-kernel',
        'pytorch/serve'
    ]
