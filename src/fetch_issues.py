import os
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import List, Dict, Any
from utils import (
    get_github_client,
    format_issue_data,
    get_relevant_repositories,
    is_relevant_issue
)


def fetch_closed_issues(
    repo_name: str,
    client,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[Dict[str, Any]]:
    """
    Fetch closed issues from a GitHub repository within the specified date range.

    Args:
        repo_name: Name of the repository
        client: GitHub client
        start_date: Start date for fetching issues (default: 6 months ago)
        end_date: End date for fetching issues (default: current date)
    """
    try:
        repo = client.get_repo(repo_name)
        issues = []

        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            # Last 6 months by default
            start_date = end_date - timedelta(days=180)

        # Create query string for GitHub API
        query = f"repo:{repo_name} is:issue is:closed closed:{
            start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"

        # Fetch issues using search API
        for issue in tqdm(repo.get_issues(state='closed'), desc=f'Fetching issues from {repo_name}'):
            if issue.closed_at and start_date <= issue.closed_at <= end_date:
                if is_relevant_issue(issue):
                    issues.append(format_issue_data(issue))

        return issues
    except Exception as e:
        print(f"Error fetching issues from {repo_name}: {str(e)}")
        return []


def save_to_files(issues: List[Dict[str, Any]], base_filename: str):
    """Save the collected data in both CSV and JSON formats."""
    if not issues:
        print("No issues were collected.")
        return

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(issues)
    csv_file = f'data/{base_filename}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved {len(issues)} issues to {csv_file}")

    # Save as JSON with more detailed information
    json_file = f'data/{base_filename}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed data to {json_file}")

    # Generate summary statistics
    summary = {
        'total_issues': len(issues),
        'repositories': len(set(issue['repository'] for issue in issues)),
        'date_range': {
            'earliest': min(issue['created_at'] for issue in issues),
            'latest': max(issue['closed_at'] for issue in issues if issue['closed_at'])
        },
        'issues_by_repo': dict(df['repository'].value_counts()),
        'avg_resolution_time': df['resolution_time_hours'].mean(),
        'total_comments': df['comments_count'].sum()
    }

    # Save summary
    summary_file = f'data/{base_filename}_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_file}")


def main():
    """Main function to fetch issues and save to files."""
    # Initialize GitHub client
    client = get_github_client()

    # Get list of repositories to analyze
    repositories = get_relevant_repositories()

    # Set date range (last 6 months by default)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    # Fetch issues from all repositories
    all_issues = []
    for repo_name in repositories:
        issues = fetch_closed_issues(repo_name, client, start_date, end_date)
        all_issues.extend(issues)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'github_issues_{timestamp}'

    # Save the collected data
    save_to_files(all_issues, base_filename)


if __name__ == '__main__':
    main()
