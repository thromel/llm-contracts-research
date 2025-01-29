import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
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
    end_date: datetime = None,
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch closed issues from a GitHub repository within the specified date range.

    Args:
        repo_name: Name of the repository
        client: GitHub client
        start_date: Start date for fetching issues (default: 6 months ago)
        end_date: End date for fetching issues (default: current date)
        batch_size: Number of issues to process before saving (default: 100)
    """
    try:
        repo = client.get_repo(repo_name)
        issues = []
        processed = 0

        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            # Last 6 months by default
            start_date = end_date - timedelta(days=180)

        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Get closed issues
        closed_issues = repo.get_issues(state='closed')

        with tqdm(desc=f'Fetching issues from {repo_name}') as pbar:
            for issue in closed_issues:
                try:
                    if issue.closed_at and start_date <= issue.closed_at <= end_date:
                        if is_relevant_issue(issue):
                            issue_data = format_issue_data(issue)
                            if issue_data:  # Only add if successfully formatted
                                issues.append(issue_data)
                                processed += 1
                                pbar.update(1)

                                # Save intermediate results for large repositories
                                if processed % batch_size == 0:
                                    yield issues[-batch_size:]
                except Exception as e:
                    print(f"Error processing issue #{issue.number}: {str(e)}")
                    continue

        # Yield any remaining issues
        remaining = processed % batch_size
        if remaining > 0 and len(issues) >= remaining:
            yield issues[-remaining:]

    except Exception as e:
        print(f"Error fetching issues from {repo_name}: {str(e)}")
        return []


def save_to_files(issues: List[Dict[str, Any]], base_filename: str, mode: str = 'w'):
    """
    Save the collected data in both CSV and JSON formats.

    Args:
        issues: List of issue data to save
        base_filename: Base name for output files
        mode: File open mode ('w' for write, 'a' for append)
    """
    if not issues:
        print("No issues to save.")
        return

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(issues)
    csv_file = f'data/{base_filename}.csv'
    if mode == 'a' and os.path.exists(csv_file):
        # Append without header if file exists
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    print(f"Saved {len(issues)} issues to {csv_file}")

    # Save as JSON
    json_file = f'data/{base_filename}.json'
    if mode == 'a' and os.path.exists(json_file):
        # Load existing data and append
        with open(json_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        issues = existing_data + issues

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed data to {json_file}")

    # Generate and save summary
    summary = {
        'total_issues': len(issues),
        'repositories': len(set(issue['repository'] for issue in issues)),
        'date_range': {
            'earliest': min(issue['created_at'] for issue in issues),
            'latest': max(issue['closed_at'] for issue in issues if issue['closed_at'])
        },
        'issues_by_repo': dict(df['repository'].value_counts()),
        'avg_resolution_time': float(df['resolution_time_hours'].mean()),
        'total_comments': int(df['comments_count'].sum())
    }

    summary_file = f'data/{base_filename}_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_file}")


def main():
    """Main function to fetch issues and save to files."""
    client = get_github_client()
    repositories = get_relevant_repositories()

    # Set date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=180)

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'github_issues_{timestamp}'

    total_issues = []
    total_repos = len(repositories)

    for idx, repo_name in enumerate(repositories, 1):
        print(f"\nProcessing repository {idx}/{total_repos}: {repo_name}")
        try:
            for batch in fetch_closed_issues(repo_name, client, start_date, end_date):
                if batch:
                    partial_filename = f"{base_filename}_partial_{idx}"
                    save_to_files(batch, partial_filename, mode='a')
                    total_issues.extend(batch)
        except Exception as e:
            print(f"Error processing repository {repo_name}: {str(e)}")
            if total_issues:
                save_to_files(total_issues, f"{base_filename}_error_{idx}")

    if total_issues:
        save_to_files(total_issues, base_filename)
    else:
        print("No issues were collected from any repository.")


if __name__ == '__main__':
    main()
