import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class ContractAnalyzer:
    def __init__(self, model="gpt-4-0125-preview"):
        """Initialize the contract analyzer with specified OpenAI model."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

        # Will be populated with analysis context from the paper
        self.system_prompt = """You are an expert in analyzing LLM API contracts and their violations. 
Your task is to analyze GitHub issues and determine if they represent contract violations.

A contract violation occurs when:
[TO BE FILLED WITH PAPER CONTEXT]

For each issue, analyze:
1. Is this a contract violation? (yes/no)
2. If yes:
   - Type of violation
   - Severity (high/medium/low)
   - Impact on users
   - Whether it was properly addressed
3. If no:
   - Categorize the issue type
   - Explain why it's not a violation

Provide your analysis in JSON format."""

    def analyze_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single issue for contract violations."""
        # Construct the issue context
        issue_context = f"""
Title: {issue['title']}
Body: {issue['body']}
Comments: {json.dumps(issue['first_comments'], indent=2)}
Created: {issue['created_at']}
Closed: {issue['closed_at']}
Labels: {issue['labels']}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": issue_context}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)
            return {
                'issue_number': issue['issue_number'],
                'repository': issue['repository'],
                'url': issue['url'],
                'analysis': analysis
            }

        except Exception as e:
            print(f"Error analyzing issue {issue['issue_number']}: {str(e)}")
            return None

    def analyze_batch(self, issues: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Analyze a batch of issues with progress tracking."""
        results = []

        with tqdm(total=len(issues), desc="Analyzing issues") as pbar:
            for i in range(0, len(issues), batch_size):
                batch = issues[i:i + batch_size]
                for issue in batch:
                    result = self.analyze_issue(issue)
                    if result:
                        results.append(result)
                    pbar.update(1)

                # Save intermediate results
                self.save_results(results, f"analysis_partial_{len(results)}")

        return results

    def save_results(self, results: List[Dict[str, Any]], filename_prefix: str):
        """Save analysis results to files."""
        # Create output directory
        os.makedirs('data/analyzed', exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"data/analyzed/{filename_prefix}_{timestamp}"

        # Save detailed results
        with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create summary
        summary = {
            'total_analyzed': len(results),
            'violations_found': sum(1 for r in results if r['analysis'].get('is_violation', False)),
            'by_repository': {},
            'by_violation_type': {},
            'by_severity': {'high': 0, 'medium': 0, 'low': 0}
        }

        # Update summary statistics
        for result in results:
            repo = result['repository']
            summary['by_repository'][repo] = summary['by_repository'].get(
                repo, 0) + 1

            if result['analysis'].get('is_violation'):
                vtype = result['analysis'].get('violation_type', 'unknown')
                severity = result['analysis'].get('severity', 'unknown')

                summary['by_violation_type'][vtype] = summary['by_violation_type'].get(
                    vtype, 0) + 1
                if severity in summary['by_severity']:
                    summary['by_severity'][severity] += 1

        # Save summary
        with open(f"{base_filename}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main function to run contract violation analysis."""
    # Load the most recent issues file
    data_dir = 'data'
    json_files = [f for f in os.listdir(data_dir) if f.endswith(
        '.json') and not f.endswith('_summary.json')]
    if not json_files:
        print("No issue data files found. Please run fetch_issues.py first.")
        return

    latest_file = max(json_files, key=lambda x: os.path.getctime(
        os.path.join(data_dir, x)))
    with open(os.path.join(data_dir, latest_file), 'r', encoding='utf-8') as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues from {latest_file}")

    # Initialize analyzer and run analysis
    analyzer = ContractAnalyzer()
    results = analyzer.analyze_batch(issues)

    # Save final results
    analyzer.save_results(results, "analysis_final")
    print("Analysis complete. Results saved in data/analyzed/")


if __name__ == '__main__':
    main()
