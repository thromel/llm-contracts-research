"""Script to analyze violation types from GitHub issues analysis results."""

import pandas as pd
import argparse
import os
from pathlib import Path
from typing import Dict, Any
import json
from src.core.config import load_config  # Updated import
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def analyze_violations(csv_path: str) -> Dict[str, Any]:
    """Analyze violation types from a CSV file of analyzed GitHub issues."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Log available columns for debugging
        logger.info("Available columns in CSV: {}".format(list(df.columns)))

        # Get total number of issues
        total_issues = len(df)

        # Get number of violations
        violations = df[df['has_violation'] == True]
        total_violations = len(violations)

        # Get counts and URLs for each violation type
        violation_types = violations['violation_type'].value_counts().to_dict()
        violation_urls = {}
        for vtype in violation_types.keys():
            type_violations = violations[violations['violation_type'] == vtype]
            violation_urls[vtype] = {
                'count': len(type_violations),
                'issues': [
                    {
                        'url': url,
                        'title': title,
                        'severity': severity,
                        'confidence': confidence,
                        'resolution_status': status,
                        'resolution_details': details
                    }
                    for url, title, severity, confidence, status, details in
                    zip(type_violations['url'],
                        type_violations['title'],
                        type_violations['severity'],
                        type_violations['confidence'],
                        type_violations['resolution_status'],
                        type_violations['resolution_details'])
                ]
            }

        # Get counts for each severity level
        severity_counts = violations['severity'].value_counts().to_dict()

        # Get average confidence
        confidence_map = {'high': 3, 'medium': 2, 'low': 1}
        df['confidence_score'] = df['confidence'].map(confidence_map)
        avg_confidence = df['confidence_score'].mean()

        # Calculate percentages
        violation_percentages = {
            k: (v/total_issues)*100 for k, v in violation_types.items()}

        # Get resolution time statistics for violations vs non-violations
        resolution_stats = {
            'violation_issues_avg_hours': violations['resolution_time'].mean(),
            'non_violation_issues_avg_hours': df[df['has_violation'] == False]['resolution_time'].mean(),
            'overall_avg_hours': df['resolution_time'].mean()
        }

        return {
            'total_issues': total_issues,
            'total_violations': total_violations,
            'violation_types': violation_types,
            'violation_details': violation_urls,
            'violation_percentages': violation_percentages,
            'severity_distribution': severity_counts,
            'average_confidence_score': avg_confidence,
            'resolution_time_stats': resolution_stats
        }

    except Exception as e:
        logger.error("Error analyzing violations: {}".format(str(e)))
        raise


def save_analysis(analysis: Dict[str, Any], output_dir: Path, input_filename: str) -> None:
    """Save the analysis results to a JSON file."""
    try:
        # Create analysis filename based on input filename
        base_name = Path(input_filename).stem
        output_path = output_dir / \
            "{}_violation_analysis.json".format(base_name)

        # Save as formatted JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)

        logger.info("Analysis saved to {}".format(output_path))

        # Print summary to console
        print("\nAnalysis Summary:")
        print("-----------------")
        print("Total Issues Analyzed: {}".format(analysis['total_issues']))
        print("Total Violations Found: {}".format(
            analysis['total_violations']))

        print("\nViolation Types:")
        for vtype, count in analysis['violation_types'].items():
            print("\n  {}: {} ({:.2f}%)".format(
                vtype, count, analysis['violation_percentages'][vtype]))
            print("  Issues:")
            for issue in analysis['violation_details'][vtype]['issues']:
                print("    - {} (Severity: {}, Confidence: {})".format(
                    issue['url'], issue['severity'], issue['confidence']))
                if issue['resolution_status']:
                    print("      Resolution Status: {}".format(
                        issue['resolution_status']))
                if issue['resolution_details']:
                    print("      Resolution Details: {}".format(
                        issue['resolution_details']))

        print("\nSeverity Distribution:")
        for severity, count in analysis['severity_distribution'].items():
            print("  {}: {}".format(severity, count))

        print("\nResolution Time Statistics (hours):")
        for key, value in analysis['resolution_time_stats'].items():
            print("  {}: {:.2f}".format(key, value))

    except Exception as e:
        logger.error("Error saving analysis: {}".format(str(e)))
        raise


def main():
    """Main function to run the violation analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze violation types from GitHub issues analysis results.')
    parser.add_argument('--file', type=str,
                        help='Path to the CSV file to analyze')
    parser.add_argument('--latest', action='store_true',
                        help='Analyze the most recent analysis file in the data directory')
    parser.add_argument('--config', type=str,
                        help='Path to YAML config file (optional)')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Determine export dir from environment or config
        data_dir = os.getenv('DATA_DIR', 'data')

        # Determine which file to analyze
        if args.file:
            csv_path = args.file
        elif args.latest:
            # Find the most recent final analysis file
            data_dir = Path(data_dir) / 'analyzed'
            analysis_files = list(data_dir.glob(
                'github_issues_analysis_*_final_*.csv'))
            if not analysis_files:
                logger.error("No analysis files found in {}".format(data_dir))
                return
            csv_path = str(
                max(analysis_files, key=lambda x: x.stat().st_mtime))
        else:
            logger.error("Either --file or --latest must be specified")
            return

        # Run the analysis
        logger.info("Analyzing file: {}".format(csv_path))
        analysis_results = analyze_violations(csv_path)

        # Save and display results
        output_dir = Path(data_dir) / 'analysis'
        output_dir.mkdir(exist_ok=True)
        save_analysis(analysis_results, output_dir, csv_path)

    except Exception as e:
        logger.error("Analysis failed: {}".format(str(e)))


if __name__ == '__main__':
    main()
