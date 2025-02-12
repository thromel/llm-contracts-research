"""CSV export functionality for analysis results."""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.analysis.core.dto import (
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO
)

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports analysis results to CSV format."""

    def __init__(self, output_dir: Path):
        """Initialize CSV exporter.

        Args:
            output_dir: Directory to store CSV files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_results(self, results: AnalysisResultsDTO) -> None:
        """Export analysis results to CSV files.

        Args:
            results: Analysis results to export
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export main analysis results
            self._export_analysis_results(results.analyzed_issues, timestamp)

            # Export contract suggestions
            self._export_contract_suggestions(
                results.analyzed_issues, timestamp)

            # Export metadata
            self._export_metadata(results.metadata, timestamp)

            logger.info("Exported analysis results to CSV")

        except Exception as e:
            logger.error("Failed to export results to CSV: {}".format(str(e)))
            raise

    def _export_analysis_results(self, analyses: List[ContractAnalysisDTO], timestamp: str) -> None:
        """Export main analysis results to CSV.

        Args:
            analyses: List of analysis results
            timestamp: Export timestamp
        """
        output_file = self.output_dir / f"contract_violations_{timestamp}.csv"

        fieldnames = [
            'issue_number',
            'issue_title',
            'issue_url',
            'repository_name',
            'repository_owner',
            'has_violation',
            'violation_type',
            'severity',
            'confidence',
            'root_cause',
            'effects',
            'resolution_status',
            'resolution_details',
            'contract_category',
            'supporting_evidence',
            'frequency',
            'workarounds',
            'impact',
            'affected_stages',
            'propagation_path',
            'analysis_timestamp'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for analysis in analyses:
                row = {
                    'issue_number': analysis.issue_number,
                    'issue_title': analysis.issue_title,
                    'issue_url': analysis.issue_url,
                    'repository_name': analysis.repository_name,
                    'repository_owner': analysis.repository_owner,
                    'has_violation': analysis.has_violation,
                    'violation_type': analysis.violation_type,
                    'severity': analysis.severity,
                    'confidence': analysis.confidence,
                    'root_cause': analysis.root_cause,
                    'effects': '; '.join(analysis.effects),
                    'resolution_status': analysis.resolution_status,
                    'resolution_details': analysis.resolution_details,
                    'contract_category': analysis.contract_category,
                    'supporting_evidence': '; '.join(analysis.comment_analysis.supporting_evidence),
                    'frequency': analysis.comment_analysis.frequency,
                    'workarounds': '; '.join(analysis.comment_analysis.workarounds),
                    'impact': analysis.comment_analysis.impact,
                    'affected_stages': '; '.join(analysis.error_propagation.affected_stages),
                    'propagation_path': analysis.error_propagation.propagation_path,
                    'analysis_timestamp': analysis.analysis_timestamp
                }
                writer.writerow(row)

    def _export_contract_suggestions(self, analyses: List[ContractAnalysisDTO], timestamp: str) -> None:
        """Export suggested new contracts to CSV.

        Args:
            analyses: List of analysis results
            timestamp: Export timestamp
        """
        output_file = self.output_dir / f"suggested_contracts_{timestamp}.csv"

        fieldnames = [
            'issue_number',
            'issue_url',
            'contract_name',
            'description',
            'rationale',
            'examples',
            'parent_category',
            'observed_count',
            'confidence',
            'supporting_evidence'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for analysis in analyses:
                for contract in analysis.suggested_new_contracts:
                    row = {
                        'issue_number': analysis.issue_number,
                        'issue_url': analysis.issue_url,
                        'contract_name': contract.name,
                        'description': contract.description,
                        'rationale': contract.rationale,
                        'examples': '; '.join(contract.examples),
                        'parent_category': contract.parent_category,
                        'observed_count': contract.pattern_frequency.observed_count,
                        'confidence': contract.pattern_frequency.confidence,
                        'supporting_evidence': contract.pattern_frequency.supporting_evidence
                    }
                    writer.writerow(row)

    def _export_metadata(self, metadata: AnalysisMetadataDTO, timestamp: str) -> None:
        """Export analysis metadata to CSV.

        Args:
            metadata: Analysis metadata
            timestamp: Export timestamp
        """
        output_file = self.output_dir / f"analysis_metadata_{timestamp}.csv"

        fieldnames = [
            'repository',
            'repository_url',
            'repository_owner',
            'repository_name',
            'repository_description',
            'repository_stars',
            'repository_forks',
            'repository_language',
            'analysis_timestamp',
            'num_issues',
            'analysis_version',
            'analysis_model',
            'analysis_batch_id'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            row = {
                'repository': metadata.repository,
                'repository_url': metadata.repository_url,
                'repository_owner': metadata.repository_owner,
                'repository_name': metadata.repository_name,
                'repository_description': metadata.repository_description,
                'repository_stars': metadata.repository_stars,
                'repository_forks': metadata.repository_forks,
                'repository_language': metadata.repository_language,
                'analysis_timestamp': metadata.analysis_timestamp,
                'num_issues': metadata.num_issues,
                'analysis_version': metadata.analysis_version,
                'analysis_model': metadata.analysis_model,
                'analysis_batch_id': metadata.analysis_batch_id
            }
            writer.writerow(row)
