"""CSV storage implementation for exporting analysis results."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from src.analysis.core.interfaces import IResultWriter
from src.analysis.core.dto import AnalysisMetadataDTO, ContractAnalysisDTO

logger = logging.getLogger(__name__)


class CSVExporter:
    """Handles exporting data to CSV files."""

    def __init__(self, output_dir: Path):
        """Initialize CSV exporter.

        Args:
            output_dir: Directory to store CSV files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> None:
        """Export data to a CSV file.

        Args:
            data: List of dictionaries containing the data to export
            filename: Name of the output CSV file
        """
        if not data:
            logger.warning(f"No data to export to {filename}")
            return

        filepath = self.output_dir / filename
        try:
            # Get headers from the first item's keys
            fieldnames = list(data[0].keys())

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Successfully exported data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {str(e)}")
            raise


class CSVStorageAdapter(IResultWriter):
    """Adapter to use CSVExporter as a storage implementation."""

    def __init__(self, exporter: CSVExporter):
        """Initialize CSV storage adapter.

        Args:
            exporter: CSVExporter instance to use
        """
        self.exporter = exporter

    def write(self, data: Dict[str, Any], **kwargs) -> None:
        """Write data to CSV storage.

        Args:
            data: Dictionary containing the data to write
            **kwargs: Additional arguments (e.g., filename)
        """
        try:
            # Convert single dict to list for consistent handling
            data_list = [data] if isinstance(data, dict) else data

            # Use timestamp as default filename if not provided
            filename = kwargs.get('filename', 'analysis_results.csv')
            if not filename.endswith('.csv'):
                filename += '.csv'

            self.exporter.export_to_csv(data_list, filename)
        except Exception as e:
            logger.error(f"Failed to write data to CSV storage: {str(e)}")
            raise

    def save_results(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis results to CSV.

        Args:
            analyzed_issues: List of analyzed issues
            metadata: Analysis metadata
        """
        try:
            # Convert DTOs to dictionaries
            issues_data = []
            for issue in analyzed_issues:
                issue_dict = issue.__dict__.copy()
                # Add metadata fields
                issue_dict.update({
                    'repository_name': metadata.repository_name,
                    'repository_owner': metadata.repository_owner,
                    'analysis_timestamp': metadata.analysis_timestamp,
                    'analysis_version': metadata.analysis_version,
                    'analysis_model': metadata.analysis_model,
                    'analysis_batch_id': metadata.analysis_batch_id
                })
                issues_data.append(issue_dict)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"github_issues_analysis_{timestamp}_final.csv"

            self.exporter.export_to_csv(issues_data, filename)
            logger.info(
                f"Successfully saved analysis results to CSV: {filename}")
        except Exception as e:
            logger.error(f"Failed to save analysis results to CSV: {str(e)}")
            raise

    def close(self) -> None:
        """Close any open resources."""
        pass  # No resources to close for CSV storage
