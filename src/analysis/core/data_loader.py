"""Data loading functionality for GitHub issues analysis."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoadError(Exception):
    """Base exception for data loading errors."""
    pass


class CSVDataLoader:
    """Loads GitHub issues data from CSV files."""

    @staticmethod
    def load_from_csv(file_path: Path) -> List[Dict[str, Any]]:
        """Load issues from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of issue dictionaries

        Raises:
            DataLoadError: If loading or parsing fails
        """
        try:
            logger.info("Loading issues from {}".format(file_path))
            df = pd.read_csv(file_path)

            # Convert DataFrame to list of dictionaries
            issues = df.to_dict('records')

            # Validate required fields
            required_fields = {'number', 'title', 'body',
                               'state', 'created_at', 'closed_at', 'url'}
            missing_fields = required_fields - set(df.columns)
            if missing_fields:
                raise DataLoadError("CSV file missing required fields: {}".format(
                    ', '.join(missing_fields)))

            logger.info(
                "Successfully loaded {} issues from CSV".format(len(issues)))
            return issues

        except Exception as exc:
            error_msg = "Error loading CSV file {}: {}".format(
                file_path, str(exc))
            logger.error(error_msg)
            raise DataLoadError(error_msg) from exc
