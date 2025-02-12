"""Factory for creating storage implementations."""

import logging
from pathlib import Path
from typing import Optional, List

from src.config import settings
from src.analysis.core.storage.base import ResultsStorage
from src.analysis.core.storage.json_storage import JSONResultsStorage
from src.analysis.core.storage.mongodb.repository import MongoDBRepository
from src.analysis.core.exporters.csv_exporter import CSVExporter

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage implementations."""

    @staticmethod
    def create_storage(storage_types: Optional[List[str]] = None) -> List[ResultsStorage]:
        """Create storage implementations.

        Args:
            storage_types: List of storage types to create (json, mongodb, csv)
                         If None, uses settings to determine enabled storage types

        Returns:
            List of storage implementations
        """
        if storage_types is None:
            storage_types = []
            if settings.JSON_EXPORT:
                storage_types.append('json')
            if settings.MONGODB_ENABLED:
                storage_types.append('mongodb')
            if settings.CSV_EXPORT:
                storage_types.append('csv')

        storage_impls = []

        for storage_type in storage_types:
            try:
                if storage_type == 'json':
                    json_dir = settings.EXPORT_DIR / 'json'
                    json_dir.mkdir(parents=True, exist_ok=True)
                    storage_impls.append(
                        JSONResultsStorage(output_dir=json_dir))
                    logger.info("Created JSON storage implementation")

                elif storage_type == 'mongodb':
                    if not settings.MONGODB_URI or not settings.MONGODB_DB:
                        logger.warning(
                            "MongoDB settings not configured, skipping MongoDB storage")
                        continue
                    storage_impls.append(MongoDBRepository(
                        connection_uri=settings.MONGODB_URI,
                        db_name=settings.MONGODB_DB
                    ))
                    logger.info("Created MongoDB storage implementation")

                elif storage_type == 'csv':
                    csv_dir = settings.EXPORT_DIR / 'csv'
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    exporter = CSVExporter(output_dir=csv_dir)
                    # Wrap CSV exporter in a ResultsStorage adapter
                    storage_impls.append(CSVStorageAdapter(exporter))
                    logger.info("Created CSV storage implementation")

                else:
                    logger.warning(f"Unknown storage type: {storage_type}")

            except Exception as e:
                logger.error(
                    f"Failed to create {storage_type} storage: {str(e)}")

        return storage_impls


class CSVStorageAdapter(ResultsStorage):
    """Adapter to make CSVExporter conform to ResultsStorage protocol."""

    def __init__(self, exporter: CSVExporter):
        """Initialize adapter with CSV exporter.

        Args:
            exporter: CSV exporter instance
        """
        self.exporter = exporter

    def save_results(self, analyzed_issues: list[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save results using CSV exporter.

        Args:
            analyzed_issues: List of analysis results
            metadata: Analysis metadata
        """
        results = AnalysisResultsDTO(
            metadata=metadata, analyzed_issues=analyzed_issues)
        self.exporter.export_results(results)

    def load_results(self, file_path: str) -> AnalysisResultsDTO:
        """Load results from CSV is not supported.

        Args:
            file_path: Path to results file

        Raises:
            NotImplementedError: CSV loading is not supported
        """
        raise NotImplementedError("Loading results from CSV is not supported")
