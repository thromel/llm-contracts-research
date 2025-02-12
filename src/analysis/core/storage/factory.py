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
                    storage_impls.append(
                        JSONResultsStorage(output_dir=json_dir))
                    logger.info("Created JSON storage implementation")

                elif storage_type == 'mongodb':
                    storage_impls.append(MongoDBRepository(
                        connection_uri=settings.MONGODB_URI,
                        db_name=settings.MONGODB_DB
                    ))
                    logger.info("Created MongoDB storage implementation")

                elif storage_type == 'csv':
                    csv_dir = settings.EXPORT_DIR / 'csv'
                    storage_impls.append(CSVExporter(output_dir=csv_dir))
                    logger.info("Created CSV storage implementation")

                else:
                    logger.warning(
                        "Unknown storage type: {}".format(storage_type))

            except Exception as e:
                logger.error("Failed to create {} storage: {}".format(
                    storage_type, str(e)))

        return storage_impls
