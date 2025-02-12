"""Storage factory implementation."""

import logging
from typing import List, Optional, Type

from src.analysis.core.interfaces import IResultWriter, IStorageFactory
from src.analysis.core.storage.json_storage import JSONResultsStorage
from src.analysis.core.storage.mongodb.repository import MongoDBRepository
from src.analysis.core.storage.csv_storage import CSVExporter, CSVStorageAdapter
from src.config import settings

logger = logging.getLogger(__name__)


class StorageStrategy:
    """Base class for storage creation strategies."""

    def create_storage(self, is_intermediate: bool = False) -> Optional[IResultWriter]:
        """Create storage implementation.

        Args:
            is_intermediate: Whether this is for intermediate results

        Returns:
            Storage implementation or None if creation fails
        """
        raise NotImplementedError


class JSONStorageStrategy(StorageStrategy):
    """Strategy for creating JSON storage."""

    def create_storage(self, is_intermediate: bool = False) -> Optional[IResultWriter]:
        """Create JSON storage implementation.

        Args:
            is_intermediate: Whether this is for intermediate results

        Returns:
            JSON storage implementation or None if creation fails
        """
        try:
            json_dir = settings.EXPORT_DIR / \
                ('intermediate' if is_intermediate else 'json')
            json_dir.mkdir(parents=True, exist_ok=True)
            return JSONResultsStorage(output_dir=json_dir)
        except Exception as e:
            logger.error(f"Failed to create JSON storage: {str(e)}")
            return None


class MongoDBStorageStrategy(StorageStrategy):
    """Strategy for creating MongoDB storage."""

    def create_storage(self, is_intermediate: bool = False) -> Optional[IResultWriter]:
        """Create MongoDB storage implementation.

        Args:
            is_intermediate: Whether this is for intermediate results

        Returns:
            MongoDB storage implementation or None if creation fails
        """
        if is_intermediate:
            return None

        try:
            if not settings.MONGODB_URI or not settings.MONGODB_DB:
                logger.warning("MongoDB settings not configured")
                return None
            return MongoDBRepository(
                connection_uri=settings.MONGODB_URI,
                db_name=settings.MONGODB_DB
            )
        except Exception as e:
            logger.error(f"Failed to create MongoDB storage: {str(e)}")
            return None


class CSVStorageStrategy(StorageStrategy):
    """Strategy for creating CSV storage."""

    def create_storage(self, is_intermediate: bool = False) -> Optional[IResultWriter]:
        """Create CSV storage implementation.

        Args:
            is_intermediate: Whether this is for intermediate results

        Returns:
            CSV storage implementation or None if creation fails
        """
        if is_intermediate:
            return None

        try:
            csv_dir = settings.EXPORT_DIR / 'csv'
            csv_dir.mkdir(parents=True, exist_ok=True)
            return CSVStorageAdapter(CSVExporter(output_dir=csv_dir))
        except Exception as e:
            logger.error(f"Failed to create CSV storage: {str(e)}")
            return None


class StorageFactory(IStorageFactory):
    """Factory for creating storage implementations using strategies."""

    # Map storage types to their creation strategies
    STORAGE_STRATEGIES = {
        'json': JSONStorageStrategy,
        'mongodb': MongoDBStorageStrategy,
        'csv': CSVStorageStrategy
    }

    @classmethod
    def create_storage(cls, storage_types: Optional[List[str]] = None, is_intermediate: bool = False) -> List[IResultWriter]:
        """Create storage implementations.

        Args:
            storage_types: List of storage types to create
            is_intermediate: Whether this is for intermediate results

        Returns:
            List of storage implementations
        """
        if storage_types is None:
            storage_types = cls._get_default_storage_types(is_intermediate)

        storage_impls = []
        for storage_type in storage_types:
            strategy_class = cls.STORAGE_STRATEGIES.get(storage_type)
            if not strategy_class:
                logger.warning(f"Unknown storage type: {storage_type}")
                continue

            storage = strategy_class().create_storage(is_intermediate)
            if storage:
                storage_impls.append(storage)

        return storage_impls

    @staticmethod
    def _get_default_storage_types(is_intermediate: bool) -> List[str]:
        """Get default storage types based on settings.

        Args:
            is_intermediate: Whether this is for intermediate results

        Returns:
            List of default storage types
        """
        storage_types = []
        if settings.JSON_EXPORT:
            storage_types.append('json')
        if settings.MONGODB_ENABLED and not is_intermediate:
            storage_types.append('mongodb')
        if settings.CSV_EXPORT and not is_intermediate:
            storage_types.append('csv')
        return storage_types

    @classmethod
    def register_storage_strategy(cls, storage_type: str, strategy_class: Type[StorageStrategy]) -> None:
        """Register a new storage strategy.

        Args:
            storage_type: Storage type identifier
            strategy_class: Strategy class to register
        """
        cls.STORAGE_STRATEGIES[storage_type] = strategy_class
        logger.info(f"Registered storage strategy for type: {storage_type}")
