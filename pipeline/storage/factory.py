"""
Storage factory for creating storage adapters.
"""

from typing import Dict, Any
from .base import StorageAdapter
from .mongodb import MongoDBAdapter


class StorageFactory:
    """Factory for creating storage adapters."""
    
    @staticmethod
    def create_storage(storage_type: str, config: Dict[str, Any]) -> StorageAdapter:
        """Create a storage adapter based on type and configuration."""
        if storage_type.lower() == "mongodb":
            return MongoDBAdapter(
                connection_string=config.get("connection_string"),
                database_name=config.get("database_name")
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")