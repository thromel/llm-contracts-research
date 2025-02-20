"""Repository factory."""
from typing import Optional
from ..config import AppConfig
from .mongodb import MongoDBRepository


class RepositoryFactory:
    """Factory for creating repository instances."""

    _instance: Optional[MongoDBRepository] = None

    @classmethod
    def create_repository(cls, config: AppConfig = None) -> MongoDBRepository:
        """Create a repository instance.

        Args:
            config: Optional application configuration

        Returns:
            Repository instance
        """
        if cls._instance is None:
            cls._instance = MongoDBRepository()

        return cls._instance

    @classmethod
    def get_repository(cls) -> Optional[MongoDBRepository]:
        """Get the current repository instance.

        Returns:
            Current repository instance or None if not created
        """
        return cls._instance
