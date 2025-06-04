"""
Base storage adapter interface and utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator, TypeVar, Generic
from contextlib import asynccontextmanager
import logging

from ..core.interfaces import Storage
from ..core.exceptions import StorageError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StorageAdapter(Storage):
    """
    Extended storage adapter with additional features like transactions
    and bulk operations.
    """
    
    @abstractmethod
    async def create_index(
        self,
        collection: str,
        index_spec: List[tuple],
        unique: bool = False,
        **kwargs
    ) -> None:
        """Create an index on a collection.
        
        Args:
            collection: Collection name
            index_spec: Index specification
            unique: Whether index should be unique
            **kwargs: Additional index options
        """
        pass
    
    @abstractmethod
    async def drop_collection(self, collection: str) -> None:
        """Drop a collection.
        
        Args:
            collection: Collection to drop
        """
        pass
    
    @abstractmethod
    async def bulk_insert(
        self,
        collection: str,
        documents: List[Dict[str, Any]],
        ordered: bool = False
    ) -> List[str]:
        """Insert multiple documents efficiently.
        
        Args:
            collection: Collection name
            documents: Documents to insert
            ordered: Whether to stop on first error
            
        Returns:
            List of inserted document IDs
        """
        pass
    
    @abstractmethod
    async def bulk_update(
        self,
        collection: str,
        updates: List[Dict[str, Any]]
    ) -> int:
        """Perform bulk updates.
        
        Args:
            collection: Collection name
            updates: List of update operations
            
        Returns:
            Number of documents modified
        """
        pass
    
    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Create a transaction context.
        
        Yields:
            Transaction object
        """
        pass
    
    async def exists(self, collection: str, query: Dict[str, Any]) -> bool:
        """Check if document exists.
        
        Args:
            collection: Collection name
            query: Query to match
            
        Returns:
            True if document exists
        """
        count = await self.count(collection, query)
        return count > 0
    
    async def find_one_or_none(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find one document or return None.
        
        Args:
            collection: Collection name
            query: Query to match
            
        Returns:
            Document or None
        """
        async for doc in self.find(collection, query, limit=1):
            return doc
        return None


class StorageTransaction(ABC):
    """Abstract transaction interface."""
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass
    
    @abstractmethod
    async def save(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        """Save document within transaction.
        
        Args:
            collection: Collection name
            document: Document to save
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """Update documents within transaction.
        
        Args:
            collection: Collection name
            query: Query to match
            update: Update operations
            
        Returns:
            Number of documents updated
        """
        pass


class Repository(Generic[T], ABC):
    """
    Generic repository pattern for domain objects.
    
    Provides a clean interface for data access without exposing
    storage implementation details.
    """
    
    def __init__(self, storage: StorageAdapter, collection: str):
        """Initialize repository.
        
        Args:
            storage: Storage adapter
            collection: Collection name
        """
        self.storage = storage
        self.collection = collection
    
    @abstractmethod
    def _to_document(self, entity: T) -> Dict[str, Any]:
        """Convert entity to storage document.
        
        Args:
            entity: Domain entity
            
        Returns:
            Document dictionary
        """
        pass
    
    @abstractmethod
    def _from_document(self, document: Dict[str, Any]) -> T:
        """Convert storage document to entity.
        
        Args:
            document: Storage document
            
        Returns:
            Domain entity
        """
        pass
    
    async def save(self, entity: T) -> str:
        """Save an entity.
        
        Args:
            entity: Entity to save
            
        Returns:
            Entity ID
        """
        document = self._to_document(entity)
        return await self.storage.save(self.collection, document, upsert=True)
    
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity or None
        """
        document = await self.storage.find_one_or_none(
            self.collection,
            {"_id": entity_id}
        )
        return self._from_document(document) if document else None
    
    async def find_all(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        """Find all entities matching query.
        
        Args:
            query: Optional query filter
            limit: Maximum results
            sort: Sort specification
            
        Returns:
            List of entities
        """
        entities = []
        async for document in self.storage.find(
            self.collection,
            query or {},
            limit=limit
        ):
            entities.append(self._from_document(document))
        return entities
    
    async def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching query.
        
        Args:
            query: Optional query filter
            
        Returns:
            Entity count
        """
        return await self.storage.count(self.collection, query)
    
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if deleted
        """
        result = await self.storage.update(
            self.collection,
            {"_id": entity_id},
            {"$set": {"deleted": True, "deleted_at": datetime.utcnow()}}
        )
        return result > 0
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if exists
        """
        return await self.storage.exists(
            self.collection,
            {"_id": entity_id}
        )