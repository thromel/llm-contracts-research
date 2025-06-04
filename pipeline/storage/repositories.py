"""
Repository classes for different data types.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from .base import StorageAdapter
from ..common.models import RawPost, FilteredPost


class BaseRepository:
    """Base repository class."""
    
    def __init__(self, storage: StorageAdapter, collection_name: str):
        """Initialize repository."""
        self.storage = storage
        self.collection = collection_name


class RawPostRepository(BaseRepository):
    """Repository for raw posts."""
    
    def __init__(self, storage: StorageAdapter):
        """Initialize raw post repository."""
        super().__init__(storage, "raw_posts")
        
    async def save(self, post: RawPost) -> str:
        """Save a raw post."""
        return await self.storage.save(self.collection, post.to_dict())
        
    async def find_unfiltered(self, limit: Optional[int] = None) -> AsyncIterator[RawPost]:
        """Find unfiltered raw posts."""
        async for doc in self.storage.find(
            self.collection, 
            {"filtered": {"$ne": True}}, 
            limit=limit
        ):
            yield RawPost.from_dict(doc)


class FilteredPostRepository(BaseRepository):
    """Repository for filtered posts."""
    
    def __init__(self, storage: StorageAdapter):
        """Initialize filtered post repository."""
        super().__init__(storage, "filtered_posts")
        
    async def save(self, post: FilteredPost) -> str:
        """Save a filtered post."""
        return await self.storage.save(self.collection, post.to_dict())
        
    async def find_passed_filter(self, limit: Optional[int] = None) -> AsyncIterator[FilteredPost]:
        """Find posts that passed filtering."""
        async for doc in self.storage.find(
            self.collection,
            {"passed_keyword_filter": True},
            limit=limit
        ):
            yield FilteredPost.from_dict(doc)


class ScreeningResultRepository(BaseRepository):
    """Repository for LLM screening results."""
    
    def __init__(self, storage: StorageAdapter):
        """Initialize screening result repository."""
        super().__init__(storage, "llm_screening_results")
        
    async def save(self, result: Dict[str, Any]) -> str:
        """Save a screening result."""
        return await self.storage.save(self.collection, result)


class LabeledPostRepository(BaseRepository):
    """Repository for labeled posts."""
    
    def __init__(self, storage: StorageAdapter):
        """Initialize labeled post repository."""
        super().__init__(storage, "labeled_posts")
        
    async def save(self, labeled_post: Dict[str, Any]) -> str:
        """Save a labeled post."""
        return await self.storage.save(self.collection, labeled_post)