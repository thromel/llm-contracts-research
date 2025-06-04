"""Enhanced database management with connection pooling and validation."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, TypeVar, AsyncGenerator
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pydantic import BaseModel, Field, ConfigDict
from pymongo import ASCENDING, DESCENDING
from contextlib import asynccontextmanager

from pipeline.foundation.config import DatabaseConfig
from pipeline.foundation.logging import get_logger, LogContext
from pipeline.foundation.retry import with_retry, DATABASE_RETRY
from pipeline.foundation.types import PipelineStage

T = TypeVar('T', bound='MongoDocument')


class MongoDocument(BaseModel):
    """Base class for MongoDB documents with auto-validation."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        str_strip_whitespace=True
    )
    
    id: Optional[str] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata for document management
    _collection_name: Optional[str] = None
    _indexes: List[Dict[str, Any]] = []
    
    @classmethod
    def get_collection_name(cls) -> str:
        """Get the MongoDB collection name for this document type."""
        if cls._collection_name:
            return cls._collection_name
        return cls.__name__.lower() + "s"
    
    @classmethod
    def get_indexes(cls) -> List[Dict[str, Any]]:
        """Get the index definitions for this document type."""
        base_indexes = [
            {"keys": [("created_at", DESCENDING)], "name": "created_at_desc"},
            {"keys": [("updated_at", DESCENDING)], "name": "updated_at_desc"}
        ]
        return base_indexes + cls._indexes
    
    def model_dump_mongo(self) -> Dict[str, Any]:
        """Convert to MongoDB-compatible dictionary."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        
        # Handle ObjectId conversion
        if self.id:
            data["_id"] = ObjectId(self.id) if not isinstance(self.id, ObjectId) else self.id
        
        return data
    
    @classmethod
    def from_mongo(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from MongoDB document."""
        if "_id" in data:
            data["_id"] = str(data["_id"])
        
        return cls.model_validate(data)
    
    def mark_updated(self) -> None:
        """Mark document as updated with current timestamp."""
        self.updated_at = datetime.utcnow()


class DatabaseManager:
    """Enhanced database manager with connection pooling and validation."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.logger = get_logger(__name__, LogContext(component="DatabaseManager"))
        self._connected = False
        self._collections: Dict[str, AsyncIOMotorCollection] = {}
    
    async def connect(self) -> None:
        """Connect to MongoDB with connection pooling."""
        if self._connected:
            return
        
        try:
            self.logger.info("Connecting to MongoDB", connection_string=self.config.connection_string)
            
            self.client = AsyncIOMotorClient(
                self.config.connection_string,
                maxPoolSize=self.config.connection_pool_size,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=self.config.query_timeout * 1000,
                socketTimeoutMS=self.config.query_timeout * 1000,
                connectTimeoutMS=self.config.query_timeout * 1000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.config.database_name]
            self._connected = True
            
            self.logger.info("Successfully connected to MongoDB", 
                           database=self.config.database_name,
                           pool_size=self.config.connection_pool_size)
            
        except Exception as e:
            self.logger.error("Failed to connect to MongoDB", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self._connected = False
            self.logger.info("Disconnected from MongoDB")
    
    async def ensure_connected(self) -> None:
        """Ensure database connection is active."""
        if not self._connected:
            await self.connect()
    
    def get_collection(self, document_class: Type[MongoDocument]) -> AsyncIOMotorCollection:
        """Get collection for document class with caching."""
        collection_name = document_class.get_collection_name()
        
        if collection_name not in self._collections:
            if not self.database:
                raise RuntimeError("Database not connected")
            self._collections[collection_name] = self.database[collection_name]
        
        return self._collections[collection_name]
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def save(self, document: MongoDocument, upsert: bool = False) -> str:
        """Save document to MongoDB with retry logic."""
        await self.ensure_connected()
        
        collection = self.get_collection(type(document))
        document.mark_updated()
        data = document.model_dump_mongo()
        
        op_logger = self.logger.start_operation("save_document", 
                                              collection=collection.name,
                                              upsert=upsert)
        
        try:
            if document.id and not upsert:
                # Update existing document
                result = await collection.replace_one(
                    {"_id": ObjectId(document.id)},
                    data,
                    upsert=False
                )
                
                if result.matched_count == 0:
                    raise ValueError(f"Document with id {document.id} not found")
                
                op_logger.complete("Document updated successfully")
                return document.id
            else:
                # Insert new document or upsert
                if upsert and document.id:
                    result = await collection.replace_one(
                        {"_id": ObjectId(document.id)},
                        data,
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        document.id = str(result.upserted_id)
                    
                    op_logger.complete("Document upserted successfully")
                    return document.id
                else:
                    # Regular insert
                    if "_id" in data:
                        del data["_id"]
                    
                    result = await collection.insert_one(data)
                    document.id = str(result.inserted_id)
                    
                    op_logger.complete("Document inserted successfully")
                    return document.id
                    
        except Exception as e:
            op_logger.fail("Failed to save document", exception=e)
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def save_many(self, documents: List[MongoDocument]) -> List[str]:
        """Save multiple documents efficiently."""
        await self.ensure_connected()
        
        if not documents:
            return []
        
        # Group by collection
        by_collection: Dict[str, List[MongoDocument]] = {}
        for doc in documents:
            collection_name = type(doc).get_collection_name()
            if collection_name not in by_collection:
                by_collection[collection_name] = []
            by_collection[collection_name].append(doc)
        
        op_logger = self.logger.start_operation("save_many_documents",
                                              total_documents=len(documents),
                                              collections=len(by_collection))
        
        try:
            all_ids = []
            
            for collection_name, docs in by_collection.items():
                collection = self.database[collection_name]
                
                # Prepare documents for insertion
                data_list = []
                for doc in docs:
                    doc.mark_updated()
                    data = doc.model_dump_mongo()
                    if "_id" in data:
                        del data["_id"]
                    data_list.append(data)
                
                # Batch insert
                result = await collection.insert_many(data_list, ordered=False)
                
                # Update document IDs
                for doc, inserted_id in zip(docs, result.inserted_ids):
                    doc.id = str(inserted_id)
                    all_ids.append(doc.id)
            
            op_logger.complete("Batch save completed successfully",
                             inserted_count=len(all_ids))
            return all_ids
            
        except Exception as e:
            op_logger.fail("Failed to save documents", exception=e)
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def find_one(self, document_class: Type[T], filter_dict: Dict[str, Any]) -> Optional[T]:
        """Find single document by filter."""
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        
        try:
            data = await collection.find_one(filter_dict)
            if data:
                return document_class.from_mongo(data)
            return None
            
        except Exception as e:
            self.logger.error("Failed to find document", 
                            collection=collection.name,
                            filter=filter_dict,
                            error=str(e))
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def find_by_id(self, document_class: Type[T], document_id: str) -> Optional[T]:
        """Find document by ID."""
        return await self.find_one(document_class, {"_id": ObjectId(document_id)})
    
    async def find_many(self, 
                       document_class: Type[T], 
                       filter_dict: Dict[str, Any] = None,
                       sort: List[tuple] = None,
                       limit: Optional[int] = None,
                       skip: Optional[int] = None) -> AsyncGenerator[T, None]:
        """Find multiple documents with streaming support."""
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        filter_dict = filter_dict or {}
        
        try:
            cursor = collection.find(filter_dict)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            async for data in cursor:
                yield document_class.from_mongo(data)
                
        except Exception as e:
            self.logger.error("Failed to find documents",
                            collection=collection.name,
                            filter=filter_dict,
                            error=str(e))
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def count(self, document_class: Type[MongoDocument], filter_dict: Dict[str, Any] = None) -> int:
        """Count documents matching filter."""
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        filter_dict = filter_dict or {}
        
        try:
            return await collection.count_documents(filter_dict)
        except Exception as e:
            self.logger.error("Failed to count documents",
                            collection=collection.name,
                            filter=filter_dict,
                            error=str(e))
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def delete_one(self, document_class: Type[MongoDocument], filter_dict: Dict[str, Any]) -> bool:
        """Delete single document."""
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        
        try:
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            self.logger.error("Failed to delete document",
                            collection=collection.name,
                            filter=filter_dict,
                            error=str(e))
            raise
    
    @with_retry(**DATABASE_RETRY.__dict__)
    async def delete_by_id(self, document_class: Type[MongoDocument], document_id: str) -> bool:
        """Delete document by ID."""
        return await self.delete_one(document_class, {"_id": ObjectId(document_id)})
    
    async def create_indexes(self, document_class: Type[MongoDocument]) -> None:
        """Create indexes for document class."""
        if not self.config.enable_indexes:
            return
        
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        indexes = document_class.get_indexes()
        
        if not indexes:
            return
        
        op_logger = self.logger.start_operation("create_indexes",
                                              collection=collection.name,
                                              index_count=len(indexes))
        
        try:
            for index_spec in indexes:
                keys = index_spec["keys"]
                options = {k: v for k, v in index_spec.items() if k != "keys"}
                
                await collection.create_index(keys, **options)
            
            op_logger.complete("Indexes created successfully")
            
        except Exception as e:
            op_logger.fail("Failed to create indexes", exception=e)
            raise
    
    async def aggregate(self, 
                       document_class: Type[MongoDocument], 
                       pipeline: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute aggregation pipeline."""
        await self.ensure_connected()
        
        collection = self.get_collection(document_class)
        
        try:
            async for result in collection.aggregate(pipeline):
                yield result
        except Exception as e:
            self.logger.error("Failed to execute aggregation",
                            collection=collection.name,
                            pipeline=pipeline,
                            error=str(e))
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        await self.ensure_connected()
        
        async with await self.client.start_session() as session:
            async with session.start_transaction():
                yield session
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return status."""
        try:
            await self.ensure_connected()
            
            # Run ping command
            result = await self.client.admin.command('ping')
            
            # Get server info
            server_info = await self.client.admin.command('serverStatus')
            
            return {
                "status": "healthy",
                "ping": result,
                "uptime": server_info.get("uptime", 0),
                "connections": server_info.get("connections", {}),
                "database": self.config.database_name,
                "connected": self._connected
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": self._connected
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database() first.")
    return _db_manager


def initialize_database(config: DatabaseConfig) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(config)
    return _db_manager