"""Tests for database infrastructure."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from bson import ObjectId

from pipeline.infrastructure.database import DatabaseManager, MongoDocument, get_database_manager, initialize_database
from pipeline.foundation.config import DatabaseConfig
from pipeline.foundation.types import ConfigValidationError


class TestDocument(MongoDocument):
    """Test document class."""
    
    _collection_name = "test_documents"
    _indexes = [
        {"keys": [("name", 1)], "name": "name_unique", "unique": True}
    ]
    
    name: str
    value: int = 0


class TestMongoDocument:
    """Test MongoDocument base class."""
    
    def test_basic_creation(self):
        """Test basic document creation."""
        doc = TestDocument(name="test", value=42)
        
        assert doc.name == "test"
        assert doc.value == 42
        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert doc.id is None
    
    def test_collection_name(self):
        """Test collection name derivation."""
        assert TestDocument.get_collection_name() == "test_documents"
    
    def test_indexes(self):
        """Test index definitions."""
        indexes = TestDocument.get_indexes()
        
        # Should have base indexes plus custom ones
        assert len(indexes) >= 3
        
        # Check for custom index
        custom_index = next((idx for idx in indexes if idx["name"] == "name_unique"), None)
        assert custom_index is not None
        assert custom_index["unique"] is True
    
    def test_model_dump_mongo(self):
        """Test MongoDB serialization."""
        doc = TestDocument(name="test", value=42)
        doc.id = "507f1f77bcf86cd799439011"
        
        mongo_data = doc.model_dump_mongo()
        
        assert mongo_data["name"] == "test"
        assert mongo_data["value"] == 42
        assert isinstance(mongo_data["_id"], ObjectId)
        assert "created_at" in mongo_data
        assert "updated_at" in mongo_data
    
    def test_from_mongo(self):
        """Test creating document from MongoDB data."""
        mongo_data = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "test",
            "value": 42,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        doc = TestDocument.from_mongo(mongo_data)
        
        assert doc.name == "test"
        assert doc.value == 42
        assert doc.id == "507f1f77bcf86cd799439011"
    
    def test_mark_updated(self):
        """Test updating timestamp."""
        doc = TestDocument(name="test", value=42)
        original_time = doc.updated_at
        
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.001)
        
        doc.mark_updated()
        
        assert doc.updated_at > original_time


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.fixture
    def db_config(self):
        """Database configuration for testing."""
        return DatabaseConfig(
            connection_string="mongodb://localhost:27017/",
            database_name="test_db",
            connection_pool_size=5
        )
    
    @pytest.fixture
    def db_manager(self, db_config):
        """Database manager instance."""
        return DatabaseManager(db_config)
    
    @pytest.fixture
    def mock_client(self):
        """Mock MongoDB client."""
        client = AsyncMock()
        
        # Mock admin command for ping
        client.admin.command = AsyncMock(return_value={"ok": 1})
        
        # Mock database and collection
        database = AsyncMock()
        collection = AsyncMock()
        
        client.__getitem__ = Mock(return_value=database)
        database.__getitem__ = Mock(return_value=collection)
        
        return client, database, collection
    
    @pytest.mark.asyncio
    async def test_connect_success(self, db_manager, mock_client):
        """Test successful database connection."""
        client, database, collection = mock_client
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            assert db_manager._connected is True
            assert db_manager.client == client
            assert db_manager.database == database
            
            # Verify ping was called
            client.admin.command.assert_called_once_with('ping')
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, db_manager):
        """Test database connection failure."""
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient') as mock_motor:
            mock_motor.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await db_manager.connect()
            
            assert db_manager._connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, db_manager, mock_client):
        """Test database disconnection."""
        client, _, _ = mock_client
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            await db_manager.disconnect()
            
            assert db_manager._connected is False
            client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_new_document(self, db_manager, mock_client):
        """Test saving a new document."""
        client, database, collection = mock_client
        
        # Mock insert result
        insert_result = Mock()
        insert_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")
        collection.insert_one = AsyncMock(return_value=insert_result)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            doc = TestDocument(name="test", value=42)
            result_id = await db_manager.save(doc)
            
            assert result_id == "507f1f77bcf86cd799439011"
            assert doc.id == "507f1f77bcf86cd799439011"
            collection.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_existing_document(self, db_manager, mock_client):
        """Test updating an existing document."""
        client, database, collection = mock_client
        
        # Mock replace result
        replace_result = Mock()
        replace_result.matched_count = 1
        collection.replace_one = AsyncMock(return_value=replace_result)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            doc = TestDocument(name="test", value=42)
            doc.id = "507f1f77bcf86cd799439011"
            
            result_id = await db_manager.save(doc)
            
            assert result_id == "507f1f77bcf86cd799439011"
            collection.replace_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_many_documents(self, db_manager, mock_client):
        """Test batch saving documents."""
        client, database, collection = mock_client
        
        # Mock insert_many result
        insert_result = Mock()
        insert_result.inserted_ids = [
            ObjectId("507f1f77bcf86cd799439011"),
            ObjectId("507f1f77bcf86cd799439012")
        ]
        collection.insert_many = AsyncMock(return_value=insert_result)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            docs = [
                TestDocument(name="test1", value=1),
                TestDocument(name="test2", value=2)
            ]
            
            result_ids = await db_manager.save_many(docs)
            
            assert len(result_ids) == 2
            assert docs[0].id == "507f1f77bcf86cd799439011"
            assert docs[1].id == "507f1f77bcf86cd799439012"
            collection.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_one(self, db_manager, mock_client):
        """Test finding a single document."""
        client, database, collection = mock_client
        
        # Mock find_one result
        mongo_data = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "test",
            "value": 42,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        collection.find_one = AsyncMock(return_value=mongo_data)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            doc = await db_manager.find_one(TestDocument, {"name": "test"})
            
            assert doc is not None
            assert doc.name == "test"
            assert doc.value == 42
            assert doc.id == "507f1f77bcf86cd799439011"
            collection.find_one.assert_called_once_with({"name": "test"})
    
    @pytest.mark.asyncio
    async def test_find_by_id(self, db_manager, mock_client):
        """Test finding document by ID."""
        client, database, collection = mock_client
        
        # Mock find_one result
        mongo_data = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "name": "test",
            "value": 42,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        collection.find_one = AsyncMock(return_value=mongo_data)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            doc = await db_manager.find_by_id(TestDocument, "507f1f77bcf86cd799439011")
            
            assert doc is not None
            assert doc.id == "507f1f77bcf86cd799439011"
            # Verify ObjectId was used in query
            call_args = collection.find_one.call_args[0][0]
            assert isinstance(call_args["_id"], ObjectId)
    
    @pytest.mark.asyncio
    async def test_count(self, db_manager, mock_client):
        """Test counting documents."""
        client, database, collection = mock_client
        
        collection.count_documents = AsyncMock(return_value=5)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            count = await db_manager.count(TestDocument, {"value": {"$gt": 0}})
            
            assert count == 5
            collection.count_documents.assert_called_once_with({"value": {"$gt": 0}})
    
    @pytest.mark.asyncio
    async def test_delete_one(self, db_manager, mock_client):
        """Test deleting a document."""
        client, database, collection = mock_client
        
        # Mock delete result
        delete_result = Mock()
        delete_result.deleted_count = 1
        collection.delete_one = AsyncMock(return_value=delete_result)
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            success = await db_manager.delete_one(TestDocument, {"name": "test"})
            
            assert success is True
            collection.delete_one.assert_called_once_with({"name": "test"})
    
    @pytest.mark.asyncio
    async def test_create_indexes(self, db_manager, mock_client):
        """Test creating indexes."""
        client, database, collection = mock_client
        
        collection.create_index = AsyncMock()
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            await db_manager.create_indexes(TestDocument)
            
            # Should have called create_index for each index
            assert collection.create_index.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, db_manager, mock_client):
        """Test health check when database is healthy."""
        client, database, collection = mock_client
        
        # Mock server status
        client.admin.command.side_effect = [
            {"ok": 1},  # ping
            {"uptime": 3600, "connections": {"current": 5}}  # serverStatus
        ]
        
        with patch('pipeline.infrastructure.database.AsyncIOMotorClient', return_value=client):
            await db_manager.connect()
            
            health = await db_manager.health_check()
            
            assert health["status"] == "healthy"
            assert health["uptime"] == 3600
            assert health["connected"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, db_manager):
        """Test health check when database is unhealthy."""
        health = await db_manager.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert health["connected"] is False


class TestGlobalManager:
    """Test global database manager functions."""
    
    def test_initialize_database(self):
        """Test initializing global database manager."""
        config = DatabaseConfig(
            connection_string="mongodb://localhost:27017/",
            database_name="test_db"
        )
        
        manager = initialize_database(config)
        
        assert isinstance(manager, DatabaseManager)
        assert manager.config == config
        
        # Should be able to get the same instance
        same_manager = get_database_manager()
        assert same_manager is manager
    
    def test_get_database_manager_not_initialized(self):
        """Test getting manager when not initialized."""
        # Reset global state
        import pipeline.infrastructure.database
        pipeline.infrastructure.database._db_manager = None
        
        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            get_database_manager()


if __name__ == "__main__":
    pytest.main([__file__])