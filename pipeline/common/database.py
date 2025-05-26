"""
Database management for the LLM Contracts Research Pipeline.

Provides MongoDB integration with full provenance tracking and
research-specific collection management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import os

logger = logging.getLogger(__name__)


class MongoDBManager:
    """
    MongoDB manager for research pipeline with provenance tracking.

    Collections:
    - raw_posts: Initial data from GitHub/Stack Overflow
    - filtered_posts: After keyword pre-filtering
    - labelled_posts: After LLM screening and human labelling
    - labelling_sessions: Metadata for labelling sessions
    - reliability_metrics: Fleiss kappa and other metrics
    - taxonomy_definitions: Hierarchical taxonomy definitions
    """

    def __init__(self, connection_string: str, database_name: str = "llm_contracts_research"):
        """Initialize MongoDB manager.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name for the research project
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

        # Collection names
        self.collections = {
            'raw_posts': 'raw_posts',
            'filtered_posts': 'filtered_posts',
            'llm_screening_results': 'llm_screening_results',
            'agentic_screening_results': 'agentic_screening_results',
            'labelled_posts': 'labelled_posts',
            'labelling_sessions': 'labelling_sessions',
            'reliability_metrics': 'reliability_metrics',
            'taxonomy_definitions': 'taxonomy_definitions',
            'pipeline_runs': 'pipeline_runs'
        }

    async def connect(self) -> None:
        """Connect to MongoDB and setup collections."""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]

            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}")

            # Setup collections and indexes
            await self._setup_collections()

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error setting up MongoDB: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def _setup_collections(self) -> None:
        """Setup collections and create necessary indexes."""

        # Raw posts indexes
        await self.db[self.collections['raw_posts']].create_index([
            ('platform', 1),
            ('source_id', 1)
        ], unique=True)

        await self.db[self.collections['raw_posts']].create_index([
            ('acquisition_timestamp', -1)
        ])

        await self.db[self.collections['raw_posts']].create_index([
            ('created_at', -1)
        ])

        # Filtered posts indexes
        await self.db[self.collections['filtered_posts']].create_index([
            ('raw_post_id', 1)
        ], unique=True)

        await self.db[self.collections['filtered_posts']].create_index([
            ('passed_keyword_filter', 1),
            ('filter_confidence', -1)
        ])

        await self.db[self.collections['filtered_posts']].create_index([
            ('llm_screened', 1)
        ])

        # LLM screening results indexes
        await self.db[self.collections['llm_screening_results']].create_index([
            ('filtered_post_id', 1)
        ])

        await self.db[self.collections['llm_screening_results']].create_index([
            ('decision', 1),
            ('confidence', -1)
        ])

        await self.db[self.collections['llm_screening_results']].create_index([
            ('created_at', -1)
        ])

        # Agentic screening results indexes
        await self.db[self.collections['agentic_screening_results']].create_index([
            ('filtered_post_id', 1)
        ])

        await self.db[self.collections['agentic_screening_results']].create_index([
            ('timestamp', -1)
        ])

        # Labelled posts indexes
        await self.db[self.collections['labelled_posts']].create_index([
            ('filtered_post_id', 1)
        ], unique=True)

        await self.db[self.collections['labelled_posts']].create_index([
            ('labelling_session_id', 1)
        ])

        await self.db[self.collections['labelled_posts']].create_index([
            ('majority_agreement', 1),
            ('required_arbitration', 1)
        ])

        # Reliability metrics indexes
        await self.db[self.collections['reliability_metrics']].create_index([
            ('session_id', 1),
            ('calculation_date', -1)
        ])

        await self.db[self.collections['reliability_metrics']].create_index([
            ('fleiss_kappa', -1)
        ])

        logger.info("MongoDB collections and indexes setup complete")

    # Generic CRUD operations
    async def insert_one(self, collection: str, document: Dict[str, Any]) -> Any:
        """Insert a single document."""
        result = await self.db[collection].insert_one(document)
        return result

    async def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> Any:
        """Insert multiple documents."""
        if not documents:
            return None
        result = await self.db[collection].insert_many(documents, ordered=False)
        return result

    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        return await self.db[collection].find_one(query)

    async def find_many(
        self,
        collection: str,
        query: Dict[str, Any] = None,
        limit: int = None,
        sort: List[tuple] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Find multiple documents."""
        if query is None:
            query = {}

        cursor = self.db[collection].find(query)

        if sort:
            cursor = cursor.sort(sort)

        if limit:
            cursor = cursor.limit(limit)

        async for document in cursor:
            yield document

    async def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> Any:
        """Update a single document."""
        result = await self.db[collection].update_one(query, update)
        return result

    async def update_many(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> Any:
        """Update multiple documents."""
        result = await self.db[collection].update_many(query, update)
        return result

    async def delete_one(self, collection: str, query: Dict[str, Any]) -> Any:
        """Delete a single document."""
        result = await self.db[collection].delete_one(query)
        return result

    async def count_documents(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents matching query."""
        if query is None:
            query = {}
        return await self.db[collection].count_documents(query)

    # Research-specific methods
    async def save_raw_post(self, raw_post_dict: Dict[str, Any]) -> str:
        """Save raw post with deduplication."""
        try:
            result = await self.insert_one(self.collections['raw_posts'], raw_post_dict)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing post if newer
            existing = await self.find_one(
                self.collections['raw_posts'],
                {
                    'platform': raw_post_dict['platform'],
                    'source_id': raw_post_dict['source_id']
                }
            )

            if existing and raw_post_dict.get('updated_at', datetime.min) > existing.get('updated_at', datetime.min):
                await self.update_one(
                    self.collections['raw_posts'],
                    {'_id': existing['_id']},
                    {'$set': raw_post_dict}
                )
                return str(existing['_id'])
            else:
                return str(existing['_id']) if existing else None

    async def get_posts_for_filtering(
        self,
        batch_size: int = 1000,
        exclude_filtered: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get raw posts that need keyword filtering."""

        query = {}
        if exclude_filtered:
            # Find posts that haven't been filtered yet
            filtered_post_ids = []
            async for filtered_post in self.find_many(self.collections['filtered_posts'], {}):
                filtered_post_ids.append(filtered_post['raw_post_id'])

            if filtered_post_ids:
                query['_id'] = {'$nin': filtered_post_ids}

        count = 0
        async for post in self.find_many(
            self.collections['raw_posts'],
            query,
            limit=batch_size,
            sort=[('acquisition_timestamp', 1)]
        ):
            yield post
            count += 1

    async def save_filtered_post(self, filtered_post_dict: Dict[str, Any]) -> str:
        """Save filtered post."""
        try:
            result = await self.insert_one(self.collections['filtered_posts'], filtered_post_dict)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing
            existing = await self.find_one(
                self.collections['filtered_posts'],
                {'raw_post_id': filtered_post_dict['raw_post_id']}
            )
            if existing:
                await self.update_one(
                    self.collections['filtered_posts'],
                    {'_id': existing['_id']},
                    {'$set': filtered_post_dict}
                )
                return str(existing['_id'])
            else:
                raise

    async def save_screening_result(self, screening_result_dict: Dict[str, Any]) -> str:
        """Save LLM screening result."""
        result = await self.insert_one(self.collections['llm_screening_results'], screening_result_dict)

        # Mark the filtered post as screened
        await self.update_one(
            self.collections['filtered_posts'],
            {'_id': screening_result_dict['filtered_post_id']},
            {'$set': {'llm_screened': True}}
        )

        return str(result.inserted_id)

    async def get_posts_for_screening(
        self,
        batch_size: int = 50,
        screening_type: str = "llm_screening"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get filtered posts that need LLM screening."""

        # Get posts that passed keyword filter but haven't been LLM screened
        query = {
            'passed_keyword_filter': True,
            'llm_screened': {'$ne': True}
        }

        count = 0
        async for filtered_post in self.find_many(
            self.collections['filtered_posts'],
            query,
            limit=batch_size,
            sort=[('filter_confidence', -1)]
        ):
            yield filtered_post
            count += 1

    async def get_posts_for_labelling(
        self,
        session_id: str,
        batch_size: int = 50
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get filtered posts that passed the filter and need labelling."""

        # Get posts that passed keyword filter but aren't labelled yet
        labelled_post_ids = []
        async for labelled_post in self.find_many(self.collections['labelled_posts'], {}):
            labelled_post_ids.append(labelled_post['filtered_post_id'])

        query = {
            'passed_keyword_filter': True,
            '_id': {'$nin': labelled_post_ids} if labelled_post_ids else {}
        }

        count = 0
        async for filtered_post in self.find_many(
            self.collections['filtered_posts'],
            query,
            limit=batch_size,
            sort=[('filter_confidence', -1)]
        ):
            yield filtered_post
            count += 1

    async def save_labelling_session(self, session_dict: Dict[str, Any]) -> str:
        """Save labelling session metadata."""
        result = await self.insert_one(self.collections['labelling_sessions'], session_dict)
        return str(result.inserted_id)

    async def save_labelled_post(self, labelled_post_dict: Dict[str, Any]) -> str:
        """Save labelled post."""
        try:
            result = await self.insert_one(self.collections['labelled_posts'], labelled_post_dict)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing
            existing = await self.find_one(
                self.collections['labelled_posts'],
                {'filtered_post_id': labelled_post_dict['filtered_post_id']}
            )
            if existing:
                await self.update_one(
                    self.collections['labelled_posts'],
                    {'_id': existing['_id']},
                    {'$set': labelled_post_dict}
                )
                return str(existing['_id'])
            else:
                raise

    async def get_session_labels_for_kappa(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all labels from a session for Fleiss kappa calculation."""
        labels = []
        async for labelled_post in self.find_many(
            self.collections['labelled_posts'],
            {'labelling_session_id': session_id}
        ):
            labels.append(labelled_post)
        return labels

    async def save_reliability_metrics(self, metrics_dict: Dict[str, Any]) -> str:
        """Save reliability metrics (Fleiss kappa, etc.)."""
        result = await self.insert_one(self.collections['reliability_metrics'], metrics_dict)
        return str(result.inserted_id)

    async def get_latest_kappa_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest kappa calculation for a session."""
        cursor = self.db[self.collections['reliability_metrics']].find(
            {'session_id': session_id}
        ).sort('calculation_date', -1).limit(1)

        async for metrics in cursor:
            return metrics
        return None

    async def save_taxonomy_definition(self, taxonomy_dict: Dict[str, Any]) -> str:
        """Save taxonomy definition."""
        result = await self.insert_one(self.collections['taxonomy_definitions'], taxonomy_dict)
        return str(result.inserted_id)

    async def get_active_taxonomy(self) -> Optional[Dict[str, Any]]:
        """Get the most recent taxonomy definition."""
        cursor = self.db[self.collections['taxonomy_definitions']].find().sort(
            'created_date', -1).limit(1)
        async for taxonomy in cursor:
            return taxonomy
        return None

    # Analytics and reporting
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        stats = {}

        # Raw posts stats
        stats['raw_posts'] = {
            'total': await self.count_documents(self.collections['raw_posts']),
            'github': await self.count_documents(self.collections['raw_posts'], {'platform': 'github'}),
            'stackoverflow': await self.count_documents(self.collections['raw_posts'], {'platform': 'stackoverflow'})
        }

        # Filtered posts stats
        stats['filtered_posts'] = {
            'total': await self.count_documents(self.collections['filtered_posts']),
            'passed_filter': await self.count_documents(self.collections['filtered_posts'], {'passed_keyword_filter': True}),
            'failed_filter': await self.count_documents(self.collections['filtered_posts'], {'passed_keyword_filter': False})
        }

        # Labelled posts stats
        stats['labelled_posts'] = {
            'total': await self.count_documents(self.collections['labelled_posts']),
            'majority_agreement': await self.count_documents(self.collections['labelled_posts'], {'majority_agreement': True}),
            'needs_arbitration': await self.count_documents(self.collections['labelled_posts'], {'required_arbitration': True})
        }

        # Sessions stats
        stats['sessions'] = {
            'total': await self.count_documents(self.collections['labelling_sessions']),
            'active': await self.count_documents(self.collections['labelling_sessions'], {'status': 'active'}),
            'completed': await self.count_documents(self.collections['labelling_sessions'], {'status': 'completed'})
        }

        return stats


class ProvenanceTracker:
    """
    Tracks data lineage and transformations throughout the pipeline.

    Maintains full provenance from raw → filtered → labelled stages
    with timestamps, versions, and transformation metadata.
    """

    def __init__(self, db_manager: MongoDBManager):
        """Initialize provenance tracker.

        Args:
            db_manager: MongoDB manager instance
        """
        self.db = db_manager
        self.provenance_collection = 'provenance_log'

    async def log_transformation(
        self,
        source_id: str,
        source_collection: str,
        target_id: str,
        target_collection: str,
        transformation_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Log a data transformation.

        Args:
            source_id: ID of source document
            source_collection: Source collection name
            target_id: ID of target document  
            target_collection: Target collection name
            transformation_type: Type of transformation performed
            metadata: Additional transformation metadata

        Returns:
            Provenance log entry ID
        """
        provenance_entry = {
            'source_id': source_id,
            'source_collection': source_collection,
            'target_id': target_id,
            'target_collection': target_collection,
            'transformation_type': transformation_type,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        }

        result = await self.db.insert_one(self.provenance_collection, provenance_entry)
        return str(result.inserted_id)

    async def trace_lineage(self, document_id: str) -> List[Dict[str, Any]]:
        """Trace the full lineage of a document back to its source.

        Args:
            document_id: ID of document to trace

        Returns:
            List of provenance entries showing full lineage
        """
        lineage = []
        current_id = document_id

        while current_id:
            # Find provenance entry where this ID is the target
            entry = await self.db.find_one(
                self.provenance_collection,
                {'target_id': current_id}
            )

            if entry:
                lineage.append(entry)
                current_id = entry['source_id']
            else:
                break

        return list(reversed(lineage))  # Return in chronological order

    async def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about pipeline transformations."""
        stats = {}

        # Count by transformation type
        pipeline = [
            {
                '$group': {
                    '_id': '$transformation_type',
                    'count': {'$sum': 1},
                    'latest': {'$max': '$timestamp'}
                }
            }
        ]

        cursor = self.db.db[self.provenance_collection].aggregate(pipeline)
        async for result in cursor:
            stats[result['_id']] = {
                'count': result['count'],
                'latest': result['latest']
            }

        return stats
