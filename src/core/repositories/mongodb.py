"""MongoDB repository implementation."""
import logging
import os
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ..config import MongoConfig
from ..dto.github import RepositoryDTO, IssueDTO, AnalysisDTO
from ..utils.logging import get_logger
from .base import BaseRepository
from ..interfaces import IssueRepository, IssueData

logger = logging.getLogger(__name__)


class MongoDBRepository(BaseRepository, IssueRepository):
    """MongoDB implementation of the repository interface."""

    def __init__(self, config=None):
        """Initialize MongoDB repository."""
        self.uri = os.getenv('MONGODB_URI')
        self.database = os.getenv('MONGODB_DB', 'github_issues')
        self.client = None
        self.db = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.database]

            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    async def save_repository(self, repo: RepositoryDTO) -> str:
        """Save a repository record."""
        try:
            result = await self.db.repositories.update_one(
                {"github_repo_id": repo.github_repo_id},
                {"$set": repo.to_dict()},
                upsert=True
            )

            if result.upserted_id:
                logger.info(f"Created new repository: {repo.full_name}")
                return str(result.upserted_id)

            logger.info(f"Updated existing repository: {repo.full_name}")
            doc = await self.db.repositories.find_one(
                {"github_repo_id": repo.github_repo_id}
            )
            return str(doc["_id"])

        except Exception as e:
            logger.error(f"Error saving repository {repo.full_name}: {str(e)}")
            raise

    async def save_issue(self, issue: IssueData) -> str:
        """Save an issue to MongoDB.

        Args:
            issue: Issue data to save

        Returns:
            ID of the saved issue
        """
        try:
            result = await self.db.issues.update_one(
                {
                    'github_issue_id': issue.github_issue_id,
                    'repository_id': issue.repository_id
                },
                {'$set': {
                    'github_issue_id': issue.github_issue_id,
                    'repository_id': issue.repository_id,
                    'number': issue.number,
                    'title': issue.title,
                    'body': issue.body,
                    'state': issue.state,
                    'created_at': issue.created_at,
                    'updated_at': issue.updated_at,
                    'closed_at': issue.closed_at,
                    'labels': issue.labels,
                    'assignees': issue.assignees,
                    'comments': issue.comments,
                    'url': issue.url
                }},
                upsert=True
            )

            if result.upserted_id:
                return str(result.upserted_id)
            return str(result.modified_count)

        except Exception as e:
            logger.error(f"Error saving issue {issue.number}: {str(e)}")
            raise

    async def save_analysis(self, analysis: AnalysisDTO) -> str:
        """Save an analysis record."""
        try:
            result = await self.db.analysis.insert_one(analysis.to_dict())
            logger.info(f"Created new analysis for issue {analysis.issue_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(
                f"Error saving analysis for issue {analysis.issue_id}: {str(e)}")
            raise

    async def get_repository_by_full_name(self, full_name: str) -> Optional[RepositoryDTO]:
        """Get repository by full name."""
        try:
            doc = await self.db.repositories.find_one({"full_name": full_name})
            return RepositoryDTO.from_dict(doc) if doc else None
        except Exception as e:
            logger.error(f"Error fetching repository {full_name}: {str(e)}")
            raise

    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[datetime]:
        """Get the timestamp of the last fetched issue for a repository.

        Args:
            repo_id: Repository ID

        Returns:
            Timestamp of the last fetched issue or None
        """
        try:
            result = await self.db.issues.find_one(
                {'repository_id': repo_id},
                sort=[('updated_at', -1)],
                projection={'updated_at': 1}
            )
            return result['updated_at'] if result else None

        except Exception as e:
            logger.error(f"Error fetching last issue timestamp: {str(e)}")
            raise

    async def get_repository_issues(
        self,
        repo_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[IssueDTO]:
        """Get issues for a repository."""
        try:
            cursor = self.db.issues.find(
                {"repository_id": repo_id}
            ).sort("created_at", -1).skip(offset).limit(limit)

            issues = []
            async for doc in cursor:
                issues.append(IssueDTO.from_dict(doc))
            return issues
        except Exception as e:
            logger.error(
                f"Error fetching issues for repository {repo_id}: {str(e)}")
            raise

    async def get_issue_by_number(self, repo_id: str, issue_number: int) -> Optional[IssueDTO]:
        """Get issue by number."""
        try:
            doc = await self.db.issues.find_one({
                "repository_id": repo_id,
                "number": issue_number
            })
            return IssueDTO.from_dict(doc) if doc else None
        except Exception as e:
            logger.error(
                f"Error fetching issue {issue_number} for repository {repo_id}: {str(e)}")
            raise
