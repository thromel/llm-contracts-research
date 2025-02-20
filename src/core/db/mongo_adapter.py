from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .base import DatabaseAdapter
from ..config import MongoConfig
from ..dto.github import RepositoryDTO, IssueDTO, AnalysisDTO
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MongoAdapter(DatabaseAdapter):
    def __init__(self, config: MongoConfig):
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            # Use MongoDB Atlas connection string format
            connection_string = (
                f"mongodb+srv://{self.config.user}:{self.config.password}@"
                f"{self.config.host}/{self.config.database}?retryWrites=true&w=majority"
            )
            self.client = AsyncIOMotorClient(connection_string)
            self.db = self.client[self.config.database]

            # Create indexes
            await self.db.repositories.create_index("github_repo_id", unique=True)
            await self.db.repositories.create_index("full_name")
            await self.db.issues.create_index("github_issue_id", unique=True)
            await self.db.issues.create_index("repository_id")
            await self.db.issues.create_index("created_at")
            await self.db.analysis.create_index([("issue_id", 1), ("user_id", 1)])

            logger.info("Successfully connected to MongoDB Atlas")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    async def create_repository(self, repo: RepositoryDTO) -> str:
        """Create a new repository record."""
        try:
            result = await self.db.repositories.update_one(
                {"github_repo_id": repo.github_repo_id},
                {"$set": repo.to_dict()},
                upsert=True
            )

            if result.upserted_id:
                return str(result.upserted_id)

            # If no upsert, get the existing document id
            doc = await self.db.repositories.find_one(
                {"github_repo_id": repo.github_repo_id}
            )
            return str(doc["_id"])

        except Exception as e:
            logger.error(f"Error creating repository: {str(e)}")
            raise

    async def create_issue(self, issue: IssueDTO) -> str:
        """Create a new issue record."""
        try:
            result = await self.db.issues.update_one(
                {"github_issue_id": issue.github_issue_id},
                {"$set": issue.to_dict()},
                upsert=True
            )

            if result.upserted_id:
                return str(result.upserted_id)

            # If no upsert, get the existing document id
            doc = await self.db.issues.find_one(
                {"github_issue_id": issue.github_issue_id}
            )
            return str(doc["_id"])

        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            raise

    async def create_analysis(self, analysis: AnalysisDTO) -> str:
        """Create a new analysis record."""
        try:
            result = await self.db.analysis.insert_one(analysis.to_dict())
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating analysis: {str(e)}")
            raise

    async def get_repository_by_full_name(self, full_name: str) -> Optional[RepositoryDTO]:
        """Retrieve a repository by its full name."""
        try:
            doc = await self.db.repositories.find_one({"full_name": full_name})
            return RepositoryDTO.from_dict(doc) if doc else None
        except Exception as e:
            logger.error(f"Error fetching repository: {str(e)}")
            raise

    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[datetime]:
        """Get the timestamp of the last fetched issue for a repository."""
        try:
            cursor = self.db.issues.find(
                {"repository_id": repo_id}
            ).sort("updated_at", -1).limit(1)

            # Check if there are any documents
            async for doc in cursor:
                return doc["updated_at"]

            # If no documents found, return None
            return None

        except Exception as e:
            logger.error(f"Error fetching last issue timestamp: {str(e)}")
            raise

    async def get_issues_for_repository(
        self,
        repo_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[IssueDTO]:
        """Retrieve issues for a specific repository."""
        try:
            cursor = self.db.issues.find(
                {"repository_id": repo_id}
            ).sort("created_at", -1).skip(offset).limit(limit)

            issues = []
            async for doc in cursor:
                issues.append(IssueDTO.from_dict(doc))
            return issues
        except Exception as e:
            logger.error(f"Error fetching issues: {str(e)}")
            raise
