"""Data loading functionality for GitHub issues analysis."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import csv
import logging
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

from src.utils.logger import setup_logger
from src.analysis.core.dto import GithubIssueDTO as IssueDTO, CommentDTO

logger = setup_logger(__name__)


class DataLoadError(Exception):
    """Base exception for data loading errors."""
    pass


class CSVDataLoader:
    """Loads GitHub issues data from CSV files."""

    @staticmethod
    def load_from_csv(file_path: Path) -> List[Dict[str, Any]]:
        """Load issues from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of issue dictionaries

        Raises:
            DataLoadError: If loading or parsing fails
        """
        try:
            logger.info("Loading issues from {}".format(file_path))
            df = pd.read_csv(file_path)

            # Convert DataFrame to list of dictionaries
            issues = df.to_dict('records')

            # Validate required fields
            required_fields = {'number', 'title', 'body',
                               'state', 'created_at', 'closed_at', 'url'}
            missing_fields = required_fields - set(df.columns)
            if missing_fields:
                raise DataLoadError("CSV file missing required fields: {}".format(
                    ', '.join(missing_fields)))

            logger.info(
                "Successfully loaded {} issues from CSV".format(len(issues)))
            return issues

        except Exception as exc:
            error_msg = "Error loading CSV file {}: {}".format(
                file_path, str(exc))
            logger.error(error_msg)
            raise DataLoadError(error_msg) from exc


class MongoDBDataLoader:
    """Loads issue data from MongoDB."""

    def __init__(self, mongodb_uri: str, mongodb_db: str):
        """Initialize MongoDB data loader.

        Args:
            mongodb_uri: MongoDB connection URI
            mongodb_db: MongoDB database name
        """
        self.mongodb_uri = mongodb_uri
        self.mongodb_db = mongodb_db
        self.client = None
        self.db = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client[self.mongodb_db]
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise DataLoadError(f"Failed to connect to MongoDB: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    async def load_repository_issues(self, repo_name: str, limit: Optional[int] = None) -> List[IssueDTO]:
        """Load issues for a repository from MongoDB.

        Args:
            repo_name: Repository name (format: owner/repo)
            limit: Maximum number of issues to load

        Returns:
            List of IssueDTO objects

        Raises:
            DataLoadError: If loading fails
        """
        try:
            # First, find the repository ID
            owner, repo = repo_name.split('/')
            repo_query = {"full_name": repo_name}
            repo_doc = await self.db.repositories.find_one(repo_query)

            if not repo_doc:
                raise DataLoadError(
                    f"Repository {repo_name} not found in MongoDB")

            repo_id = str(repo_doc.get("github_repo_id"))

            # Now find issues for this repository
            query = {"repository_id": repo_id}
            cursor = self.db.issues.find(query)

            if limit:
                cursor = cursor.limit(limit)

            issues = []
            async for doc in cursor:
                # Convert comments to CommentDTO
                comments = []
                for comment_data in doc.get('comments', []):
                    comment = CommentDTO(
                        id=str(comment_data.get('id', '')),
                        body=comment_data.get('body', ''),
                        created_at=comment_data.get(
                            'created_at', datetime.now())
                    )
                    comments.append(comment)

                # Create IssueDTO
                issue = IssueDTO(
                    id=str(doc.get('github_issue_id', '')),
                    number=doc.get('number', 0),
                    title=doc.get('title', ''),
                    body=doc.get('body', ''),
                    state=doc.get('state', ''),
                    created_at=doc.get('created_at', datetime.now()),
                    url=doc.get('url', ''),
                    # Limit to first 10 comments for analysis
                    first_comments=comments[:10]
                )
                issues.append(issue)

            logger.info(
                f"Loaded {len(issues)} issues for {repo_name} from MongoDB")
            return issues

        except Exception as e:
            logger.error(f"Failed to load issues from MongoDB: {str(e)}")
            raise DataLoadError(
                f"Failed to load issues from MongoDB: {str(e)}")
