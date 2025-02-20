import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import DatabaseAdapter
from ..config import PostgresConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PostgresAdapter(DatabaseAdapter):
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.pool = None

    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                host=self.config.host,
                port=self.config.port,
                min_size=5,
                max_size=20
            )
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed PostgreSQL connection pool")

    async def create_repository(self, repo_data: Dict[str, Any]) -> str:
        """Create a new repository record."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO repositories (
                        github_repo_id, owner, name, full_name, url, 
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (github_repo_id) DO UPDATE 
                    SET updated_at = EXCLUDED.updated_at
                    RETURNING id;
                """
                repo_id = await conn.fetchval(
                    query,
                    repo_data['github_repo_id'],
                    repo_data['owner'],
                    repo_data['name'],
                    repo_data['full_name'],
                    repo_data['url'],
                    repo_data.get('created_at', datetime.utcnow()),
                    repo_data.get('updated_at', datetime.utcnow())
                )
                return str(repo_id)
            except Exception as e:
                logger.error(f"Error creating repository: {str(e)}")
                raise

    async def create_issue(self, issue_data: Dict[str, Any]) -> str:
        """Create a new issue record."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO issues (
                        github_issue_id, repository_id, title, body,
                        status, labels, assignees, created_at,
                        updated_at, closed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (github_issue_id) DO UPDATE 
                    SET 
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        closed_at = EXCLUDED.closed_at
                    RETURNING id;
                """
                issue_id = await conn.fetchval(
                    query,
                    issue_data['github_issue_id'],
                    issue_data['repository_id'],
                    issue_data['title'],
                    issue_data['body'],
                    issue_data['status'],
                    issue_data.get('labels', []),
                    issue_data.get('assignees', {}),
                    issue_data.get('created_at', datetime.utcnow()),
                    issue_data.get('updated_at', datetime.utcnow()),
                    issue_data.get('closed_at')
                )
                return str(issue_id)
            except Exception as e:
                logger.error(f"Error creating issue: {str(e)}")
                raise

    async def create_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Create a new analysis record."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO analysis (
                        issue_id, user_id, analysis_text,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id;
                """
                analysis_id = await conn.fetchval(
                    query,
                    analysis_data['issue_id'],
                    analysis_data['user_id'],
                    analysis_data['analysis_text'],
                    analysis_data.get('created_at', datetime.utcnow()),
                    analysis_data.get('updated_at', datetime.utcnow())
                )
                return str(analysis_id)
            except Exception as e:
                logger.error(f"Error creating analysis: {str(e)}")
                raise

    async def get_repository_by_full_name(self, full_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a repository by its full name."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT * FROM repositories 
                    WHERE full_name = $1;
                """
                row = await conn.fetchrow(query, full_name)
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Error fetching repository: {str(e)}")
                raise

    async def get_last_issue_timestamp(self, repo_id: str) -> Optional[str]:
        """Get the timestamp of the last fetched issue for a repository."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT MAX(updated_at) as last_updated
                    FROM issues
                    WHERE repository_id = $1;
                """
                result = await conn.fetchval(query, repo_id)
                return result.isoformat() if result else None
            except Exception as e:
                logger.error(f"Error fetching last issue timestamp: {str(e)}")
                raise

    async def get_issues_for_repository(
        self,
        repo_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve issues for a specific repository."""
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT * FROM issues 
                    WHERE repository_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3;
                """
                rows = await conn.fetch(query, repo_id, limit, offset)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error fetching issues: {str(e)}")
                raise
