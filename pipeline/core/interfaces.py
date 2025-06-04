"""
Core interfaces for the LLM Contracts Research Pipeline.

These abstract base classes define the contracts that all pipeline components
must implement, ensuring consistency and testability.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, List, Optional, Protocol
from datetime import datetime

from ..common.models import RawPost, FilteredPost, LLMScreeningResult


class DataSource(ABC):
    """Abstract interface for data acquisition sources."""
    
    @abstractmethod
    async def acquire(
        self, 
        since: datetime,
        max_items: int = 1000,
        **kwargs
    ) -> AsyncIterator[RawPost]:
        """Acquire raw posts from the data source.
        
        Args:
            since: Fetch posts created after this date
            max_items: Maximum number of items to fetch
            **kwargs: Source-specific parameters
            
        Yields:
            RawPost instances
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the data source is accessible."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass


class Filter(ABC):
    """Abstract interface for post filtering."""
    
    @abstractmethod
    def apply(self, post: RawPost) -> FilteredPost:
        """Apply filtering logic to a raw post.
        
        Args:
            post: Raw post to filter
            
        Returns:
            FilteredPost with filtering results
        """
        pass
    
    @abstractmethod
    async def apply_batch(
        self, 
        posts: List[RawPost],
        parallel: bool = True
    ) -> List[FilteredPost]:
        """Apply filtering to a batch of posts.
        
        Args:
            posts: List of raw posts to filter
            parallel: Whether to process in parallel
            
        Returns:
            List of filtered posts
        """
        pass
    
    @property
    @abstractmethod
    def filter_name(self) -> str:
        """Return the name of this filter."""
        pass


class Screener(ABC):
    """Abstract interface for LLM screening."""
    
    @abstractmethod
    async def screen(
        self, 
        post: FilteredPost,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMScreeningResult:
        """Screen a single post using LLM.
        
        Args:
            post: Filtered post to screen
            context: Additional context for screening
            
        Returns:
            Screening result
        """
        pass
    
    @abstractmethod
    async def screen_batch(
        self,
        posts: List[FilteredPost],
        batch_size: int = 10
    ) -> List[LLMScreeningResult]:
        """Screen a batch of posts.
        
        Args:
            posts: List of filtered posts to screen
            batch_size: Size of batches for API calls
            
        Returns:
            List of screening results
        """
        pass
    
    @property
    @abstractmethod
    def screener_name(self) -> str:
        """Return the name of this screener."""
        pass


class Storage(ABC):
    """Abstract interface for data storage."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to storage backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage backend."""
        pass
    
    @abstractmethod
    async def save(
        self, 
        collection: str,
        document: Dict[str, Any],
        upsert: bool = False
    ) -> str:
        """Save a document to storage.
        
        Args:
            collection: Collection/table name
            document: Document to save
            upsert: Whether to update if exists
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    async def find(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Find documents matching query.
        
        Args:
            collection: Collection/table name
            query: Query parameters
            limit: Maximum results
            
        Yields:
            Matching documents
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """Update documents matching query.
        
        Args:
            collection: Collection/table name
            query: Query to match documents
            update: Update operations
            
        Returns:
            Number of documents updated
        """
        pass
    
    @abstractmethod
    async def count(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count documents in collection.
        
        Args:
            collection: Collection/table name
            query: Optional query filter
            
        Returns:
            Document count
        """
        pass


class PipelineStep(ABC):
    """Abstract interface for pipeline steps."""
    
    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the pipeline step.
        
        Args:
            input_data: Input data for this step
            context: Execution context
            
        Returns:
            Output data for next step
        """
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data before execution.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def step_name(self) -> str:
        """Return the name of this step."""
        pass
    
    @property
    @abstractmethod
    def requires(self) -> List[str]:
        """Return list of required previous steps."""
        pass


class Pipeline(ABC):
    """Abstract interface for the main pipeline."""
    
    @abstractmethod
    async def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline.
        
        Args:
            step: Pipeline step to add
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        start_from: Optional[str] = None,
        stop_at: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Args:
            start_from: Step to start from
            stop_at: Step to stop at
            context: Execution context
            
        Returns:
            Execution results
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Status information
        """
        pass
    
    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Return the name of this pipeline."""
        pass


class ConfigProvider(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        pass
    
    @abstractmethod
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from file.
        
        Args:
            filepath: Path to configuration file
        """
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        ...
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        ...