"""
Custom exceptions for the LLM Contracts Research Pipeline.

These exceptions provide clear error messages and proper error handling
throughout the pipeline execution.
"""

from typing import Optional, Dict, Any


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline error.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataAcquisitionError(PipelineError):
    """Error during data acquisition phase."""
    
    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        **kwargs
    ):
        """Initialize data acquisition error.
        
        Args:
            message: Error message
            source: Data source that failed
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="DATA_ACQUISITION_ERROR",
            details={"source": source, **kwargs}
        )


class FilteringError(PipelineError):
    """Error during filtering phase."""
    
    def __init__(
        self,
        message: str,
        filter_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize filtering error.
        
        Args:
            message: Error message
            filter_name: Name of filter that failed
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="FILTERING_ERROR",
            details={"filter_name": filter_name, **kwargs}
        )


class ScreeningError(PipelineError):
    """Error during LLM screening phase."""
    
    def __init__(
        self,
        message: str,
        screener_name: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """Initialize screening error.
        
        Args:
            message: Error message
            screener_name: Name of screener that failed
            model: LLM model that failed
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="SCREENING_ERROR",
            details={
                "screener_name": screener_name,
                "model": model,
                **kwargs
            }
        )


class StorageError(PipelineError):
    """Error during storage operations."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection: Optional[str] = None,
        **kwargs
    ):
        """Initialize storage error.
        
        Args:
            message: Error message
            operation: Storage operation that failed
            collection: Collection/table involved
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="STORAGE_ERROR",
            details={
                "operation": operation,
                "collection": collection,
                **kwargs
            }
        )


class ConfigurationError(PipelineError):
    """Error in pipeline configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused error
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key, **kwargs}
        )


class ValidationError(PipelineError):
    """Error during data validation."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": value,
                **kwargs
            }
        )


class RateLimitError(ScreeningError):
    """Error when API rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            **kwargs
        )
        self.error_code = "RATE_LIMIT_ERROR"
        self.details["retry_after"] = retry_after


class AuthenticationError(PipelineError):
    """Error during authentication."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        **kwargs
    ):
        """Initialize authentication error.
        
        Args:
            message: Error message
            service: Service that failed authentication
            **kwargs: Additional error details
        """
        super().__init__(
            message,
            error_code="AUTHENTICATION_ERROR",
            details={"service": service, **kwargs}
        )