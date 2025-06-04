"""
Retry utilities for handling transient failures.
"""

import asyncio
import functools
import random
from typing import Callable, TypeVar, Optional, Tuple, Type, Union
import logging

from ..core.exceptions import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (starts from 0)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Add random jitter to prevent thundering herd
        
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add random jitter between 0 and 25% of delay
        delay = delay * (1 + random.random() * 0.25)
    
    return delay


def retry_async(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """Decorator for retrying async functions.
    
    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exceptions to retry on
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        on_retry: Optional callback on retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Check if it's a rate limit error with retry_after
                    if isinstance(e, RateLimitError) and e.details.get('retry_after'):
                        delay = e.details['retry_after']
                    else:
                        delay = exponential_backoff(
                            attempt,
                            base_delay=base_delay,
                            max_delay=max_delay
                        )
                    
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.1f}s delay. Error: {str(e)}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            
            # Re-raise the last exception
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """Initialize retry context.
        
        Args:
            max_attempts: Maximum number of attempts
            exceptions: Exceptions to retry on
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
        """
        self.max_attempts = max_attempts
        self.exceptions = exceptions
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempt = 0
        
    async def __aenter__(self):
        """Enter async context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context with retry logic."""
        if exc_type is None:
            return False
        
        if not issubclass(exc_type, self.exceptions):
            return False
        
        self.attempt += 1
        
        if self.attempt >= self.max_attempts:
            logger.error(f"Max retry attempts ({self.max_attempts}) exceeded")
            return False
        
        delay = exponential_backoff(
            self.attempt - 1,
            base_delay=self.base_delay,
            max_delay=self.max_delay
        )
        
        logger.warning(
            f"Retrying after {delay:.1f}s (attempt {self.attempt}/{self.max_attempts})"
        )
        
        await asyncio.sleep(delay)
        return True


async def retry_with_backoff(
    coro_func: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> T:
    """Execute an async function with retry and backoff.
    
    Args:
        coro_func: Async function to execute
        *args: Positional arguments for the function
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to retry on
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await coro_func(*args, **kwargs)
            
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = exponential_backoff(
                    attempt,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    
    if last_exception:
        raise last_exception


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half_open
    
    async def call(self, coro_func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            coro_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self._state == "open":
            if (self._last_failure_time and 
                asyncio.get_event_loop().time() - self._last_failure_time > self.recovery_timeout):
                self._state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self._failure_count = 0
        self._state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = asyncio.get_event_loop().time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.error(
                f"Circuit breaker opened after {self._failure_count} failures"
            )