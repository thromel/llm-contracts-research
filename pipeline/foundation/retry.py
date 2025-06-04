"""Enhanced retry mechanisms and error handling."""

import asyncio
import functools
import random
import time
from typing import Type, Union, Callable, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

from .logging import get_logger, LogContext, PipelineStage


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    RANDOM = "random"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    jitter: bool = True
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_factor ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.RANDOM:
            delay = random.uniform(self.base_delay, self.max_delay)
        else:
            delay = self.base_delay
        
        # Apply jitter
        if self.jitter and self.strategy != RetryStrategy.RANDOM:
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return min(delay, self.max_delay)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Operation failed after {attempts} attempts. Last error: {last_exception}")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = get_logger(__name__, LogContext(component="CircuitBreaker"))
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker closed after successful operation")
        
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """Decorator for adding retry behavior to functions."""
    
    def decorator(func: Callable) -> Callable:
        retry_config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            backoff_factor=backoff_factor,
            jitter=jitter,
            exceptions=exceptions
        )
        
        if asyncio.iscoroutinefunction(func):
            return _async_retry_wrapper(func, retry_config, circuit_breaker)
        else:
            return _sync_retry_wrapper(func, retry_config, circuit_breaker)
    
    return decorator


def _sync_retry_wrapper(
    func: Callable,
    retry_config: RetryConfig,
    circuit_breaker: Optional[CircuitBreaker]
) -> Callable:
    """Synchronous retry wrapper."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__, LogContext(
            component=func.__name__,
            operation="retry_operation"
        ))
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except retry_config.exceptions as e:
                last_exception = e
                
                if attempt == retry_config.max_attempts - 1:
                    logger.error(
                        f"Final retry attempt failed for {func.__name__}",
                        attempt=attempt + 1,
                        max_attempts=retry_config.max_attempts,
                        error=str(e)
                    )
                    break
                
                delay = retry_config.calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s",
                    attempt=attempt + 1,
                    max_attempts=retry_config.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                time.sleep(delay)
        
        raise RetryError(retry_config.max_attempts, last_exception)
    
    return wrapper


def _async_retry_wrapper(
    func: Callable,
    retry_config: RetryConfig,
    circuit_breaker: Optional[CircuitBreaker]
) -> Callable:
    """Asynchronous retry wrapper."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__, LogContext(
            component=func.__name__,
            operation="async_retry_operation"
        ))
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                if circuit_breaker:
                    return await circuit_breaker.acall(func, *args, **kwargs)
                else:
                    return await func(*args, **kwargs)
                    
            except retry_config.exceptions as e:
                last_exception = e
                
                if attempt == retry_config.max_attempts - 1:
                    logger.error(
                        f"Final async retry attempt failed for {func.__name__}",
                        attempt=attempt + 1,
                        max_attempts=retry_config.max_attempts,
                        error=str(e)
                    )
                    break
                
                delay = retry_config.calculate_delay(attempt)
                logger.warning(
                    f"Async retry attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s",
                    attempt=attempt + 1,
                    max_attempts=retry_config.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
        
        raise RetryError(retry_config.max_attempts, last_exception)
    
    return wrapper


class RetryableOperation:
    """Class-based retryable operation for complex scenarios."""
    
    def __init__(self, retry_config: RetryConfig, circuit_breaker: Optional[CircuitBreaker] = None):
        self.retry_config = retry_config
        self.circuit_breaker = circuit_breaker
        self.logger = get_logger(__name__, LogContext(component="RetryableOperation"))
    
    def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        return self._execute_with_retry(operation, args, kwargs, is_async=False)
    
    async def aexecute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute async operation with retry logic."""
        return await self._execute_with_retry(operation, args, kwargs, is_async=True)
    
    def _execute_with_retry(self, operation: Callable, args: tuple, kwargs: dict, is_async: bool) -> Any:
        """Internal retry execution logic."""
        if is_async:
            return self._async_execute_with_retry(operation, args, kwargs)
        else:
            return self._sync_execute_with_retry(operation, args, kwargs)
    
    def _sync_execute_with_retry(self, operation: Callable, args: tuple, kwargs: dict) -> Any:
        """Synchronous retry execution."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(operation, *args, **kwargs)
                else:
                    return operation(*args, **kwargs)
                    
            except self.retry_config.exceptions as e:
                last_exception = e
                
                if attempt == self.retry_config.max_attempts - 1:
                    self.logger.error(
                        "Final retry attempt failed",
                        attempt=attempt + 1,
                        max_attempts=self.retry_config.max_attempts,
                        error=str(e)
                    )
                    break
                
                delay = self.retry_config.calculate_delay(attempt)
                self.logger.warning(
                    "Retry attempt failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self.retry_config.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                time.sleep(delay)
        
        raise RetryError(self.retry_config.max_attempts, last_exception)
    
    async def _async_execute_with_retry(self, operation: Callable, args: tuple, kwargs: dict) -> Any:
        """Asynchronous retry execution."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                if self.circuit_breaker:
                    return await self.circuit_breaker.acall(operation, *args, **kwargs)
                else:
                    return await operation(*args, **kwargs)
                    
            except self.retry_config.exceptions as e:
                last_exception = e
                
                if attempt == self.retry_config.max_attempts - 1:
                    self.logger.error(
                        "Final async retry attempt failed",
                        attempt=attempt + 1,
                        max_attempts=self.retry_config.max_attempts,
                        error=str(e)
                    )
                    break
                
                delay = self.retry_config.calculate_delay(attempt)
                self.logger.warning(
                    "Async retry attempt failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self.retry_config.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
        
        raise RetryError(self.retry_config.max_attempts, last_exception)


# Common retry configurations
DEFAULT_RETRY = RetryConfig()

API_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    strategy=RetryStrategy.EXPONENTIAL,
    backoff_factor=2.0,
    jitter=True
)

DATABASE_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
    backoff_factor=1.5,
    jitter=True
)

NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
    backoff_factor=2.0,
    jitter=True
)