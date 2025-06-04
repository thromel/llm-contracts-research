"""Tests for retry mechanisms and error handling."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from pipeline.foundation.retry import (
    RetryConfig, RetryStrategy, RetryError, CircuitBreakerError,
    CircuitBreaker, with_retry, RetryableOperation,
    DEFAULT_RETRY, API_RETRY, DATABASE_RETRY
)


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_fixed_delay_calculation(self):
        """Test fixed delay strategy."""
        config = RetryConfig(
            base_delay=2.0,
            strategy=RetryStrategy.FIXED,
            jitter=False
        )
        
        assert config.calculate_delay(0) == 2.0
        assert config.calculate_delay(5) == 2.0
    
    def test_exponential_delay_calculation(self):
        """Test exponential delay strategy."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0,
            jitter=False
        )
        
        assert config.calculate_delay(0) == 1.0  # 1.0 * 2^0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2^1
        assert config.calculate_delay(2) == 4.0  # 1.0 * 2^2
    
    def test_linear_delay_calculation(self):
        """Test linear delay strategy."""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False
        )
        
        assert config.calculate_delay(0) == 1.0  # 1.0 * (0 + 1)
        assert config.calculate_delay(1) == 2.0  # 1.0 * (1 + 1)
        assert config.calculate_delay(2) == 3.0  # 1.0 * (2 + 1)
    
    def test_max_delay_limit(self):
        """Test that delay doesn't exceed max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0,
            jitter=False
        )
        
        # Large attempt number should be capped at max_delay
        assert config.calculate_delay(10) == 5.0
    
    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(
            base_delay=10.0,
            strategy=RetryStrategy.FIXED,
            jitter=True
        )
        
        # Run multiple times to check for variation
        delays = [config.calculate_delay(0) for _ in range(10)]
        
        # Should have some variation (not all the same)
        assert len(set(delays)) > 1
        # All should be reasonably close to base_delay
        assert all(8.0 <= delay <= 12.0 for delay in delays)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_closed_state_success(self):
        """Test circuit breaker in closed state with successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def successful_operation():
            return "success"
        
        result = breaker.call(successful_operation)
        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
    
    def test_closed_to_open_transition(self):
        """Test transition from closed to open state."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_operation():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.state == "closed"
        assert breaker.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.state == "open"
        assert breaker.failure_count == 2
    
    def test_open_state_blocks_calls(self):
        """Test that open circuit breaker blocks calls."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        
        def failing_operation():
            raise ValueError("Test error")
        
        # Cause circuit to open
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.state == "open"
        
        # Next call should be blocked
        def any_operation():
            return "should not be called"
        
        with pytest.raises(CircuitBreakerError):
            breaker.call(any_operation)
    
    def test_half_open_recovery(self):
        """Test recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def failing_operation():
            raise ValueError("Test error")
        
        def successful_operation():
            return "success"
        
        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_operation)
        assert breaker.state == "open"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Successful call should close circuit
        result = breaker.call(successful_operation)
        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test async circuit breaker functionality."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        async def async_successful_operation():
            return "async success"
        
        async def async_failing_operation():
            raise ValueError("Async test error")
        
        # Test successful operation
        result = await breaker.acall(async_successful_operation)
        assert result == "async success"
        
        # Test failure
        with pytest.raises(ValueError):
            await breaker.acall(async_failing_operation)


class TestWithRetryDecorator:
    """Test with_retry decorator functionality."""
    
    def test_successful_operation_no_retry(self):
        """Test successful operation requires no retry."""
        call_count = 0
        
        @with_retry(max_attempts=3)
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_operation()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """Test retry on failure then success."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_operation()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhaustion(self):
        """Test that retry exhaustion raises RetryError."""
        call_count = 0
        
        @with_retry(max_attempts=2, base_delay=0.01)
        def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Failure {call_count}")
        
        with pytest.raises(RetryError) as exc_info:
            always_failing_operation()
        
        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert call_count == 2
    
    def test_specific_exception_retry(self):
        """Test retry only on specific exceptions."""
        call_count = 0
        
        @with_retry(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def operation_with_different_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise TypeError("Non-retryable error")
            return "success"
        
        with pytest.raises(TypeError):
            operation_with_different_errors()
        
        assert call_count == 2  # Should not retry TypeError
    
    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry functionality."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        async def async_flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Async temporary failure")
            return "async success"
        
        result = await async_flaky_operation()
        assert result == "async success"
        assert call_count == 2


class TestRetryableOperation:
    """Test RetryableOperation class."""
    
    def test_execute_successful(self):
        """Test successful execution."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        retryable = RetryableOperation(retry_config)
        
        def operation(x, y):
            return x + y
        
        result = retryable.execute(operation, 2, 3)
        assert result == 5
    
    def test_execute_with_retry(self):
        """Test execution with retry."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        retryable = RetryableOperation(retry_config)
        
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retryable.execute(flaky_operation)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_execute(self):
        """Test async execution."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        retryable = RetryableOperation(retry_config)
        
        call_count = 0
        
        async def async_flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Async temporary failure")
            return "async success"
        
        result = await retryable.aexecute(async_flaky_operation)
        assert result == "async success"
        assert call_count == 2
    
    def test_execute_with_circuit_breaker(self):
        """Test execution with circuit breaker."""
        retry_config = RetryConfig(max_attempts=5, base_delay=0.01)
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        retryable = RetryableOperation(retry_config, circuit_breaker)
        
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Failure")
            return "success"
        
        # Should fail and open circuit after 2 attempts
        with pytest.raises(RetryError):
            retryable.execute(operation)
        
        assert circuit_breaker.state == "open"
        assert call_count == 2


class TestPreDefinedConfigs:
    """Test predefined retry configurations."""
    
    def test_default_retry_config(self):
        """Test default retry configuration."""
        assert DEFAULT_RETRY.max_attempts == 3
        assert DEFAULT_RETRY.base_delay == 1.0
        assert DEFAULT_RETRY.strategy == RetryStrategy.EXPONENTIAL
    
    def test_api_retry_config(self):
        """Test API retry configuration."""
        assert API_RETRY.max_attempts == 5
        assert API_RETRY.max_delay == 120.0
        assert API_RETRY.jitter == True
    
    def test_database_retry_config(self):
        """Test database retry configuration."""
        assert DATABASE_RETRY.max_attempts == 3
        assert DATABASE_RETRY.backoff_factor == 1.5
        assert DATABASE_RETRY.max_delay == 30.0


if __name__ == "__main__":
    pytest.main([__file__])