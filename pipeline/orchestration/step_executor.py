"""
Pipeline Step Executor.

Handles execution of individual pipeline steps with enhanced error handling,
monitoring, and retry logic using the foundation architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from ..foundation.config import ConfigManager
from ..foundation.logging import PipelineLogger
from ..foundation.retry import with_retry, CircuitBreaker
from ..infrastructure.database import DatabaseManager
from ..infrastructure.monitoring import MetricsCollector
from ..domain.models import PipelineStage


class StepStatus(Enum):
    """Pipeline step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of a pipeline step execution."""
    status: StepStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    posts_processed: int = 0
    posts_passed: int = 0
    posts_failed: int = 0
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        
        # Calculate duration if end_time is set
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""
    name: str
    stage: PipelineStage
    max_posts: int = 1000
    timeout_seconds: int = 3600  # 1 hour default
    retry_attempts: int = 3
    retry_delay: float = 1.0
    skip_on_failure: bool = False
    dependencies: List[str] = None
    validation_required: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PipelineStepExecutor:
    """
    Executes individual pipeline steps with comprehensive monitoring and error handling.
    
    Features:
    - Async step execution with timeout handling
    - Retry logic with exponential backoff
    - Circuit breaker pattern for failing steps
    - Comprehensive metrics collection
    - Dependency validation
    - Progress tracking and reporting
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: PipelineLogger,
        db_manager: DatabaseManager,
        metrics: MetricsCollector
    ):
        """Initialize the step executor.
        
        Args:
            config: Configuration manager
            logger: Pipeline logger
            db_manager: Database manager
            metrics: Metrics collector
        """
        self.config = config
        self.logger = logger
        self.db = db_manager
        self.metrics = metrics
        
        # Step execution state
        self._step_results: Dict[str, StepResult] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._step_registry: Dict[str, Callable] = {}
        
        self.logger.info("PipelineStepExecutor initialized")

    def register_step(
        self,
        step_config: StepConfig,
        execution_func: Callable
    ) -> None:
        """Register a pipeline step for execution.
        
        Args:
            step_config: Step configuration
            execution_func: Async function to execute the step
        """
        self._step_registry[step_config.name] = {
            "config": step_config,
            "func": execution_func
        }
        
        # Initialize circuit breaker for this step
        self._circuit_breakers[step_config.name] = CircuitBreaker(
            failure_threshold=step_config.retry_attempts,
            recovery_timeout=60
        )
        
        self.logger.info(f"Registered pipeline step: {step_config.name}", extra={
            "stage": step_config.stage.value,
            "max_posts": step_config.max_posts,
            "retry_attempts": step_config.retry_attempts
        })

    async def execute_step(
        self,
        step_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StepResult:
        """Execute a registered pipeline step.
        
        Args:
            step_name: Name of the step to execute
            context: Additional context for step execution
            
        Returns:
            Step execution result
            
        Raises:
            ValueError: If step is not registered
            RuntimeError: If step execution fails after all retries
        """
        if step_name not in self._step_registry:
            raise ValueError(f"Step '{step_name}' not registered")
        
        step_info = self._step_registry[step_name]
        step_config = step_info["config"]
        execution_func = step_info["func"]
        
        # Initialize step result
        result = StepResult(
            status=StepStatus.PENDING,
            start_time=datetime.utcnow()
        )
        self._step_results[step_name] = result
        
        self.logger.info(f"Starting execution of step: {step_name}", extra={
            "stage": step_config.stage.value,
            "max_posts": step_config.max_posts
        })
        
        try:
            # Validate dependencies
            await self._validate_dependencies(step_config)
            
            # Execute step with circuit breaker and retry logic
            result.status = StepStatus.RUNNING
            await self._record_step_progress(step_name, result)
            
            with self.metrics.timer(f"step_{step_name}"):
                circuit_breaker = self._circuit_breakers[step_name]
                
                # Execute with circuit breaker
                if circuit_breaker.is_open():
                    raise RuntimeError(f"Circuit breaker open for step {step_name}")
                
                step_result = await self._execute_with_retry(
                    step_config,
                    execution_func,
                    context or {}
                )
                
                # Update result with step output
                result.status = StepStatus.COMPLETED
                result.end_time = datetime.utcnow()
                result.posts_processed = step_result.get("posts_processed", 0)
                result.posts_passed = step_result.get("posts_passed", 0)
                result.posts_failed = step_result.get("posts_failed", 0)
                result.metrics.update(step_result.get("metrics", {}))
                
                # Record success with circuit breaker
                circuit_breaker.record_success()
                
                self.logger.info(f"Step {step_name} completed successfully", extra={
                    "duration": result.duration_seconds,
                    "posts_processed": result.posts_processed
                })
                
                # Record final progress
                await self._record_step_progress(step_name, result)
                
                return result
                
        except Exception as e:
            # Record failure
            result.status = StepStatus.FAILED
            result.end_time = datetime.utcnow()
            result.error = str(e)
            
            # Record failure with circuit breaker
            circuit_breaker = self._circuit_breakers[step_name]
            circuit_breaker.record_failure()
            
            self.logger.error(f"Step {step_name} failed", extra={
                "error": str(e),
                "duration": result.duration_seconds,
                "retry_count": result.retry_count
            })
            
            # Record final progress
            await self._record_step_progress(step_name, result)
            
            # Skip or re-raise based on configuration
            if step_config.skip_on_failure:
                result.status = StepStatus.SKIPPED
                self.logger.warning(f"Skipping failed step: {step_name}")
                return result
            else:
                raise RuntimeError(f"Step {step_name} failed: {str(e)}")

    async def _validate_dependencies(self, step_config: StepConfig) -> None:
        """Validate that all step dependencies have completed successfully.
        
        Args:
            step_config: Configuration for the step
            
        Raises:
            RuntimeError: If dependencies are not met
        """
        for dependency in step_config.dependencies:
            if dependency not in self._step_results:
                raise RuntimeError(f"Dependency '{dependency}' has not been executed")
            
            dep_result = self._step_results[dependency]
            if dep_result.status != StepStatus.COMPLETED:
                raise RuntimeError(f"Dependency '{dependency}' did not complete successfully (status: {dep_result.status.value})")
        
        if step_config.dependencies:
            self.logger.info(f"Dependencies validated for step {step_config.name}", extra={
                "dependencies": step_config.dependencies
            })

    async def _execute_with_retry(
        self,
        step_config: StepConfig,
        execution_func: Callable,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute step with retry logic and timeout handling.
        
        Args:
            step_config: Step configuration
            execution_func: Function to execute
            context: Execution context
            
        Returns:
            Step execution result
        """
        last_exception = None
        
        for attempt in range(step_config.retry_attempts):
            try:
                # Update retry count in result
                result = self._step_results[step_config.name]
                result.retry_count = attempt
                
                if attempt > 0:
                    result.status = StepStatus.RETRYING
                    await self._record_step_progress(step_config.name, result)
                    
                    # Exponential backoff delay
                    delay = step_config.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying step {step_config.name} (attempt {attempt + 1}), waiting {delay}s")
                    await asyncio.sleep(delay)
                
                # Execute with timeout
                step_result = await asyncio.wait_for(
                    execution_func(step_config, context),
                    timeout=step_config.timeout_seconds
                )
                
                return step_result
                
            except asyncio.TimeoutError:
                last_exception = RuntimeError(f"Step {step_config.name} timed out after {step_config.timeout_seconds}s")
                self.logger.warning(f"Step {step_config.name} timed out (attempt {attempt + 1})")
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Step {step_config.name} failed (attempt {attempt + 1})", extra={
                    "error": str(e)
                })
        
        # All retries exhausted
        raise last_exception

    async def _record_step_progress(self, step_name: str, result: StepResult) -> None:
        """Record step progress to database and metrics.
        
        Args:
            step_name: Name of the step
            result: Current step result
        """
        try:
            # Update metrics
            self.metrics.increment_counter(f"step_{step_name}_status_{result.status.value}")
            
            if result.duration_seconds:
                self.metrics.record_histogram(f"step_{step_name}_duration", result.duration_seconds)
            
            if result.posts_processed > 0:
                self.metrics.record_gauge(f"step_{step_name}_posts_processed", result.posts_processed)
            
            # Record to database
            progress_record = {
                "step_name": step_name,
                "status": result.status.value,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration_seconds": result.duration_seconds,
                "posts_processed": result.posts_processed,
                "posts_passed": result.posts_passed,
                "posts_failed": result.posts_failed,
                "retry_count": result.retry_count,
                "error": result.error,
                "metrics": result.metrics,
                "recorded_at": datetime.utcnow()
            }
            
            await self.db.insert_one("step_progress", progress_record)
            
        except Exception as e:
            self.logger.warning(f"Failed to record progress for step {step_name}", extra={
                "error": str(e)
            })

    async def execute_steps_parallel(
        self,
        step_names: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, StepResult]:
        """Execute multiple steps in parallel.
        
        Args:
            step_names: List of step names to execute
            context: Shared execution context
            
        Returns:
            Dictionary mapping step names to their results
        """
        self.logger.info(f"Executing {len(step_names)} steps in parallel", extra={
            "steps": step_names
        })
        
        # Create execution tasks
        tasks = []
        for step_name in step_names:
            task = asyncio.create_task(
                self.execute_step(step_name, context),
                name=f"step_{step_name}"
            )
            tasks.append((step_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for step_name, task in tasks:
            try:
                result = await task
                results[step_name] = result
            except Exception as e:
                # Create failed result
                results[step_name] = StepResult(
                    status=StepStatus.FAILED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    error=str(e)
                )
        
        return results

    async def execute_steps_sequential(
        self,
        step_names: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, StepResult]:
        """Execute multiple steps sequentially.
        
        Args:
            step_names: List of step names to execute in order
            context: Shared execution context
            
        Returns:
            Dictionary mapping step names to their results
        """
        self.logger.info(f"Executing {len(step_names)} steps sequentially", extra={
            "steps": step_names
        })
        
        results = {}
        shared_context = context or {}
        
        for step_name in step_names:
            try:
                result = await self.execute_step(step_name, shared_context)
                results[step_name] = result
                
                # Update shared context with step results
                shared_context[f"{step_name}_result"] = {
                    "posts_processed": result.posts_processed,
                    "posts_passed": result.posts_passed,
                    "metrics": result.metrics
                }
                
            except Exception as e:
                # Create failed result
                results[step_name] = StepResult(
                    status=StepStatus.FAILED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    error=str(e)
                )
                
                # Stop execution on failure unless configured to skip
                step_info = self._step_registry.get(step_name)
                if step_info and not step_info["config"].skip_on_failure:
                    break
        
        return results

    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get the result of a previously executed step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step result if available, None otherwise
        """
        return self._step_results.get(step_name)

    def get_all_results(self) -> Dict[str, StepResult]:
        """Get all step execution results.
        
        Returns:
            Dictionary mapping step names to their results
        """
        return self._step_results.copy()

    async def get_step_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all step statuses and metrics.
        
        Returns:
            Summary of step execution status
        """
        summary = {
            "total_steps": len(self._step_results),
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "skipped": 0,
            "total_posts_processed": 0,
            "total_duration": 0.0,
            "step_details": {}
        }
        
        for step_name, result in self._step_results.items():
            # Count by status
            if result.status == StepStatus.COMPLETED:
                summary["completed"] += 1
            elif result.status == StepStatus.FAILED:
                summary["failed"] += 1
            elif result.status == StepStatus.RUNNING:
                summary["running"] += 1
            elif result.status == StepStatus.PENDING:
                summary["pending"] += 1
            elif result.status == StepStatus.SKIPPED:
                summary["skipped"] += 1
            
            # Accumulate metrics
            summary["total_posts_processed"] += result.posts_processed
            if result.duration_seconds:
                summary["total_duration"] += result.duration_seconds
            
            # Step details
            summary["step_details"][step_name] = {
                "status": result.status.value,
                "posts_processed": result.posts_processed,
                "duration_seconds": result.duration_seconds,
                "retry_count": result.retry_count,
                "error": result.error
            }
        
        return summary

    def reset_step_results(self) -> None:
        """Reset all step execution results."""
        self._step_results.clear()
        self.logger.info("Step execution results reset")

    async def cleanup(self) -> None:
        """Cleanup step executor resources."""
        self.logger.info("Cleaning up step executor")
        
        # Reset circuit breakers
        for breaker in self._circuit_breakers.values():
            breaker.reset()
        
        self.logger.info("Step executor cleanup completed")