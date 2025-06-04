"""Monitoring and observability infrastructure."""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
import threading
from enum import Enum

from pipeline.foundation.logging import get_logger, LogContext
from pipeline.foundation.types import PipelineStage


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricSample:
    """Single metric sample with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class TimerContext:
    """Context for timing operations."""
    start_time: float
    name: str
    tags: Dict[str, str]
    collector: 'MetricsCollector'
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record_timer(self.name, duration, tags=self.tags)


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self.logger = get_logger(__name__, LogContext(component="MetricsCollector"))
        
        # Performance tracking
        self.start_time = time.time()
        self.collection_count = 0
    
    def _record_sample(self, sample: MetricSample) -> None:
        """Record a metric sample."""
        with self._lock:
            self.samples.append(sample)
            self.collection_count += 1
            
            # Update aggregated metrics
            key = f"{sample.name}:{':'.join(f'{k}={v}' for k, v in sorted(sample.tags.items()))}"
            
            if sample.metric_type == MetricType.COUNTER:
                self.counters[key] += sample.value
            elif sample.metric_type == MetricType.GAUGE:
                self.gauges[key] = sample.value
            elif sample.metric_type == MetricType.HISTOGRAM:
                self.histograms[key].append(sample.value)
                # Keep histogram size manageable
                if len(self.histograms[key]) > 1000:
                    self.histograms[key] = self.histograms[key][-1000:]
    
    def counter(self, name: str, value: Union[int, float] = 1, tags: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        tags = tags or {}
        sample = MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags,
            unit="count"
        )
        self._record_sample(sample)
    
    def gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        tags = tags or {}
        sample = MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags
        )
        self._record_sample(sample)
    
    def histogram(self, name: str, value: Union[int, float], tags: Dict[str, str] = None) -> None:
        """Record a histogram metric."""
        tags = tags or {}
        sample = MetricSample(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags
        )
        self._record_sample(sample)
    
    def timer(self, name: str, tags: Dict[str, str] = None) -> TimerContext:
        """Create a timer context manager."""
        tags = tags or {}
        return TimerContext(
            start_time=time.time(),
            name=name,
            tags=tags,
            collector=self
        )
    
    def record_timer(self, name: str, duration_seconds: float, tags: Dict[str, str] = None) -> None:
        """Record a timer metric."""
        tags = tags or {}
        sample = MetricSample(
            name=name,
            value=duration_seconds,
            metric_type=MetricType.TIMER,
            tags=tags,
            unit="seconds"
        )
        self._record_sample(sample)
    
    @contextmanager
    def timed_operation(self, operation_name: str, **tags):
        """Context manager for timing operations with automatic logging."""
        start_time = time.time()
        
        try:
            yield
            duration = time.time() - start_time
            self.record_timer(f"operation.{operation_name}.duration", duration, tags=tags)
            self.counter(f"operation.{operation_name}.success", tags=tags)
            
        except Exception as e:
            duration = time.time() - start_time
            self.record_timer(f"operation.{operation_name}.duration", duration, tags=tags)
            self.counter(f"operation.{operation_name}.error", tags={**tags, "error_type": type(e).__name__})
            raise
    
    def get_counter_value(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get current counter value."""
        tags = tags or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
        with self._lock:
            return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value."""
        tags = tags or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
        with self._lock:
            return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        tags = tags or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
        
        with self._lock:
            values = self.histograms.get(key, [])
            
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "median": sorted_values[count // 2],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_recent_samples(self, since: datetime = None, limit: int = 100) -> List[MetricSample]:
        """Get recent metric samples."""
        since = since or (datetime.utcnow() - timedelta(minutes=5))
        
        with self._lock:
            samples = list(self.samples)
        
        # Filter by timestamp
        recent = [s for s in samples if s.timestamp >= since]
        
        # Sort by timestamp and limit
        recent.sort(key=lambda s: s.timestamp, reverse=True)
        return recent[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                "uptime_seconds": uptime,
                "total_samples": self.collection_count,
                "samples_in_memory": len(self.samples),
                "active_counters": len(self.counters),
                "active_gauges": len(self.gauges),
                "active_histograms": len(self.histograms),
                "collection_rate": self.collection_count / uptime if uptime > 0 else 0
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self.counters.items():
                name, tags_str = key.split(':', 1) if ':' in key else (key, '')
                tags_formatted = '{' + tags_str.replace('=', '="').replace(':', '", ') + '"}'
                lines.append(f'# TYPE {name} counter')
                lines.append(f'{name}{tags_formatted} {value}')
            
            # Export gauges
            for key, value in self.gauges.items():
                name, tags_str = key.split(':', 1) if ':' in key else (key, '')
                tags_formatted = '{' + tags_str.replace('=', '="').replace(':', '", ') + '"}'
                lines.append(f'# TYPE {name} gauge')
                lines.append(f'{name}{tags_formatted} {value}')
        
        return '\n'.join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.samples.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.collection_count = 0
            self.start_time = time.time()
        
        self.logger.info("Metrics collector reset")


class PipelineMonitor:
    """High-level monitoring for pipeline operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger(__name__, LogContext(component="PipelineMonitor"))
        self.active_operations: Dict[str, datetime] = {}
    
    def track_pipeline_stage(self, stage: PipelineStage, operation: str):
        """Decorator for tracking pipeline stage operations."""
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_tracked_operation(
                        func, stage, operation, args, kwargs
                    )
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    return self._execute_tracked_operation_sync(
                        func, stage, operation, args, kwargs
                    )
                return sync_wrapper
        return decorator
    
    async def _execute_tracked_operation(self, func, stage, operation, args, kwargs):
        """Execute tracked async operation."""
        operation_id = f"{stage.value}.{operation}"
        tags = {"stage": stage.value, "operation": operation}
        
        self.active_operations[operation_id] = datetime.utcnow()
        self.metrics.gauge("pipeline.active_operations", len(self.active_operations))
        
        try:
            with self.metrics.timed_operation(operation_id, **tags):
                result = await func(*args, **kwargs)
                
                self.metrics.counter("pipeline.operation.success", tags=tags)
                return result
                
        except Exception as e:
            self.metrics.counter("pipeline.operation.error", 
                               tags={**tags, "error_type": type(e).__name__})
            raise
        finally:
            self.active_operations.pop(operation_id, None)
            self.metrics.gauge("pipeline.active_operations", len(self.active_operations))
    
    def _execute_tracked_operation_sync(self, func, stage, operation, args, kwargs):
        """Execute tracked sync operation."""
        operation_id = f"{stage.value}.{operation}"
        tags = {"stage": stage.value, "operation": operation}
        
        self.active_operations[operation_id] = datetime.utcnow()
        self.metrics.gauge("pipeline.active_operations", len(self.active_operations))
        
        try:
            with self.metrics.timed_operation(operation_id, **tags):
                result = func(*args, **kwargs)
                
                self.metrics.counter("pipeline.operation.success", tags=tags)
                return result
                
        except Exception as e:
            self.metrics.counter("pipeline.operation.error", 
                               tags={**tags, "error_type": type(e).__name__})
            raise
        finally:
            self.active_operations.pop(operation_id, None)
            self.metrics.gauge("pipeline.active_operations", len(self.active_operations))
    
    def record_data_quality(self, stage: PipelineStage, quality_score: float, **tags):
        """Record data quality metrics."""
        self.metrics.histogram(
            "pipeline.data_quality",
            quality_score,
            tags={"stage": stage.value, **tags}
        )
    
    def record_throughput(self, stage: PipelineStage, items_processed: int, **tags):
        """Record throughput metrics."""
        self.metrics.counter(
            "pipeline.items_processed",
            items_processed,
            tags={"stage": stage.value, **tags}
        )
    
    def record_error_rate(self, stage: PipelineStage, error_count: int, total_count: int, **tags):
        """Record error rate metrics."""
        if total_count > 0:
            error_rate = error_count / total_count
            self.metrics.gauge(
                "pipeline.error_rate",
                error_rate,
                tags={"stage": stage.value, **tags}
            )
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        now = datetime.utcnow()
        stale_threshold = timedelta(minutes=10)
        
        # Check for stale operations
        stale_operations = [
            op_id for op_id, start_time in self.active_operations.items()
            if now - start_time > stale_threshold
        ]
        
        # Get recent error rates
        recent_errors = self.metrics.get_recent_samples(
            since=now - timedelta(minutes=5)
        )
        error_samples = [s for s in recent_errors if "error" in s.name]
        
        return {
            "status": "healthy" if not stale_operations and len(error_samples) < 10 else "degraded",
            "active_operations": len(self.active_operations),
            "stale_operations": stale_operations,
            "recent_errors": len(error_samples),
            "uptime": self.metrics.get_summary()["uptime_seconds"]
        }


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None
_pipeline_monitor: Optional[PipelineMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_pipeline_monitor() -> PipelineMonitor:
    """Get the global pipeline monitor."""
    global _pipeline_monitor
    if _pipeline_monitor is None:
        _pipeline_monitor = PipelineMonitor(get_metrics_collector())
    return _pipeline_monitor


def initialize_monitoring() -> tuple[MetricsCollector, PipelineMonitor]:
    """Initialize the global monitoring system."""
    global _metrics_collector, _pipeline_monitor
    
    _metrics_collector = MetricsCollector()
    _pipeline_monitor = PipelineMonitor(_metrics_collector)
    
    return _metrics_collector, _pipeline_monitor