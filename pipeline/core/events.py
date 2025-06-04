"""
Event system for the LLM Contracts Research Pipeline.

Provides a publish-subscribe event bus for decoupled communication
between pipeline components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Set
from enum import Enum
import asyncio
import logging
from weakref import WeakSet

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types in the pipeline."""
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    
    # Step events
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    
    # Data events
    DATA_ACQUIRED = "data_acquired"
    DATA_FILTERED = "data_filtered"
    DATA_SCREENED = "data_screened"
    DATA_LABELED = "data_labeled"
    
    # Storage events
    DATA_SAVED = "data_saved"
    DATA_UPDATED = "data_updated"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERED = "error_recovered"
    
    # Metrics events
    METRIC_RECORDED = "metric_recorded"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class Event(ABC):
    """Base class for all events."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class PipelineEvent(Event):
    """Event for pipeline lifecycle."""
    pipeline_name: str = ""
    pipeline_id: str = ""
    status: str = ""
    
    def __post_init__(self):
        """Set appropriate event type based on status."""
        if self.status == "started":
            self.event_type = EventType.PIPELINE_STARTED
        elif self.status == "completed":
            self.event_type = EventType.PIPELINE_COMPLETED
        elif self.status == "failed":
            self.event_type = EventType.PIPELINE_FAILED


@dataclass
class StepEvent(Event):
    """Event for pipeline step execution."""
    step_name: str = ""
    step_status: str = ""
    duration_ms: Optional[float] = None
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    
    def __post_init__(self):
        """Set appropriate event type based on status."""
        if self.step_status == "started":
            self.event_type = EventType.STEP_STARTED
        elif self.step_status == "completed":
            self.event_type = EventType.STEP_COMPLETED
        elif self.step_status == "failed":
            self.event_type = EventType.STEP_FAILED


@dataclass
class DataProcessedEvent(Event):
    """Event for data processing stages."""
    stage: str = ""  # acquisition, filtering, screening, labeling
    items_processed: int = 0
    items_passed: int = 0
    items_failed: int = 0
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Set appropriate event type based on stage."""
        stage_to_event = {
            "acquisition": EventType.DATA_ACQUIRED,
            "filtering": EventType.DATA_FILTERED,
            "screening": EventType.DATA_SCREENED,
            "labeling": EventType.DATA_LABELED
        }
        self.event_type = stage_to_event.get(self.stage, EventType.DATA_ACQUIRED)


@dataclass
class ErrorEvent(Event):
    """Event for errors and recovery."""
    error_type: str = ""
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    is_recoverable: bool = False
    recovery_action: Optional[str] = None
    
    def __post_init__(self):
        """Set event type to ERROR_OCCURRED."""
        self.event_type = EventType.ERROR_OCCURRED


@dataclass
class MetricEvent(Event):
    """Event for metrics and monitoring."""
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    
    def __post_init__(self):
        """Set appropriate event type."""
        if self.threshold and self.metric_value > self.threshold:
            self.event_type = EventType.THRESHOLD_EXCEEDED
        else:
            self.event_type = EventType.METRIC_RECORDED


class EventBus:
    """
    Asynchronous event bus for pipeline components.
    
    Implements a publish-subscribe pattern with weak references
    to prevent memory leaks.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[EventType, WeakSet[Callable]] = {}
        self._async_subscribers: Dict[EventType, WeakSet[Callable]] = {}
        self._wildcard_subscribers: WeakSet[Callable] = WeakSet()
        self._async_wildcard_subscribers: WeakSet[Callable] = WeakSet()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the event bus processor."""
        if not self._running:
            self._running = True
            self._processor_task = asyncio.create_task(self._process_events())
            logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus processor."""
        self._running = False
        if self._processor_task:
            await self._event_queue.put(None)  # Sentinel to stop processor
            await self._processor_task
            logger.info("Event bus stopped")
    
    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: Callable[[Event], None],
        is_async: bool = False
    ):
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to (None for all events)
            handler: Function to call when event occurs
            is_async: Whether the handler is async
        """
        if event_type is None:
            # Wildcard subscription
            if is_async:
                self._async_wildcard_subscribers.add(handler)
            else:
                self._wildcard_subscribers.add(handler)
        else:
            # Specific event subscription
            if is_async:
                if event_type not in self._async_subscribers:
                    self._async_subscribers[event_type] = WeakSet()
                self._async_subscribers[event_type].add(handler)
            else:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = WeakSet()
                self._subscribers[event_type].add(handler)
        
        logger.debug(f"Subscribed {'async' if is_async else 'sync'} handler to {event_type}")
    
    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: Callable[[Event], None]
    ):
        """Unsubscribe from events.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to remove
        """
        if event_type is None:
            self._wildcard_subscribers.discard(handler)
            self._async_wildcard_subscribers.discard(handler)
        else:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(handler)
            if event_type in self._async_subscribers:
                self._async_subscribers[event_type].discard(handler)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        await self._event_queue.put(event)
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await self._event_queue.get()
                if event is None:  # Sentinel value to stop
                    break
                
                await self._dispatch_event(event)
                
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to all relevant subscribers.
        
        Args:
            event: Event to dispatch
        """
        tasks = []
        
        # Dispatch to specific event type subscribers
        if event.event_type in self._subscribers:
            for handler in self._subscribers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in sync event handler: {e}", exc_info=True)
        
        if event.event_type in self._async_subscribers:
            for handler in self._async_subscribers[event.event_type]:
                tasks.append(self._call_async_handler(handler, event))
        
        # Dispatch to wildcard subscribers
        for handler in self._wildcard_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in wildcard sync handler: {e}", exc_info=True)
        
        for handler in self._async_wildcard_subscribers:
            tasks.append(self._call_async_handler(handler, event))
        
        # Wait for all async handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_async_handler(self, handler: Callable, event: Event):
        """Call an async event handler safely.
        
        Args:
            handler: Async handler function
            event: Event to pass to handler
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in async event handler: {e}", exc_info=True)
    
    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Get count of subscribers for an event type.
        
        Args:
            event_type: Event type to check (None for total count)
            
        Returns:
            Number of subscribers
        """
        if event_type is None:
            total = len(self._wildcard_subscribers) + len(self._async_wildcard_subscribers)
            for subscribers in self._subscribers.values():
                total += len(subscribers)
            for subscribers in self._async_subscribers.values():
                total += len(subscribers)
            return total
        else:
            count = 0
            if event_type in self._subscribers:
                count += len(self._subscribers[event_type])
            if event_type in self._async_subscribers:
                count += len(self._async_subscribers[event_type])
            return count


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.
    
    Returns:
        Global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus