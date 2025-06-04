"""
Unit tests for the event system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from pipeline.core.events import (
    EventBus, Event, EventType, PipelineEvent,
    DataProcessedEvent, ErrorEvent
)


@pytest.mark.unit
class TestEventBus:
    """Test the EventBus implementation."""
    
    @pytest.mark.asyncio
    async def test_event_bus_lifecycle(self):
        """Test starting and stopping the event bus."""
        bus = EventBus()
        
        # Should start successfully
        await bus.start()
        assert bus._running is True
        assert bus._processor_task is not None
        
        # Should stop successfully
        await bus.stop()
        assert bus._running is False
    
    @pytest.mark.asyncio
    async def test_sync_event_subscription(self, event_bus: EventBus):
        """Test synchronous event subscription and publishing."""
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        # Subscribe to specific event type
        event_bus.subscribe(EventType.PIPELINE_STARTED, handler)
        
        # Publish event
        event = PipelineEvent(
            event_type=EventType.PIPELINE_STARTED,
            pipeline_name="test_pipeline",
            status="started"
        )
        await event_bus.publish(event)
        
        # Allow event processing
        await asyncio.sleep(0.1)
        
        # Should have received the event
        assert len(received_events) == 1
        assert received_events[0].pipeline_name == "test_pipeline"
    
    @pytest.mark.asyncio
    async def test_async_event_subscription(self, event_bus: EventBus):
        """Test asynchronous event subscription."""
        received_events = []
        
        async def async_handler(event: Event):
            await asyncio.sleep(0.01)  # Simulate async work
            received_events.append(event)
        
        # Subscribe async handler
        event_bus.subscribe(EventType.DATA_ACQUIRED, async_handler, is_async=True)
        
        # Publish event
        event = DataProcessedEvent(
            event_type=EventType.DATA_ACQUIRED,
            stage="acquisition",
            items_processed=100,
            items_passed=90
        )
        await event_bus.publish(event)
        
        # Allow async processing
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].items_processed == 100
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus: EventBus):
        """Test wildcard event subscription."""
        all_events = []
        
        def wildcard_handler(event: Event):
            all_events.append(event)
        
        # Subscribe to all events
        event_bus.subscribe(None, wildcard_handler)
        
        # Publish different event types
        events = [
            PipelineEvent(
                event_type=EventType.PIPELINE_STARTED,
                pipeline_name="test",
                status="started"
            ),
            ErrorEvent(
                event_type=EventType.ERROR_OCCURRED,
                error_type="TestError",
                error_message="Test error message"
            ),
            DataProcessedEvent(
                event_type=EventType.DATA_FILTERED,
                stage="filtering",
                items_processed=50
            )
        ]
        
        for event in events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Should receive all events
        assert len(all_events) == 3
        assert isinstance(all_events[0], PipelineEvent)
        assert isinstance(all_events[1], ErrorEvent)
        assert isinstance(all_events[2], DataProcessedEvent)
    
    @pytest.mark.asyncio
    async def test_event_unsubscribe(self, event_bus: EventBus):
        """Test event unsubscription."""
        call_count = 0
        
        def handler(event: Event):
            nonlocal call_count
            call_count += 1
        
        # Subscribe and then unsubscribe
        event_bus.subscribe(EventType.PIPELINE_STARTED, handler)
        event_bus.unsubscribe(EventType.PIPELINE_STARTED, handler)
        
        # Publish event
        event = PipelineEvent(
            event_type=EventType.PIPELINE_STARTED,
            pipeline_name="test",
            status="started"
        )
        await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Handler should not be called
        assert call_count == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_handlers(self, event_bus: EventBus):
        """Test that errors in handlers don't break the event bus."""
        successful_calls = []
        
        def failing_handler(event: Event):
            raise RuntimeError("Handler error")
        
        def working_handler(event: Event):
            successful_calls.append(event)
        
        # Subscribe both handlers
        event_bus.subscribe(EventType.ERROR_OCCURRED, failing_handler)
        event_bus.subscribe(EventType.ERROR_OCCURRED, working_handler)
        
        # Publish event
        event = ErrorEvent(
            event_type=EventType.ERROR_OCCURRED,
            error_type="Test",
            error_message="Test"
        )
        await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Working handler should still be called
        assert len(successful_calls) == 1
    
    def test_subscriber_count(self):
        """Test getting subscriber counts."""
        bus = EventBus()
        
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()
        
        # Add subscribers
        bus.subscribe(EventType.PIPELINE_STARTED, handler1)
        bus.subscribe(EventType.PIPELINE_STARTED, handler2)
        bus.subscribe(EventType.DATA_ACQUIRED, handler3)
        bus.subscribe(None, handler1)  # Wildcard
        
        # Check counts
        assert bus.get_subscriber_count(EventType.PIPELINE_STARTED) == 2
        assert bus.get_subscriber_count(EventType.DATA_ACQUIRED) == 1
        assert bus.get_subscriber_count(EventType.ERROR_OCCURRED) == 0
        assert bus.get_subscriber_count() == 4  # Total


@pytest.mark.unit
class TestEventTypes:
    """Test event type implementations."""
    
    def test_pipeline_event_creation(self):
        """Test PipelineEvent creation and status mapping."""
        # Test started status
        event = PipelineEvent(
            pipeline_name="test",
            pipeline_id="123",
            status="started",
            event_type=EventType.PIPELINE_STARTED  # Will be overridden
        )
        assert event.event_type == EventType.PIPELINE_STARTED
        
        # Test completed status
        event = PipelineEvent(
            pipeline_name="test",
            status="completed",
            event_type=EventType.PIPELINE_STARTED  # Will be overridden
        )
        assert event.event_type == EventType.PIPELINE_COMPLETED
        
        # Test failed status
        event = PipelineEvent(
            pipeline_name="test",
            status="failed",
            event_type=EventType.PIPELINE_STARTED  # Will be overridden
        )
        assert event.event_type == EventType.PIPELINE_FAILED
    
    def test_data_processed_event_stage_mapping(self):
        """Test DataProcessedEvent stage to event type mapping."""
        stages = [
            ("acquisition", EventType.DATA_ACQUIRED),
            ("filtering", EventType.DATA_FILTERED),
            ("screening", EventType.DATA_SCREENED),
            ("labeling", EventType.DATA_LABELED)
        ]
        
        for stage, expected_type in stages:
            event = DataProcessedEvent(
                stage=stage,
                items_processed=10,
                event_type=EventType.PIPELINE_STARTED  # Will be overridden
            )
            assert event.event_type == expected_type
    
    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = ErrorEvent(
            event_type=EventType.ERROR_OCCURRED,
            source="test_component",
            error_type="ValidationError",
            error_message="Invalid input",
            error_details={"field": "test_field"},
            is_recoverable=True
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "error_occurred"
        assert event_dict["source"] == "test_component"
        assert "timestamp" in event_dict
        assert "metadata" in event_dict