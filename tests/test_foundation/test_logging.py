"""Tests for the enhanced logging system."""

import json
import pytest
import logging
import tempfile
from unittest.mock import patch, MagicMock
from io import StringIO

from pipeline.foundation.logging import (
    StructuredFormatter, PipelineLogger, LogContext, OperationLogger,
    setup_logging, get_logger, set_correlation_id, get_correlation_id,
    generate_correlation_id, with_correlation_id
)
from pipeline.foundation.types import PipelineStage, LogLevel


class TestLogContext:
    """Test LogContext functionality."""
    
    def test_basic_context(self):
        """Test basic context creation."""
        context = LogContext(
            correlation_id="test-123",
            stage=PipelineStage.DATA_ACQUISITION,
            component="TestComponent"
        )
        
        assert context.correlation_id == "test-123"
        assert context.stage == PipelineStage.DATA_ACQUISITION
        assert context.component == "TestComponent"
    
    def test_to_dict_filters_none(self):
        """Test that to_dict filters out None values."""
        context = LogContext(
            correlation_id="test-123",
            stage=None,
            component="TestComponent"
        )
        
        result = context.to_dict()
        assert "correlation_id" in result
        assert "component" in result
        assert "stage" not in result


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        
        logger = logging.getLogger("test_logger")
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "Test message"
        assert data["module"] == "test_module"
        assert data["function"] == "test_function"
        assert data["line"] == 42
    
    def test_formatting_with_context(self):
        """Test formatting with context information."""
        formatter = StructuredFormatter()
        
        logger = logging.getLogger("test_logger")
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        context = LogContext(
            correlation_id="test-123",
            component="TestComponent"
        )
        record.context = context
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert data["correlation_id"] == "test-123"
        assert data["component"] == "TestComponent"
    
    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            logger = logging.getLogger("test_logger")
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.module = "test_module"
            record.funcName = "test_function"
            
            formatted = formatter.format(record)
            data = json.loads(formatted)
            
            assert "exception" in data
            assert data["exception"]["type"] == "ValueError"
            assert data["exception"]["message"] == "Test error"
            assert "traceback" in data["exception"]


class TestPipelineLogger:
    """Test PipelineLogger functionality."""
    
    def test_basic_logging(self):
        """Test basic logging operations."""
        context = LogContext(component="TestComponent")
        logger = PipelineLogger("test_logger", context)
        
        # Mock the underlying logger
        with patch.object(logger.adapter, 'info') as mock_info:
            logger.info("Test message", extra_field="extra_value")
            mock_info.assert_called_once_with("Test message", extra={'extra_field': 'extra_value'})
    
    def test_with_context(self):
        """Test creating logger with updated context."""
        original_context = LogContext(component="OriginalComponent")
        logger = PipelineLogger("test_logger", original_context)
        
        new_logger = logger.with_context(operation="test_operation")
        
        assert new_logger.context.component == "OriginalComponent"
        assert new_logger.context.operation == "test_operation"
    
    def test_start_operation(self):
        """Test starting a tracked operation."""
        logger = PipelineLogger("test_logger")
        
        with patch.object(logger, 'with_context') as mock_with_context:
            mock_new_logger = MagicMock()
            mock_with_context.return_value = mock_new_logger
            
            op_logger = logger.start_operation("test_operation")
            
            assert isinstance(op_logger, OperationLogger)
            mock_with_context.assert_called_once()


class TestOperationLogger:
    """Test OperationLogger functionality."""
    
    def test_operation_lifecycle(self):
        """Test complete operation lifecycle."""
        parent_logger = PipelineLogger("test_logger")
        
        with patch.object(parent_logger, 'with_context') as mock_with_context:
            mock_logger = MagicMock()
            mock_with_context.return_value = mock_logger
            
            op_logger = OperationLogger(parent_logger, "test_operation")
            
            # Test progress logging
            op_logger.progress("50% complete", items_processed=50)
            mock_logger.info.assert_called_with("Operation progress: 50% complete", items_processed=50)
            
            # Test completion
            op_logger.complete("Operation finished successfully")
            
            # Verify completion was logged with duration
            completion_call = mock_logger.info.call_args_list[-1]
            assert "Operation completed" in completion_call[0][0]
            assert "duration_seconds" in completion_call[1]
            assert completion_call[1]["operation_status"] == "completed"
    
    def test_operation_failure(self):
        """Test operation failure logging."""
        parent_logger = PipelineLogger("test_logger")
        
        with patch.object(parent_logger, 'with_context') as mock_with_context:
            mock_logger = MagicMock()
            mock_with_context.return_value = mock_logger
            
            op_logger = OperationLogger(parent_logger, "test_operation")
            
            # Test failure without exception
            op_logger.fail("Operation failed due to timeout")
            
            failure_call = mock_logger.error.call_args_list[-1]
            assert "Operation failed" in failure_call[0][0]
            assert failure_call[1]["operation_status"] == "failed"
            
            # Test failure with exception
            test_exception = ValueError("Test error")
            op_logger.fail("Operation failed with exception", exception=test_exception)
            
            mock_logger.exception.assert_called()


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def test_setup_console_logging(self):
        """Test setting up console logging."""
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging(level=LogLevel.DEBUG, console=True, structured=False)
            
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]
            
            assert 'console' in config['handlers']
            assert config['handlers']['console']['level'] == 'DEBUG'
            assert config['handlers']['console']['formatter'] == 'simple'
    
    def test_setup_file_logging(self):
        """Test setting up file logging."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_file = f.name
        
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging(level=LogLevel.INFO, log_file=log_file, structured=True)
            
            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]
            
            assert 'file' in config['handlers']
            assert config['handlers']['file']['filename'] == log_file
            assert config['handlers']['file']['formatter'] == 'structured'


class TestCorrelationId:
    """Test correlation ID functionality."""
    
    def test_set_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"
        set_correlation_id(test_id)
        
        assert get_correlation_id() == test_id
    
    def test_generate_correlation_id(self):
        """Test generating correlation ID."""
        corr_id = generate_correlation_id()
        
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
        # UUID format check
        assert len(corr_id.split('-')) == 5
    
    def test_with_correlation_id_decorator(self):
        """Test correlation ID decorator."""
        @with_correlation_id("test-decorator-123")
        def test_function():
            return get_correlation_id()
        
        result = test_function()
        assert result == "test-decorator-123"
        
        # Correlation ID should be reset after function
        assert get_correlation_id() is None
    
    def test_with_correlation_id_auto_generate(self):
        """Test correlation ID decorator with auto-generation."""
        @with_correlation_id()
        def test_function():
            return get_correlation_id()
        
        result = test_function()
        assert result is not None
        assert isinstance(result, str)
        
        # Should generate different IDs for different calls
        result2 = test_function()
        assert result != result2


class TestLoggerMixin:
    """Test LoggerMixin functionality."""
    
    def test_logger_mixin_basic(self):
        """Test basic LoggerMixin functionality."""
        from pipeline.foundation.logging import LoggerMixin
        
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
            
            def do_something(self):
                self.logger.info("Doing something")
        
        instance = TestClass()
        assert isinstance(instance.logger, PipelineLogger)
        assert instance.logger.context.component == "TestClass"
    
    def test_logger_mixin_context_update(self):
        """Test updating context in LoggerMixin."""
        from pipeline.foundation.logging import LoggerMixin
        
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
        
        instance = TestClass()
        original_logger = instance.logger
        
        instance.set_log_context(operation="test_operation")
        
        # Logger should be recreated with new context
        assert instance.logger.context.operation == "test_operation"


if __name__ == "__main__":
    pytest.main([__file__])