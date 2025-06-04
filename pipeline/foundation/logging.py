"""Enhanced logging system with structured logging and correlation IDs."""

import logging
import logging.config
import sys
import json
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from pathlib import Path

from .types import LogLevel, PipelineStage

# Context variable for correlation ID tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


@dataclass
class LogContext:
    """Structured logging context."""
    correlation_id: Optional[str] = None
    stage: Optional[PipelineStage] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_data['correlation_id'] = corr_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the log record
        extra_fields = getattr(record, 'extra', {})
        if extra_fields:
            log_data.update(extra_fields)
        
        # Add context if available
        if hasattr(record, 'context'):
            context_data = record.context.to_dict() if isinstance(record.context, LogContext) else record.context
            log_data.update(context_data)
        
        return json.dumps(log_data, default=str)


class ContextualLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes contextual information."""
    
    def __init__(self, logger: logging.Logger, context: LogContext):
        super().__init__(logger, {})
        self.context = context
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the logging call to add context."""
        # Add context to extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra']['context'] = self.context
        
        # Add correlation ID to extra if not already present
        if 'correlation_id' not in kwargs['extra']:
            corr_id = correlation_id.get()
            if corr_id:
                kwargs['extra']['correlation_id'] = corr_id
        
        return msg, kwargs
    
    def with_context(self, **kwargs) -> 'ContextualLoggerAdapter':
        """Create a new adapter with updated context."""
        new_context = LogContext(**{**asdict(self.context), **kwargs})
        return ContextualLoggerAdapter(self.logger, new_context)


class PipelineLogger:
    """Enhanced logger for pipeline operations."""
    
    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context or LogContext()
        self.adapter = ContextualLoggerAdapter(self.logger, self.context)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        self.adapter.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        self.adapter.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        self.adapter.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log error message."""
        self.adapter.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log critical message."""
        self.adapter.critical(msg, extra=kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.adapter.exception(msg, extra=kwargs)
    
    def with_context(self, **kwargs) -> 'PipelineLogger':
        """Create a new logger with updated context."""
        new_context = LogContext(**{**asdict(self.context), **kwargs})
        return PipelineLogger(self.logger.name, new_context)
    
    def start_operation(self, operation: str, **kwargs) -> 'OperationLogger':
        """Start a tracked operation."""
        return OperationLogger(self, operation, **kwargs)


class OperationLogger:
    """Logger for tracking specific operations."""
    
    def __init__(self, parent_logger: PipelineLogger, operation: str, **context_kwargs):
        self.parent_logger = parent_logger
        self.operation = operation
        self.start_time = datetime.now()
        self.logger = parent_logger.with_context(
            operation=operation,
            **context_kwargs
        )
        
        # Generate operation ID if not provided
        if 'operation_id' not in context_kwargs:
            self.operation_id = str(uuid.uuid4())
            self.logger = self.logger.with_context(operation_id=self.operation_id)
        
        self.logger.info(f"Starting operation: {operation}")
    
    def progress(self, msg: str, **kwargs) -> None:
        """Log operation progress."""
        self.logger.info(f"Operation progress: {msg}", **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log operation warning."""
        self.logger.warning(f"Operation warning: {msg}", **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log operation error."""
        self.logger.error(f"Operation error: {msg}", **kwargs)
    
    def complete(self, msg: Optional[str] = None, **kwargs) -> None:
        """Mark operation as complete."""
        duration = (datetime.now() - self.start_time).total_seconds()
        complete_msg = msg or f"Operation completed: {self.operation}"
        
        self.logger.info(complete_msg, 
                        duration_seconds=duration, 
                        operation_status="completed",
                        **kwargs)
    
    def fail(self, msg: Optional[str] = None, exception: Optional[Exception] = None, **kwargs) -> None:
        """Mark operation as failed."""
        duration = (datetime.now() - self.start_time).total_seconds()
        fail_msg = msg or f"Operation failed: {self.operation}"
        
        if exception:
            self.logger.exception(fail_msg,
                                duration_seconds=duration,
                                operation_status="failed",
                                **kwargs)
        else:
            self.logger.error(fail_msg,
                            duration_seconds=duration,
                            operation_status="failed",
                            **kwargs)


def setup_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None,
    structured: bool = True,
    console: bool = True
) -> None:
    """Setup logging configuration."""
    
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        level = LogLevel(level.upper())
    
    # Create logging configuration
    handlers = {}
    
    if console:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'structured' if structured else 'simple',
            'level': level.value
        }
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_path),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'structured' if structured else 'simple',
            'level': level.value
        }
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'structured': {
                '()': StructuredFormatter
            }
        },
        'handlers': handlers,
        'root': {
            'level': level.value,
            'handlers': list(handlers.keys())
        }
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str, context: Optional[LogContext] = None) -> PipelineLogger:
    """Get a pipeline logger with optional context."""
    return PipelineLogger(name, context)


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(corr_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def with_correlation_id(corr_id: Optional[str] = None):
    """Decorator to set correlation ID for a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            id_to_use = corr_id or generate_correlation_id()
            token = correlation_id.set(id_to_use)
            try:
                return func(*args, **kwargs)
            finally:
                correlation_id.reset(token)
        return wrapper
    return decorator


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
        self._log_context = LogContext(component=self.__class__.__name__)
    
    @property
    def logger(self) -> PipelineLogger:
        """Get logger for this component."""
        if self._logger is None:
            self._logger = get_logger(self.__class__.__module__, self._log_context)
        return self._logger
    
    def set_log_context(self, **kwargs) -> None:
        """Update logging context."""
        self._log_context = LogContext(**{**asdict(self._log_context), **kwargs})
        if self._logger:
            self._logger = get_logger(self.__class__.__module__, self._log_context)