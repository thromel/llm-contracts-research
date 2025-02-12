"""Progress tracking implementation."""

import logging
from src.analysis.core.interfaces import IProgressTracker

logger = logging.getLogger(__name__)


class DefaultProgressTracker(IProgressTracker):
    """Default implementation of progress tracking."""

    def update(self, current: int, total: int, message: str) -> None:
        """Update progress.

        Args:
            current: Current progress
            total: Total items
            message: Progress message
        """
        logger.info(f"Progress: {current}/{total} - {message}")

    def complete(self) -> None:
        """Mark progress as complete."""
        logger.info("Analysis complete")
