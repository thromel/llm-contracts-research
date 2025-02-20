"""Checkpoint manager implementation."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol

import logging

logger = logging.getLogger(__name__)


class CheckpointData(Protocol):
    """Protocol defining the structure of checkpoint data."""
    timestamp: str
    analyzed_issues: List[Dict]
    current_index: int
    total_issues: List[Dict]
    repo_name: str


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""
    pass


class CheckpointIOError(CheckpointError):
    """Exception raised when checkpoint I/O operations fail."""
    pass


class CheckpointManager:
    """Manages checkpoints for analysis progress."""

    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    def save_checkpoint(
        self,
        analyzed_issues: List[Any],
        current_index: int,
        total_issues: List[Any],
        repo_name: str
    ) -> None:
        """Save checkpoint data.

        Args:
            analyzed_issues: List of analyzed issues
            current_index: Current processing index
            total_issues: Total list of issues
            repo_name: Repository name
        """
        try:
            checkpoint_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_index": current_index,
                "total_issues": len(total_issues),
                "analyzed_issues": len(analyzed_issues),
                "repository": repo_name
            }

            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"Saved checkpoint at index {current_index}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data.

        Returns:
            Checkpoint data if exists, None otherwise
        """
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                logger.info(
                    f"Loaded checkpoint at index {checkpoint_data['current_index']}")
                return checkpoint_data
            return None

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear the existing checkpoint file.

        Raises:
            CheckpointIOError: If clearing checkpoint fails
        """
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint cleared")
            except Exception as exc:
                error_msg = "Error clearing checkpoint: {}".format(str(exc))
                logger.error(error_msg)
                raise CheckpointIOError(error_msg) from exc
