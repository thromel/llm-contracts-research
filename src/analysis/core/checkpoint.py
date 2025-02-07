"""Checkpoint management for analysis processes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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
    """Manages checkpoints for analysis processes."""

    def __init__(self, output_dir: Path):
        """Initialize the checkpoint manager.

        Args:
            output_dir: Directory where checkpoints will be stored
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / 'analysis_checkpoint.json'

    def save_checkpoint(
        self,
        analyzed_issues: List[Dict],
        current_index: int,
        total_issues: List[Dict],
        repo_name: str
    ) -> None:
        """Save the current analysis state to a checkpoint file.

        Args:
            analyzed_issues: List of analyzed issue data
            current_index: Current position in analysis
            total_issues: Complete list of issues to analyze
            repo_name: Name of the repository being analyzed

        Raises:
            CheckpointIOError: If saving checkpoint fails
        """
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'analyzed_issues': analyzed_issues,
            'current_index': current_index,
            'total_issues': total_issues,
            'repo_name': repo_name
        }

        temp_checkpoint = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_checkpoint, 'w', encoding='utf-8') as file:
                json.dump(checkpoint_data, file, indent=2, ensure_ascii=False)
            temp_checkpoint.replace(self.checkpoint_file)
            logger.info("Checkpoint saved at index {}".format(current_index))
        except Exception as exc:
            error_msg = "Error saving checkpoint: {}".format(str(exc))
            logger.error(error_msg)
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
            raise CheckpointIOError(error_msg) from exc

    def load_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint if it exists.

        Returns:
            Optional[Dict]: Checkpoint data if exists, None otherwise

        Raises:
            CheckpointIOError: If loading checkpoint fails
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as file:
                checkpoint_data = json.load(file)
            logger.info("Loaded checkpoint from index {}".format(
                checkpoint_data['current_index']))
            return checkpoint_data
        except Exception as exc:
            error_msg = "Error loading checkpoint: {}".format(str(exc))
            logger.error(error_msg)
            raise CheckpointIOError(error_msg) from exc

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
