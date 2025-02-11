"""Checkpoint handling for analysis process."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.config import settings
from src.analysis.core.dto import ContractAnalysisDTO, AnalysisMetadataDTO

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Base class for checkpoint errors."""
    pass


class CheckpointIOError(CheckpointError):
    """Error during checkpoint I/O operations."""
    pass


class CheckpointHandler:
    """Handles checkpointing during analysis."""

    def __init__(self, checkpoint_dir: Path = None):
        """Initialize checkpoint handler.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir or Path(
            settings.DATA_DIR) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis checkpoint.

        Args:
            analyzed_issues: List of analyzed issues
            metadata: Analysis metadata

        Raises:
            CheckpointIOError: If checkpoint cannot be saved
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / \
                f"checkpoint_{timestamp}.json"

            # Convert DTOs to dictionaries for JSON serialization
            checkpoint_data = {
                "metadata": metadata.__dict__,
                "analyzed_issues": [self._dto_to_dict(issue) for issue in analyzed_issues]
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"Saved checkpoint to {checkpoint_file}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise CheckpointIOError(f"Failed to save checkpoint: {str(e)}")

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load latest checkpoint if it exists.

        Returns:
            Checkpoint data or None if no checkpoint exists

        Raises:
            CheckpointIOError: If checkpoint cannot be loaded
        """
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return None

            # Get most recent checkpoint
            latest_checkpoint = max(
                checkpoints, key=lambda p: p.stat().st_mtime)

            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
                logger.info(f"Loaded checkpoint from {latest_checkpoint}")
                return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise CheckpointIOError(f"Failed to load checkpoint: {str(e)}")

    def clear_checkpoint(self) -> None:
        """Clear all checkpoints.

        Raises:
            CheckpointIOError: If checkpoints cannot be cleared
        """
        try:
            for checkpoint in self.checkpoint_dir.glob("checkpoint_*.json"):
                checkpoint.unlink()
            logger.info("Cleared all checkpoints")

        except Exception as e:
            logger.error(f"Error clearing checkpoints: {e}")
            raise CheckpointIOError(f"Failed to clear checkpoints: {str(e)}")

    def _dto_to_dict(self, dto: Any) -> Dict[str, Any]:
        """Convert a DTO to a dictionary, handling nested DTOs.

        Args:
            dto: Any DTO object

        Returns:
            Dictionary representation of the DTO
        """
        if hasattr(dto, '__dict__'):
            result = {}
            for key, value in dto.__dict__.items():
                if isinstance(value, list):
                    result[key] = [self._dto_to_dict(item) for item in value]
                elif hasattr(value, '__dict__'):
                    result[key] = self._dto_to_dict(value)
                else:
                    result[key] = value
            return result
        return dto
