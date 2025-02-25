"""Checkpoint handler for analysis process."""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.analysis.core.dto import AnalysisMetadataDTO

logger = logging.getLogger(__name__)


class CheckpointHandler:
    """Manages checkpoints for analysis process."""

    def __init__(self, checkpoint_dir: str = "analysis_checkpoints"):
        """Initialize checkpoint handler.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            checkpoint_dir, "analysis_checkpoint.json")
        logger.info(
            f"Initialized checkpoint handler with file: {self.checkpoint_file}")

    def save_checkpoint(self, analyzed_issues: List[Dict[str, Any]], metadata: AnalysisMetadataDTO) -> None:
        """Save checkpoint state.

        Args:
            analyzed_issues: List of analyzed issues
            metadata: Analysis metadata
        """
        try:
            # Create checkpoint data
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "repo_name": metadata.repository,
                "analyzed_issues": analyzed_issues,
                "current_index": len(analyzed_issues),
                "total_issues": metadata.num_issues
            }

            # Ensure directory exists
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Create temp file for atomic write
            temp_file = f"{self.checkpoint_file}.tmp"

            # Write to temp file and then rename
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, default=self._json_serializer)

            # Atomic rename
            os.replace(temp_file, self.checkpoint_file)

            logger.info(
                f"Saved checkpoint with {len(analyzed_issues)} analyzed issues")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(f"{self.checkpoint_file}.tmp"):
                try:
                    os.remove(f"{self.checkpoint_file}.tmp")
                except Exception:
                    pass

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        if not os.path.exists(self.checkpoint_file):
            logger.info("No checkpoint found")
            return None

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            logger.info(
                f"Loaded checkpoint with {checkpoint_data.get('current_index', 0)} analyzed issues")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear checkpoint state."""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.error(f"Error clearing checkpoint: {str(e)}")

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for objects not serializable by default json code.

        Args:
            obj: Object to serialize

        Returns:
            Serialized object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")
