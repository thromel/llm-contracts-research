"""Checkpoint manager for saving and loading fetch state."""
import os
import json
import asyncio
from typing import Optional, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for resumable fetching."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(
            f"Initialized checkpoint manager with directory: {checkpoint_dir}")

    def _get_checkpoint_file(self, owner: str, repo: str) -> str:
        """Get the path to a repository's checkpoint file.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Path to checkpoint file
        """
        # Sanitize owner and repo names to be safe for filenames
        safe_owner = "".join(c for c in owner if c.isalnum() or c in "-_")
        safe_repo = "".join(c for c in repo if c.isalnum() or c in "-_")
        return os.path.join(self.checkpoint_dir, f"{safe_owner}_{safe_repo}_checkpoint.json")

    async def save_checkpoint(self, owner: str, repo: str, state: Dict) -> None:
        """Save checkpoint state for a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            state: State to save
        """
        checkpoint_file = self._get_checkpoint_file(owner, repo)
        temp_file = f"{checkpoint_file}.tmp"

        # Add metadata to state
        state.update({
            'last_updated': datetime.now().isoformat(),
            'owner': owner,
            'repo': repo
        })

        try:
            def write_checkpoint():
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=2)
                os.replace(temp_file, checkpoint_file)

            # Run file operations in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, write_checkpoint)
            logger.debug(f"Saved checkpoint for {owner}/{repo}: {state}")

        except Exception as e:
            logger.error(f"Error saving checkpoint for {owner}/{repo}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    async def load_checkpoint(self, owner: str, repo: str) -> Optional[Dict]:
        """Load checkpoint state for a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Saved state or None if no checkpoint exists
        """
        checkpoint_file = self._get_checkpoint_file(owner, repo)
        if not os.path.exists(checkpoint_file):
            logger.info(f"No checkpoint found for {owner}/{repo}")
            return None

        try:
            def read_checkpoint():
                with open(checkpoint_file, 'r') as f:
                    state = json.load(f)
                # Validate checkpoint data
                required_fields = {'page', 'issue_count',
                                   'owner', 'repo', 'last_updated'}
                if not all(field in state for field in required_fields):
                    logger.warning(
                        f"Corrupt checkpoint found for {owner}/{repo}, ignoring")
                    return None
                if state['owner'] != owner or state['repo'] != repo:
                    logger.warning(
                        f"Mismatched checkpoint found for {owner}/{repo}, ignoring")
                    return None
                return state

            # Run file operations in executor to avoid blocking
            loop = asyncio.get_running_loop()
            state = await loop.run_in_executor(None, read_checkpoint)

            if state:
                logger.info(
                    f"Loaded checkpoint for {owner}/{repo} from {state['last_updated']}")
            return state

        except Exception as e:
            logger.error(f"Error loading checkpoint for {owner}/{repo}: {e}")
            return None

    async def clear_checkpoint(self, owner: str, repo: str) -> None:
        """Clear checkpoint for a repository.

        Args:
            owner: Repository owner
            repo: Repository name
        """
        checkpoint_file = self._get_checkpoint_file(owner, repo)
        if os.path.exists(checkpoint_file):
            try:
                def remove_checkpoint():
                    os.remove(checkpoint_file)

                # Run file operation in executor to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, remove_checkpoint)
                logger.info(f"Cleared checkpoint for {owner}/{repo}")

            except Exception as e:
                logger.error(
                    f"Error clearing checkpoint for {owner}/{repo}: {e}")
