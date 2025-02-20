"""Progress tracking implementations."""
from typing import Optional
from tqdm import tqdm

from .interfaces import ProgressTracker


class TqdmProgressTracker(ProgressTracker):
    """Progress tracker implementation using tqdm."""

    def __init__(self):
        """Initialize the progress tracker."""
        self._progress_bar: Optional[tqdm] = None

    def start_operation(self, total: int, description: str) -> None:
        """Start tracking an operation.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        if self._progress_bar:
            self._progress_bar.close()

        self._progress_bar = tqdm(
            total=total,
            desc=description,
            unit="issues",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=100,
            leave=False
        )

    def update(self, amount: int = 1) -> None:
        """Update progress.

        Args:
            amount: Amount to increment progress by
        """
        if self._progress_bar:
            self._progress_bar.update(amount)

    def complete(self) -> None:
        """Complete the operation."""
        if self._progress_bar:
            self._progress_bar.close()
            self._progress_bar = None
