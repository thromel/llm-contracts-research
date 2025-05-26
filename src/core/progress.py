"""Progress tracking implementations."""
from typing import Optional
import sys
from datetime import datetime

from .interfaces import ProgressTracker


class TqdmProgressTracker(ProgressTracker):
    """Progress tracker implementation using simple print statements."""

    def __init__(self):
        """Initialize the progress tracker."""
        self._count = 0
        self._total = None
        self._description = ""
        self._start_time = None

    def start_operation(self, total: Optional[int], description: str) -> None:
        """Start tracking an operation.

        Args:
            total: Total number of items to process (None for unknown)
            description: Description of the operation
        """
        self._count = 0
        self._total = total
        self._description = description
        self._start_time = datetime.now()
        self._print_progress()

    def update(self, amount: int = 1, description: Optional[str] = None) -> None:
        """Update progress.

        Args:
            amount: Amount to increment progress by
            description: Optional new description
        """
        self._count += amount
        if description:
            self._description = description
        self._print_progress()

    def _print_progress(self) -> None:
        """Print the current progress."""
        if self._total:
            percentage = (self._count / self._total) * 100
            message = f"\r{self._description} - {self._count}/{self._total} ({percentage:.1f}%)"
        else:
            message = f"\r{self._description} - {self._count} issues processed"

        # Calculate elapsed time
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            message += f" - {elapsed.seconds}s elapsed"

        sys.stdout.write(message)
        sys.stdout.flush()

    def complete(self) -> None:
        """Complete the operation."""
        if self._count > 0:  # Only print final newline if we showed progress
            sys.stdout.write("\n")
            sys.stdout.flush()

    def start_analysis(self, total_issues: int, repository_name: str) -> None:
        """Start analysis tracking.

        Args:
            total_issues: Total number of issues to analyze
            repository_name: Name of the repository being analyzed
        """
        self.start_operation(total_issues, f"Analyzing {repository_name}")

    def update_progress(self, amount: int = 1) -> None:
        """Update analysis progress.

        Args:
            amount: Amount to increment progress by
        """
        self.update(amount)
