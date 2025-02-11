"""Base protocol for results storage."""

from typing import Protocol
from src.analysis.core.dto import ContractAnalysisDTO, AnalysisMetadataDTO, AnalysisResultsDTO


class ResultsStorage(Protocol):
    """Protocol for analysis results storage."""

    def save_results(self, analyzed_issues: list[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis results.

        Args:
            analyzed_issues: List of analysis results as DTOs
            metadata: Analysis metadata as DTO
        """
        pass

    def load_results(self, file_path: str) -> AnalysisResultsDTO:
        """Load analysis results from storage.

        Args:
            file_path: Path to the results file

        Returns:
            Analysis results as DTO
        """
        pass
