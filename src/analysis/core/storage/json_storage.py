"""JSON file storage implementation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ..dto import (
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO,
    dict_to_analysis_results_dto
)

logger = logging.getLogger(__name__)


class JSONResultsStorage:
    """JSON file storage for analysis results."""

    def __init__(self, output_dir: Path):
        """Initialize storage with output directory.

        Args:
            output_dir: Directory to store results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, analyzed_issues: list[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save results to JSON file.

        Args:
            analyzed_issues: List of analysis results as DTOs
            metadata: Analysis metadata as DTO
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / \
            f"github_issues_analysis_{timestamp}_final_violation_analysis.json"

        # Convert DTOs to dictionaries for JSON serialization
        data = {
            "metadata": metadata.__dict__,
            "analyzed_issues": [self._dto_to_dict(issue) for issue in analyzed_issues]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved analysis results to {output_file}")

    def load_results(self, file_path: str) -> AnalysisResultsDTO:
        """Load results from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Analysis results as DTO
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            return dict_to_analysis_results_dto(data)

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
