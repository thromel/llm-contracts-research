"""Response validation implementations."""

import logging
from typing import Dict, Any, List, Protocol

logger = logging.getLogger(__name__)


class ResponseValidator(Protocol):
    """Protocol for response validation strategies."""

    def validate(self, analysis: Dict[str, Any]) -> None:
        """Validate the analysis response.

        Args:
            analysis: Analysis dict to validate

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
            TypeError: If field types are incorrect
        """
        pass


class ContractAnalysisValidator:
    """Validates contract analysis responses."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(self, analysis: Dict[str, Any]) -> None:
        """Validate contract analysis response.

        Args:
            analysis: Analysis dict to validate

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
            TypeError: If field types are incorrect
        """
        # First ensure we have a dict with basic required fields
        if not isinstance(analysis, dict):
            raise TypeError("Analysis must be a dictionary")

        # Set default values for missing fields
        analysis.setdefault('has_violation', False)
        analysis.setdefault('violation_type', None)
        analysis.setdefault('severity', 'low')
        analysis.setdefault('confidence', 'low')
        analysis.setdefault('effects', [])
        analysis.setdefault('resolution_status', 'ERROR')
        analysis.setdefault('contract_category', 'unknown')

        # Ensure comment_analysis exists
        if 'comment_analysis' not in analysis:
            analysis['comment_analysis'] = {
                'supporting_evidence': [],
                'frequency': 'unknown',
                'workarounds': [],
                'impact': 'unknown'
            }

        # Ensure error_propagation exists
        if 'error_propagation' not in analysis:
            analysis['error_propagation'] = {
                'affected_stages': [],
                'propagation_path': 'Analysis error contained'
            }

        self._validate_required_fields(analysis)
        self._validate_field_types(analysis)
        self._validate_field_values(analysis)
        self._validate_optional_fields(analysis)

    def _validate_required_fields(self, analysis: Dict[str, Any]) -> None:
        """Validate presence of required fields."""
        required_fields = [
            "has_violation", "violation_type", "severity", "description",
            "confidence", "root_cause", "effects", "resolution_status",
            "resolution_details", "contract_category"
        ]
        missing = [field for field in required_fields if field not in analysis]
        if missing:
            raise KeyError(
                "Missing required fields in analysis: {}".format(missing))

    def _validate_field_types(self, analysis: Dict[str, Any]) -> None:
        """Validate field types."""
        if not isinstance(analysis["has_violation"], bool):
            analysis["has_violation"] = False

        if not isinstance(analysis["effects"], list):
            analysis["effects"] = []

    def _validate_field_values(self, analysis: Dict[str, Any]) -> None:
        """Validate field values."""
        valid_severities = ["high", "medium", "low"]
        if analysis["severity"] not in valid_severities:
            analysis["severity"] = "low"

        valid_confidences = ["high", "medium", "low"]
        if analysis["confidence"] not in valid_confidences:
            analysis["confidence"] = "low"

    def _validate_optional_fields(self, analysis: Dict[str, Any]) -> None:
        """Validate optional fields if present."""
        if "error_propagation" in analysis:
            self._validate_error_propagation(analysis["error_propagation"])
        if "suggested_new_contracts" in analysis:
            self._validate_suggested_contracts(
                analysis["suggested_new_contracts"])
        if "comment_analysis" in analysis:
            self._validate_comment_analysis(analysis["comment_analysis"])

    def _validate_error_propagation(self, error_prop: Dict[str, Any]) -> None:
        """Validate error propagation fields."""
        required_fields = ["affected_stages", "propagation_path"]
        for field in required_fields:
            if field not in error_prop:
                error_prop[field] = [
                ] if field == "affected_stages" else "Analysis error contained"

    def _validate_suggested_contracts(self, contracts: List[Dict[str, Any]]) -> None:
        """Validate suggested contracts."""
        if not isinstance(contracts, list):
            raise TypeError("suggested_new_contracts must be an array")

        required_fields = [
            "name", "description", "rationale", "examples",
            "parent_category", "pattern_frequency"
        ]
        for contract in contracts:
            missing = [
                field for field in required_fields if field not in contract]
            if missing:
                raise KeyError(
                    "Missing required fields in suggested contract: {}".format(missing))

    def _validate_comment_analysis(self, comment_analysis: Dict[str, Any]) -> None:
        """Validate comment analysis fields."""
        required_fields = ["supporting_evidence",
                           "frequency", "workarounds", "impact"]
        for field in required_fields:
            if field not in comment_analysis:
                if field in ["supporting_evidence", "workarounds"]:
                    comment_analysis[field] = []
                else:
                    comment_analysis[field] = "unknown"
