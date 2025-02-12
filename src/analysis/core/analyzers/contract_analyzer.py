"""Contract analyzer implementation."""

import json
import logging
from typing import Optional

from src.analysis.core.prompts import get_system_prompt, get_user_prompt
from src.analysis.core.clients.base import LLMClient
from src.analysis.core.interfaces import IAnalyzer, IResponseCleaner, IResponseValidator
from src.analysis.core.dto import ContractAnalysisDTO, dict_to_contract_analysis_dto

logger = logging.getLogger(__name__)


class ContractAnalyzer(IAnalyzer):
    """Core contract violation analyzer."""

    def __init__(
        self,
        llm_client: LLMClient,
        response_cleaner: IResponseCleaner,
        response_validator: IResponseValidator
    ):
        """Initialize analyzer with components.

        Args:
            llm_client: LLM API client
            response_cleaner: Response cleaning strategy
            response_validator: Response validation strategy
        """
        self.llm_client = llm_client
        self.response_cleaner = response_cleaner
        self.response_validator = response_validator
        self.system_prompt = get_system_prompt()

    def analyze_issue(self, title: str, body: str, comments: Optional[str] = None) -> ContractAnalysisDTO:
        """Analyze a GitHub issue for contract violations.

        Args:
            title: Issue title
            body: Issue body
            comments: Optional issue comments

        Returns:
            Analysis results as ContractAnalysisDTO
        """
        try:
            formatted_comments = self._format_comments(comments)
            user_prompt = get_user_prompt(title, body, formatted_comments)
            content = self.llm_client.get_analysis(
                self.system_prompt, user_prompt)

            if not content:
                raise ValueError("Empty response from LLM")

            cleaned_content = self.response_cleaner.clean(content)
            try:
                analysis_dict = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse response: {e}\nCleaned content: {cleaned_content}")
                return self._get_error_analysis("Failed to parse analysis response")

            try:
                self.response_validator.validate(analysis_dict)
                return dict_to_contract_analysis_dto(analysis_dict)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Invalid analysis structure: {e}")
                return self._get_error_analysis("Invalid analysis structure")

        except Exception as e:
            logger.error(f"Error analyzing issue: {e}")
            return self._get_error_analysis(str(e))

    def _format_comments(self, comments: Optional[str]) -> Optional[str]:
        """Format comments for analysis.

        Args:
            comments: Raw comments

        Returns:
            Formatted comments string
        """
        if not comments:
            return None

        if isinstance(comments, list):
            return "\n\n".join([
                f"Comment by {c.get('user', 'unknown')} at {c.get('created_at', 'unknown')}:\n{c.get('body', '')}"
                for c in comments
            ])
        return comments

    def _get_error_analysis(self, error_msg: str) -> ContractAnalysisDTO:
        """Generate error analysis result.

        Args:
            error_msg: Error message

        Returns:
            Error analysis as ContractAnalysisDTO
        """
        error_dict = {
            "has_violation": False,
            "violation_type": "ERROR",
            "severity": "low",
            "description": f"Analysis failed: {error_msg}",
            "confidence": "low",
            "root_cause": "Analysis error",
            "effects": ["Unable to determine contract violations"],
            "resolution_status": "ERROR",
            "resolution_details": "Please try analyzing the issue again",
            "contract_category": "unknown",
            "comment_analysis": {
                "supporting_evidence": [],
                "frequency": "unknown",
                "workarounds": [],
                "impact": "unknown"
            },
            "error_propagation": {
                "affected_stages": [],
                "propagation_path": "Analysis error contained"
            }
        }
        return dict_to_contract_analysis_dto(error_dict)
