"""Data Transfer Objects for the analysis module."""

from src.analysis.core.dto.base import (
    CommentDTO,
    GithubIssueDTO,
    PatternFrequencyDTO,
    NewContractDTO,
    CommentAnalysisDTO,
    ErrorPropagationDTO,
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO,
    IssueAnalysisDTO
)
from src.analysis.core.dto.converters import (
    dict_to_comment_dto,
    dict_to_github_issue_dto,
    dict_to_pattern_frequency_dto,
    dict_to_new_contract_dto,
    dict_to_comment_analysis_dto,
    dict_to_error_propagation_dto,
    dict_to_contract_analysis_dto,
    dict_to_analysis_metadata_dto,
    dict_to_analysis_results_dto
)

__all__ = [
    # DTOs
    'CommentDTO',
    'GithubIssueDTO',
    'PatternFrequencyDTO',
    'NewContractDTO',
    'CommentAnalysisDTO',
    'ErrorPropagationDTO',
    'ContractAnalysisDTO',
    'AnalysisMetadataDTO',
    'AnalysisResultsDTO',
    'IssueAnalysisDTO',
    # Converters
    'dict_to_comment_dto',
    'dict_to_github_issue_dto',
    'dict_to_pattern_frequency_dto',
    'dict_to_new_contract_dto',
    'dict_to_comment_analysis_dto',
    'dict_to_error_propagation_dto',
    'dict_to_contract_analysis_dto',
    'dict_to_analysis_metadata_dto',
    'dict_to_analysis_results_dto'
]
