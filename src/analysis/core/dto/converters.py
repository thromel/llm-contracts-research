"""Converter functions for DTOs."""

from datetime import datetime
from typing import Dict, Any, List

from src.analysis.core.dto.base import (
    CommentDTO,
    GithubIssueDTO,
    PatternFrequencyDTO,
    NewContractDTO,
    CommentAnalysisDTO,
    ErrorPropagationDTO,
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO
)


def dict_to_comment_dto(data: Dict[str, Any]) -> CommentDTO:
    """Convert dictionary to CommentDTO."""
    return CommentDTO(
        body=data['body'],
        created_at=data['created_at'],
        user=data.get('user')
    )


def dict_to_github_issue_dto(data: Dict[str, Any]) -> GithubIssueDTO:
    """Convert dictionary to GithubIssueDTO."""
    return GithubIssueDTO(
        number=data['number'],
        title=data['title'],
        body=data['body'],
        state=data['state'],
        created_at=data['created_at'],
        url=data['url'],
        labels=data['labels'],
        first_comments=[dict_to_comment_dto(
            c) for c in data.get('first_comments', [])],
        user=data.get('user'),
        closed_at=data.get('closed_at'),
        resolution_time=data.get('resolution_time')
    )


def dict_to_pattern_frequency_dto(data: Dict[str, Any]) -> PatternFrequencyDTO:
    """Convert dictionary to PatternFrequencyDTO."""
    return PatternFrequencyDTO(
        observed_count=data['observed_count'],
        confidence=data['confidence'],
        supporting_evidence=data['supporting_evidence']
    )


def dict_to_new_contract_dto(data: Dict[str, Any]) -> NewContractDTO:
    """Convert dictionary to NewContractDTO."""
    return NewContractDTO(
        name=data['name'],
        description=data['description'],
        rationale=data['rationale'],
        examples=data['examples'],
        parent_category=data['parent_category'],
        pipeline_stage=data['pipeline_stage'],
        pattern_frequency=dict_to_pattern_frequency_dto(
            data['pattern_frequency'])
    )


def dict_to_comment_analysis_dto(data: Dict[str, Any]) -> CommentAnalysisDTO:
    """Convert dictionary to CommentAnalysisDTO."""
    return CommentAnalysisDTO(
        supporting_evidence=data['supporting_evidence'],
        frequency=data['frequency'],
        workarounds=data['workarounds'],
        impact=data['impact']
    )


def dict_to_error_propagation_dto(data: Dict[str, Any]) -> ErrorPropagationDTO:
    """Convert dictionary to ErrorPropagationDTO."""
    return ErrorPropagationDTO(
        origin_stage=data['origin_stage'],
        affected_stages=data['affected_stages'],
        propagation_path=data['propagation_path']
    )


def dict_to_contract_analysis_dto(data: Dict[str, Any]) -> ContractAnalysisDTO:
    """Convert dictionary to ContractAnalysisDTO."""
    return ContractAnalysisDTO(
        has_violation=data['has_violation'],
        violation_type=data.get('violation_type'),
        severity=data['severity'],
        description=data['description'],
        confidence=data['confidence'],
        root_cause=data['root_cause'],
        effects=data['effects'],
        resolution_status=data['resolution_status'],
        resolution_details=data['resolution_details'],
        pipeline_stage=data['pipeline_stage'],
        contract_category=data['contract_category'],
        comment_analysis=dict_to_comment_analysis_dto(
            data['comment_analysis']),
        error_propagation=dict_to_error_propagation_dto(
            data['error_propagation']),
        suggested_new_contracts=[dict_to_new_contract_dto(
            c) for c in data.get('suggested_new_contracts', [])]
    )


def dict_to_analysis_metadata_dto(data: Dict[str, Any]) -> AnalysisMetadataDTO:
    """Convert dictionary to AnalysisMetadataDTO."""
    return AnalysisMetadataDTO(
        repository=data.get('repository'),
        analysis_timestamp=data['analysis_timestamp'],
        num_issues=data['num_issues']
    )


def dict_to_analysis_results_dto(data: Dict[str, Any]) -> AnalysisResultsDTO:
    """Convert dictionary to AnalysisResultsDTO."""
    return AnalysisResultsDTO(
        metadata=dict_to_analysis_metadata_dto(data['metadata']),
        analyzed_issues=[dict_to_contract_analysis_dto(
            issue) for issue in data['analyzed_issues']]
    )
