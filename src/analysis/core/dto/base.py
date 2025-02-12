"""Base DTOs for data transfer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class CommentDTO:
    """DTO for GitHub issue comments."""
    body: str
    created_at: str
    user: Optional[str] = None
    comment_id: Optional[int] = None
    html_url: Optional[str] = None
    updated_at: Optional[str] = None
    reactions: Optional[Dict[str, int]] = None


@dataclass
class GithubIssueDTO:
    """DTO for GitHub issues."""
    number: int
    title: str
    body: str
    state: str
    created_at: str
    url: str
    labels: List[str]
    first_comments: List[CommentDTO]
    user: Optional[str] = None
    closed_at: Optional[str] = None
    resolution_time: Optional[float] = None
    html_url: Optional[str] = None
    issue_id: Optional[int] = None
    updated_at: Optional[str] = None
    milestone: Optional[str] = None
    assignees: Optional[List[str]] = None
    reactions: Optional[Dict[str, int]] = None
    repository_url: Optional[str] = None
    repository_name: Optional[str] = None
    repository_owner: Optional[str] = None


@dataclass
class PatternFrequencyDTO:
    """DTO for contract pattern frequency analysis."""
    observed_count: int
    confidence: str
    supporting_evidence: str


@dataclass
class NewContractDTO:
    """DTO for suggested new contract types."""
    name: str
    description: str
    rationale: str
    examples: List[str]
    parent_category: str
    pattern_frequency: PatternFrequencyDTO


@dataclass
class CommentAnalysisDTO:
    """DTO for comment analysis results."""
    supporting_evidence: List[str]
    frequency: str
    workarounds: List[str]
    impact: str


@dataclass
class ErrorPropagationDTO:
    """DTO for error propagation analysis."""
    affected_stages: List[str]
    propagation_path: str


@dataclass
class ContractAnalysisDTO:
    """DTO for contract analysis results."""
    has_violation: bool
    violation_type: Optional[str]
    severity: str
    description: str
    confidence: str
    root_cause: str
    effects: List[str]
    resolution_status: str
    resolution_details: str
    contract_category: str
    comment_analysis: CommentAnalysisDTO
    error_propagation: ErrorPropagationDTO
    suggested_new_contracts: List[NewContractDTO] = field(default_factory=list)
    # Additional metadata
    issue_url: Optional[str] = None
    issue_number: Optional[int] = None
    issue_title: Optional[str] = None
    repository_name: Optional[str] = None
    repository_owner: Optional[str] = None
    analysis_timestamp: Optional[str] = None


@dataclass
class AnalysisMetadataDTO:
    """DTO for analysis metadata."""
    repository: Optional[str]
    analysis_timestamp: str
    num_issues: int
    repository_url: Optional[str] = None
    repository_owner: Optional[str] = None
    repository_name: Optional[str] = None
    repository_description: Optional[str] = None
    repository_stars: Optional[int] = None
    repository_forks: Optional[int] = None
    repository_language: Optional[str] = None
    analysis_version: Optional[str] = None
    analysis_model: Optional[str] = None
    analysis_batch_id: Optional[str] = None


@dataclass
class AnalysisResultsDTO:
    """DTO for complete analysis results."""
    metadata: AnalysisMetadataDTO
    analyzed_issues: List[ContractAnalysisDTO]
