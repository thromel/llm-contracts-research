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
    pipeline_stage: str
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
    origin_stage: str
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
    pipeline_stage: str
    contract_category: str
    comment_analysis: CommentAnalysisDTO
    error_propagation: ErrorPropagationDTO
    suggested_new_contracts: List[NewContractDTO] = field(default_factory=list)


@dataclass
class AnalysisMetadataDTO:
    """DTO for analysis metadata."""
    repository: Optional[str]
    analysis_timestamp: str
    num_issues: int


@dataclass
class AnalysisResultsDTO:
    """DTO for complete analysis results."""
    metadata: AnalysisMetadataDTO
    analyzed_issues: List[ContractAnalysisDTO]
