"""Data transfer objects for analysis module."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class CommentDTO:
    """Data transfer object for a GitHub comment."""
    id: str
    body: str
    created_at: datetime


@dataclass
class IssueDTO:
    """Data transfer object for a GitHub issue."""
    id: str
    number: int
    title: str
    body: str
    state: str
    created_at: datetime
    url: str
    first_comments: List[CommentDTO] = field(default_factory=list)


@dataclass
class AnalysisMetadataDTO:
    """Data transfer object for analysis metadata."""
    repository: str
    analysis_timestamp: str
    num_issues: int
    analyzer_model: str = "gpt-4"
    analyzer_version: str = "1.0.0"


@dataclass
class IssueAnalysisDTO:
    """Data transfer object for a single issue analysis."""
    issue_id: str
    issue_number: int
    issue_title: str
    issue_url: str
    has_violation: bool
    violation_type: Optional[str] = None
    severity: Optional[str] = None  # high, medium, low
    confidence: Optional[str] = None  # high, medium, low
    description: Optional[str] = None
    resolution_status: Optional[str] = None
    resolution_details: Optional[str] = None
    resolution_time: Optional[float] = None  # in hours
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_id": self.issue_id,
            "issue_number": self.issue_number,
            "issue_title": self.issue_title,
            "issue_url": self.issue_url,
            "has_violation": self.has_violation,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "resolution_status": self.resolution_status,
            "resolution_details": self.resolution_details,
            "resolution_time": self.resolution_time,
            "analysis_timestamp": self.analysis_timestamp
        }


@dataclass
class AnalysisResultsDTO:
    """Data transfer object for complete analysis results."""
    metadata: AnalysisMetadataDTO
    analyzed_issues: List[IssueAnalysisDTO] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "repository": self.metadata.repository,
                "analysis_timestamp": self.metadata.analysis_timestamp,
                "num_issues": self.metadata.num_issues,
                "analyzer_model": self.metadata.analyzer_model,
                "analyzer_version": self.metadata.analyzer_version
            },
            "analyzed_issues": [issue.to_dict() for issue in self.analyzed_issues]
        }
