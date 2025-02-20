from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from .base import BaseDTO


@dataclass
class RepositoryDTO(BaseDTO):
    """Repository data transfer object."""
    github_repo_id: int
    owner: str
    name: str
    full_name: str
    url: str
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RepositoryDTO':
        return cls(
            github_repo_id=data['github_repo_id'],
            owner=data['owner'],
            name=data['name'],
            full_name=data['full_name'],
            url=data['url'],
            id=str(data.get('_id', data.get('id'))),
            created_at=data.get('created_at', datetime.utcnow()),
            updated_at=data.get('updated_at', datetime.utcnow())
        )

    def to_dict(self) -> Dict:
        return {
            'github_repo_id': self.github_repo_id,
            'owner': self.owner,
            'name': self.name,
            'full_name': self.full_name,
            'url': self.url,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


@dataclass
class IssueDTO(BaseDTO):
    """Issue data transfer object."""
    github_issue_id: int
    repository_id: str
    title: str
    body: str
    status: str
    id: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    assignees: Dict[str, int] = field(default_factory=dict)
    comments: List[Dict[str, any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'IssueDTO':
        return cls(
            github_issue_id=data['github_issue_id'],
            repository_id=data['repository_id'],
            title=data['title'],
            body=data['body'],
            status=data['status'],
            id=str(data.get('_id', data.get('id'))),
            labels=data.get('labels', []),
            assignees=data.get('assignees', {}),
            comments=data.get('comments', []),
            created_at=data.get('created_at', datetime.utcnow()),
            updated_at=data.get('updated_at', datetime.utcnow()),
            closed_at=data.get('closed_at')
        )

    def to_dict(self) -> Dict:
        return {
            'github_issue_id': self.github_issue_id,
            'repository_id': self.repository_id,
            'title': self.title,
            'body': self.body,
            'status': self.status,
            'labels': self.labels,
            'assignees': self.assignees,
            'comments': self.comments,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'closed_at': self.closed_at
        }


@dataclass
class AnalysisDTO(BaseDTO):
    """Analysis data transfer object."""
    issue_id: str
    user_id: str
    analysis_text: str
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisDTO':
        return cls(
            issue_id=data['issue_id'],
            user_id=data['user_id'],
            analysis_text=data['analysis_text'],
            id=str(data.get('_id', data.get('id'))),
            created_at=data.get('created_at', datetime.utcnow()),
            updated_at=data.get('updated_at', datetime.utcnow())
        )

    def to_dict(self) -> Dict:
        return {
            'issue_id': self.issue_id,
            'user_id': self.user_id,
            'analysis_text': self.analysis_text,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
