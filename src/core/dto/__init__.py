"""Data Transfer Objects for the application."""

from .base import BaseDTO
from .github import RepositoryDTO, IssueDTO, AnalysisDTO

__all__ = ['BaseDTO', 'RepositoryDTO', 'IssueDTO', 'AnalysisDTO']
