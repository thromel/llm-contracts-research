"""
Data Acquisition Module

Handles multi-source data collection from GitHub Issues & Discussions
and Stack Overflow posts with normalization to common JSON format.
"""

from .github import GitHubAcquisition
from .stackoverflow import StackOverflowAcquisition
from .normalizer import DataNormalizer
from .acquisition_orchestrator import AcquisitionOrchestrator

__all__ = [
    'GitHubAcquisition',
    'StackOverflowAcquisition',
    'DataNormalizer',
    'AcquisitionOrchestrator'
]
