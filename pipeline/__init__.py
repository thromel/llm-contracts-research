"""
LLM Contracts Research Pipeline

A modular pipeline for multi-source data acquisition, processing, and analysis
of LLM API contract violations in GitHub issues and Stack Overflow posts.

Pipeline Stages:
1. Data Acquisition (GitHub Issues + Stack Overflow)
2. Keyword Pre-filtering 
3. LLM Screening (DeepSeek-R1 + GPT-4.1)
4. Human Labelling & Taxonomy
5. Reliability Validation (Fleiss κ ≥ 0.80)
6. Statistical Analysis & Dashboards

Architecture follows the methodology described in the research paper,
with full provenance tracking in MongoDB Atlas.
"""

__version__ = "1.0.0"
__author__ = "LLM Contracts Research Team"

# Pipeline stage imports
from .data_acquisition import GitHubAcquisition, StackOverflowAcquisition
from .preprocessing import KeywordPreFilter
from .llm_screening import BulkScreener, BorderlineScreener
from .labelling import TripleLabeller, TaxonomyManager
from .reliability import FleissKappaValidator, ReliabilityAnalyzer
from .analysis import StatisticalAnalyzer, FrequencyAnalyzer
from .dashboards import PipelineMonitor, ResultsDashboard

__all__ = [
    # Data acquisition
    'GitHubAcquisition',
    'StackOverflowAcquisition',

    # Preprocessing
    'KeywordPreFilter',

    # LLM screening
    'BulkScreener',
    'BorderlineScreener',

    # Labelling
    'TripleLabeller',
    'TaxonomyManager',

    # Reliability
    'FleissKappaValidator',
    'ReliabilityAnalyzer',

    # Analysis
    'StatisticalAnalyzer',
    'FrequencyAnalyzer',

    # Dashboards
    'PipelineMonitor',
    'ResultsDashboard'
]
