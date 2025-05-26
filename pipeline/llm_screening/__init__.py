"""
LLM Screening Module

Multi-modal LLM screening approaches:
1. Bulk screening with DeepSeek-R1 (cost-effective, high-throughput)
2. Agentic screening with LangChain (comprehensive, multi-agent analysis)
3. Borderline case re-evaluation with GPT-4.1 (high-accuracy edge cases)
"""

from .bulk_screener import BulkScreener
from .borderline_screener import BorderlineScreener
from .agentic_screener import (
    AgenticScreeningOrchestrator,
    ContractViolationDetectorAgent,
    TechnicalErrorAnalystAgent,
    ContextRelevanceJudgeAgent,
    FinalDecisionSynthesizerAgent,
    ContractViolationAnalysis,
    TechnicalErrorAnalysis,
    ContextRelevanceAnalysis,
    FinalDecision,
    AgentResult
)
from .screening_orchestrator import ScreeningOrchestrator

__all__ = [
    # Traditional screening
    'BulkScreener',
    'BorderlineScreener',

    # Agentic screening
    'AgenticScreeningOrchestrator',
    'ContractViolationDetectorAgent',
    'TechnicalErrorAnalystAgent',
    'ContextRelevanceJudgeAgent',
    'FinalDecisionSynthesizerAgent',

    # Data models
    'ContractViolationAnalysis',
    'TechnicalErrorAnalysis',
    'ContextRelevanceAnalysis',
    'FinalDecision',
    'AgentResult',

    # Orchestration
    'ScreeningOrchestrator'
]
