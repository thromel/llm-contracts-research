"""
LangChain-Based Agentic LLM Screening Pipeline.

Multi-agent system for comprehensive analysis of LLM API contract violations:
1. Contract Violation Detector Agent
2. Technical Error Analyst Agent
3. Context Relevance Judge Agent
4. Final Decision Synthesizer Agent

Each agent has specialized prompts and reasoning capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import os

# LangChain imports
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import LLMChain, SequentialChain
from pydantic import BaseModel, Field

from ..common.models import FilteredPost, LLMScreeningResult, ContractType, RootCause, Effect
from ..common.database import MongoDBManager, ProvenanceTracker
from .prompts.agentic_screening_prompts import AgenticScreeningPrompts

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class ContractViolationAnalysis(BaseModel):
    """Structured analysis of contract violations."""
    has_violation: bool = Field(
        description="Whether a contract violation is present")
    violation_type: Optional[str] = Field(
        default=None, description="Type of contract violation")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting the analysis")
    violation_severity: str = Field(
        default="unknown", description="Severity: low, medium, high, critical")


class TechnicalErrorAnalysis(BaseModel):
    """Structured analysis of technical errors."""
    is_technical_error: bool = Field(
        description="Whether technical error is present")
    error_category: Optional[str] = Field(
        default=None, description="Category of error")
    root_cause: Optional[str] = Field(
        default=None, description="Likely root cause")
    api_related: bool = Field(description="Whether error is API-related")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    error_patterns: List[str] = Field(
        default_factory=list, description="Identified error patterns")


class ContextRelevanceAnalysis(BaseModel):
    """Structured analysis of context relevance."""
    is_llm_related: bool = Field(description="Whether content is LLM-related")
    relevance_score: float = Field(description="Relevance score 0.0-1.0")
    llm_indicators: List[str] = Field(
        default_factory=list, description="LLM-related indicators found")
    context_quality: str = Field(
        default="unknown", description="Quality: poor, fair, good, excellent")
    requires_expert_review: bool = Field(
        description="Whether expert review is needed")


class FinalDecision(BaseModel):
    """Final screening decision."""
    decision: str = Field(description="Y, N, or Borderline")
    confidence: float = Field(description="Overall confidence 0.0-1.0")
    rationale: str = Field(description="Decision rationale")
    contract_types_identified: List[str] = Field(
        default_factory=list, description="Contract types found")
    recommended_action: str = Field(description="Recommended next action")
    quality_flags: List[str] = Field(
        default_factory=list, description="Quality flags or concerns")


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    analysis: BaseModel
    processing_time: float
    token_usage: Optional[Dict[str, int]] = None
    errors: Optional[List[str]] = None


class CustomLLM(LLM):
    """Custom LLM wrapper for different API providers."""

    # Declare fields for Pydantic v2
    api_key: str = Field(description="API key for the LLM provider")
    model_name: str = Field(description="Name of the model to use")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the API")

    def __init__(self, api_key: str, model_name: str, base_url: str = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "custom_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make API call to the LLM provider."""
        try:
            return self._make_api_call(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            # Return a fallback response
            return self._get_fallback_response(prompt)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make actual API call based on the provider."""
        # Check if this is OpenAI
        if 'gpt' in self.model_name.lower() or 'openai' in str(self.base_url).lower():
            return self._make_openai_call(prompt, **kwargs)
        # Check if this is DeepSeek
        elif 'deepseek' in self.model_name.lower() or 'deepseek' in str(self.base_url).lower():
            return self._make_deepseek_call(prompt, **kwargs)
        else:
            # Generic implementation
            return self._make_generic_call(prompt, **kwargs)

    def _make_openai_call(self, prompt: str, **kwargs) -> str:
        """Make OpenAI API call."""
        try:
            # Import OpenAI here to avoid import errors
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("OpenAI library not properly installed")
                return self._get_fallback_response(prompt)

            # Set up the client
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2000),
                timeout=30
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            return self._get_fallback_response(prompt)

    def _make_deepseek_call(self, prompt: str, **kwargs) -> str:
        """Make DeepSeek API call."""
        try:
            import httpx

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2000)
            }

            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"DeepSeek API call failed: {str(e)}")
            return self._get_fallback_response(prompt)

    def _make_generic_call(self, prompt: str, **kwargs) -> str:
        """Make generic API call."""
        try:
            import httpx

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2000)
            }

            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Generic API call failed: {str(e)}")
            return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str, agent_type: str = None) -> str:
        """Generate a fallback response when API calls fail."""
        # If agent type is explicitly provided, use it
        if agent_type == "contract_violation":
            return '''{"has_violation": false, "violation_type": null, "confidence": 0.3, "evidence": ["API unavailable - fallback analysis"], "violation_severity": "low"}'''
        elif agent_type == "technical_error":
            return '''{"is_technical_error": false, "error_category": null, "root_cause": null, "api_related": false, "confidence": 0.3, "error_patterns": ["API unavailable - fallback analysis"]}'''
        elif agent_type == "context_relevance":
            return '''{"is_llm_related": true, "relevance_score": 0.5, "llm_indicators": ["potential LLM content"], "context_quality": "fair", "requires_expert_review": true}'''
        elif agent_type == "final_decision":
            return '''{"decision": "Borderline", "confidence": 0.3, "rationale": "API unavailable - requires manual review", "contract_types_identified": [], "recommended_action": "manual_review", "quality_flags": ["api_fallback"]}'''

        # Fallback to prompt-based detection
        prompt_lower = prompt.lower()

        # Look for context relevance indicators first (most specific)
        if ("context relevance" in prompt_lower or
            "llm_related" in prompt_lower or
            "relevance_score" in prompt_lower or
            "llm_indicators" in prompt_lower or
                "requires_expert_review" in prompt_lower):
            return '''{"is_llm_related": true, "relevance_score": 0.5, "llm_indicators": ["potential LLM content"], "context_quality": "fair", "requires_expert_review": true}'''

        # Contract violation detection
        elif ("contract violation" in prompt_lower or
              "has_violation" in prompt_lower or
              "violation_type" in prompt_lower):
            return '''{"has_violation": false, "violation_type": null, "confidence": 0.3, "evidence": ["API unavailable - fallback analysis"], "violation_severity": "low"}'''

        # Technical error detection
        elif ("technical error" in prompt_lower or
              "is_technical_error" in prompt_lower or
              "error_category" in prompt_lower):
            return '''{"is_technical_error": false, "error_category": null, "root_cause": null, "api_related": false, "confidence": 0.3, "error_patterns": ["API unavailable - fallback analysis"]}'''

        # Final decision detection
        elif ("final decision" in prompt_lower or
              "decision" in prompt_lower or
              "rationale" in prompt_lower):
            return '''{"decision": "Borderline", "confidence": 0.3, "rationale": "API unavailable - requires manual review", "contract_types_identified": [], "recommended_action": "manual_review", "quality_flags": ["api_fallback"]}'''

        else:
            # Default to context relevance if unclear
            return '''{"is_llm_related": true, "relevance_score": 0.5, "llm_indicators": ["potential LLM content"], "context_quality": "fair", "requires_expert_review": true}'''


class ContractViolationDetectorAgent:
    """Agent specialized in detecting LLM API contract violations."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.parser = PydanticOutputParser(
            pydantic_object=ContractViolationAnalysis)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content="{input_text}")
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt,
                              output_parser=self.parser)

    def _get_system_prompt(self) -> str:
        return AgenticScreeningPrompts.get_contract_violation_detector_prompt()

    async def analyze(self, post_content: str, title: str = "") -> ContractViolationAnalysis:
        """Analyze content for contract violations."""
        input_text = f"Title: {title}\n\nContent: {post_content}"

        try:
            result = await self.chain.arun(input_text=input_text)

            # Try to parse JSON response
            try:
                import json

                # Check if result is already parsed
                if isinstance(result, ContractViolationAnalysis):
                    return result
                elif isinstance(result, dict):
                    return ContractViolationAnalysis(**result)
                elif isinstance(result, str):
                    # Clean up the result string
                    result = result.strip()
                    if not result.startswith('{'):
                        # Try to extract JSON from the response
                        start = result.find('{')
                        end = result.rfind('}') + 1
                        if start != -1 and end != 0:
                            result = result[start:end]

                    parsed_result = json.loads(result)
                    return ContractViolationAnalysis(**parsed_result)
                else:
                    # Try the original parser as fallback
                    return self.parser.parse(str(result))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Raw response: {str(result)[:200]}...")
                # Try the original parser as fallback
                return self.parser.parse(str(result))
        except Exception as e:
            logger.error(f"Contract violation analysis failed: {str(e)}")
            # Return fallback analysis
            return ContractViolationAnalysis(
                has_violation=False,
                violation_type=None,
                confidence=0.0,
                evidence=[],
                violation_severity="unknown"
            )


class TechnicalErrorAnalystAgent:
    """Agent specialized in analyzing technical errors and root causes."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.parser = PydanticOutputParser(
            pydantic_object=TechnicalErrorAnalysis)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content="{input_text}")
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt,
                              output_parser=self.parser)

    def _get_system_prompt(self) -> str:
        return AgenticScreeningPrompts.get_technical_error_analyst_prompt()

    async def analyze(self, post_content: str, title: str = "") -> TechnicalErrorAnalysis:
        """Analyze content for technical errors."""
        input_text = f"Title: {title}\n\nContent: {post_content}"

        try:
            result = await self.chain.arun(input_text=input_text)

            # Try to parse JSON response
            try:
                import json

                # Check if result is already parsed
                if isinstance(result, TechnicalErrorAnalysis):
                    return result
                elif isinstance(result, dict):
                    return TechnicalErrorAnalysis(**result)
                elif isinstance(result, str):
                    # Clean up the result string
                    result = result.strip()
                    if not result.startswith('{'):
                        # Try to extract JSON from the response
                        start = result.find('{')
                        end = result.rfind('}') + 1
                        if start != -1 and end != 0:
                            result = result[start:end]

                    parsed_result = json.loads(result)
                    return TechnicalErrorAnalysis(**parsed_result)
                else:
                    # Try the original parser as fallback
                    return self.parser.parse(str(result))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Raw response: {str(result)[:200]}...")
                # Try the original parser as fallback
                return self.parser.parse(str(result))
        except Exception as e:
            logger.error(f"Technical error analysis failed: {str(e)}")
            return TechnicalErrorAnalysis(
                is_technical_error=False,
                error_category=None,
                root_cause=None,
                api_related=False,
                confidence=0.0,
                error_patterns=[]
            )


class ContextRelevanceJudgeAgent:
    """Agent specialized in judging LLM context relevance and quality."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.parser = PydanticOutputParser(
            pydantic_object=ContextRelevanceAnalysis)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content="{input_text}")
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt,
                              output_parser=self.parser)

    def _get_system_prompt(self) -> str:
        return AgenticScreeningPrompts.get_context_relevance_judge_prompt()

    async def analyze(self, post_content: str, title: str = "", metadata: Dict = None) -> ContextRelevanceAnalysis:
        """Analyze content relevance and quality."""
        # Handle metadata serialization safely
        meta_info = ""
        if metadata:
            try:
                import json
                meta_info = f"Metadata: {json.dumps(metadata, default=str)}"
            except Exception as e:
                logger.warning(f"Failed to serialize metadata: {e}")
                meta_info = "Metadata: unavailable"

        input_text = f"Title: {title}\n\nContent: {post_content}\n\n{meta_info}"

        try:
            result = await self.chain.arun(input_text=input_text)

            # Try to parse JSON response
            try:
                import json

                # Check if result is already parsed
                if isinstance(result, ContextRelevanceAnalysis):
                    return result
                elif isinstance(result, dict):
                    return ContextRelevanceAnalysis(**result)
                elif isinstance(result, str):
                    # Clean up the result string
                    result = result.strip()
                    if not result.startswith('{'):
                        # Try to extract JSON from the response
                        start = result.find('{')
                        end = result.rfind('}') + 1
                        if start != -1 and end != 0:
                            result = result[start:end]

                    parsed_result = json.loads(result)
                    return ContextRelevanceAnalysis(**parsed_result)
                else:
                    # Try the original parser as fallback
                    return self.parser.parse(str(result))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Raw response: {str(result)[:200]}...")
                # Try the original parser as fallback
                return self.parser.parse(str(result))
        except Exception as e:
            logger.error(f"Context relevance analysis failed: {str(e)}")
            return ContextRelevanceAnalysis(
                is_llm_related=False,
                relevance_score=0.0,
                llm_indicators=[],
                context_quality="unknown",
                requires_expert_review=False
            )


class FinalDecisionSynthesizerAgent:
    """Agent that synthesizes all analyses into a final screening decision."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=FinalDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content="{analysis_summary}")
        ])
        self.chain = LLMChain(llm=llm, prompt=self.prompt,
                              output_parser=self.parser)

    def _get_system_prompt(self) -> str:
        return AgenticScreeningPrompts.get_final_decision_synthesizer_prompt()

    async def synthesize(
        self,
        contract_analysis: ContractViolationAnalysis,
        technical_analysis: TechnicalErrorAnalysis,
        relevance_analysis: ContextRelevanceAnalysis,
        post_metadata: Dict = None
    ) -> FinalDecision:
        """Synthesize all analyses into final decision."""

        # Convert analyses to dictionaries for better serialization
        contract_data = contract_analysis.model_dump() if hasattr(
            contract_analysis, 'model_dump') else contract_analysis.__dict__
        technical_data = technical_analysis.model_dump() if hasattr(
            technical_analysis, 'model_dump') else technical_analysis.__dict__
        relevance_data = relevance_analysis.model_dump() if hasattr(
            relevance_analysis, 'model_dump') else relevance_analysis.__dict__

        analysis_summary = f"""
CONTRACT VIOLATION ANALYSIS:
- Has Violation: {contract_data.get('has_violation', False)}
- Violation Type: {contract_data.get('violation_type', 'None')}
- Confidence: {contract_data.get('confidence', 0.0)}
- Evidence: {contract_data.get('evidence', [])}
- Severity: {contract_data.get('violation_severity', 'unknown')}

TECHNICAL ERROR ANALYSIS:
- Is Technical Error: {technical_data.get('is_technical_error', False)}
- Error Category: {technical_data.get('error_category', 'None')}
- Root Cause: {technical_data.get('root_cause', 'None')}
- API Related: {technical_data.get('api_related', False)}
- Confidence: {technical_data.get('confidence', 0.0)}
- Error Patterns: {technical_data.get('error_patterns', [])}

CONTEXT RELEVANCE ANALYSIS:
- Is LLM Related: {relevance_data.get('is_llm_related', False)}
- Relevance Score: {relevance_data.get('relevance_score', 0.0)}
- LLM Indicators: {relevance_data.get('llm_indicators', [])}
- Context Quality: {relevance_data.get('context_quality', 'unknown')}
- Requires Expert Review: {relevance_data.get('requires_expert_review', False)}

POST METADATA: {post_metadata if post_metadata else 'None'}
"""

        try:
            result = await self.chain.arun(analysis_summary=analysis_summary)

            # Try to parse JSON response
            try:
                import json

                # Check if result is already parsed
                if isinstance(result, FinalDecision):
                    return result
                elif isinstance(result, dict):
                    return FinalDecision(**result)
                elif isinstance(result, str):
                    # Clean up the result string
                    result = result.strip()
                    if not result.startswith('{'):
                        # Try to extract JSON from the response
                        start = result.find('{')
                        end = result.rfind('}') + 1
                        if start != -1 and end != 0:
                            result = result[start:end]

                    parsed_result = json.loads(result)
                    return FinalDecision(**parsed_result)
                else:
                    # Try the original parser as fallback
                    return self.parser.parse(str(result))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {e}. Raw response: {str(result)[:200]}...")
                # Try the original parser as fallback
                return self.parser.parse(str(result))
        except Exception as e:
            logger.error(f"Final decision synthesis failed: {str(e)}")
            # Fallback decision logic
            return self._fallback_decision(contract_analysis, technical_analysis, relevance_analysis)

    def _fallback_decision(
        self,
        contract_analysis: ContractViolationAnalysis,
        technical_analysis: TechnicalErrorAnalysis,
        relevance_analysis: ContextRelevanceAnalysis
    ) -> FinalDecision:
        """Fallback decision logic when LLM synthesis fails."""

        # More nuanced decision logic
        if not relevance_analysis.is_llm_related:
            decision = "N"
            confidence = 0.8
            rationale = "Not LLM-related content"
        elif contract_analysis.has_violation and contract_analysis.confidence > 0.5:
            decision = "Y"
            confidence = min(contract_analysis.confidence +
                             0.1, 1.0)  # Slightly boost confidence
            rationale = f"Contract violation detected: {contract_analysis.violation_type or 'unspecified'}"
        elif technical_analysis.is_technical_error and technical_analysis.api_related:
            decision = "Y"
            confidence = min(technical_analysis.confidence +
                             0.1, 1.0)  # Slightly boost confidence
            rationale = f"API-related technical error: {technical_analysis.error_category or 'unspecified'}"
        elif contract_analysis.has_violation and contract_analysis.confidence > 0.3:
            decision = "Borderline"
            confidence = 0.6
            rationale = f"Potential contract violation needs review: {contract_analysis.violation_type or 'unspecified'}"
        elif technical_analysis.is_technical_error:
            decision = "Borderline"
            confidence = 0.5
            rationale = f"Technical error may be API-related: {technical_analysis.error_category or 'unspecified'}"
        elif relevance_analysis.requires_expert_review:
            decision = "Borderline"
            confidence = 0.5
            rationale = "Content requires expert review for API relevance"
        elif relevance_analysis.relevance_score > 0.7:
            decision = "Borderline"
            confidence = 0.4
            rationale = "High relevance but unclear contract violations"
        else:
            decision = "N"
            confidence = 0.6
            rationale = "Insufficient evidence for inclusion"

        # Identify contract types if violations found
        contract_types = []
        if contract_analysis.has_violation and contract_analysis.violation_type:
            contract_types.append(contract_analysis.violation_type)

        return FinalDecision(
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            contract_types_identified=contract_types,
            recommended_action="Standard processing",
            quality_flags=["Fallback decision used"]
        )


class AgenticScreeningOrchestrator:
    """Orchestrates the multi-agent screening pipeline."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        db_manager: MongoDBManager,
        base_url: str = None
    ):
        """Initialize the agentic screening orchestrator."""
        self.db = db_manager
        self.provenance = ProvenanceTracker(db_manager)

        # Initialize LLM
        self.llm = CustomLLM(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url
        )

        # Initialize agents
        self.contract_agent = ContractViolationDetectorAgent(self.llm)
        self.technical_agent = TechnicalErrorAnalystAgent(self.llm)
        self.relevance_agent = ContextRelevanceJudgeAgent(self.llm)
        self.decision_agent = FinalDecisionSynthesizerAgent(self.llm)

    async def screen_post(
        self,
        filtered_post: Dict[str, Any],
        include_metadata: bool = True
    ) -> Tuple[LLMScreeningResult, Dict[str, AgentResult]]:
        """Screen a single post using all agents."""
        start_time = datetime.utcnow()

        # Get original post content
        raw_post = await self.db.find_one('raw_posts', {'_id': filtered_post['raw_post_id']})
        if not raw_post:
            raise ValueError(
                f"Raw post not found for filtered post {filtered_post['_id']}")

        title = raw_post.get('title', '')
        content = raw_post.get('body_md', '') or raw_post.get(
            'body', '') or raw_post.get('content', '')

        # Ensure we have content to analyze
        if not content and not title:
            logger.warning(f"No content found for post {filtered_post['_id']}")
            content = "No content available for analysis"

        # Prepare metadata with proper datetime handling
        metadata = {}
        if include_metadata:
            # Convert datetime objects to strings for JSON serialization
            metadata = {
                'platform': raw_post.get('platform'),
                'tags': raw_post.get('tags', []),
                'labels': raw_post.get('labels', []),
                'score': raw_post.get('score', 0),
                'created_at': raw_post.get('created_at').isoformat() if raw_post.get('created_at') else None,
                'filter_confidence': filtered_post.get('filter_confidence', 0.0),
                'matched_keywords': filtered_post.get('matched_keywords', [])
            }
            # Remove None values to avoid JSON issues
            metadata = {k: v for k, v in metadata.items() if v is not None}

        logger.info(
            f"Screening post: '{title[:50]}...' with content length: {len(content)}")

        # Run all agents in parallel
        agent_tasks = [
            self._run_agent_with_timing("contract_detector",
                                        self.contract_agent.analyze, content, title),
            self._run_agent_with_timing("technical_analyst",
                                        self.technical_agent.analyze, content, title),
            self._run_agent_with_timing("relevance_judge",
                                        self.relevance_agent.analyze, content, title, metadata),
        ]

        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Process agent results
        contract_result, technical_result, relevance_result = agent_results

        # Handle any failed agents
        for i, result in enumerate([contract_result, technical_result, relevance_result]):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {str(result)}")
                # Create fallback result
                agent_results[i] = self._create_fallback_agent_result(i)

        # Extract successful results
        contract_result = agent_results[0] if not isinstance(
            agent_results[0], Exception) else self._create_fallback_agent_result(0)
        technical_result = agent_results[1] if not isinstance(
            agent_results[1], Exception) else self._create_fallback_agent_result(1)
        relevance_result = agent_results[2] if not isinstance(
            agent_results[2], Exception) else self._create_fallback_agent_result(2)

        # Synthesize final decision
        decision_start = datetime.utcnow()
        final_decision = await self.decision_agent.synthesize(
            contract_analysis=contract_result.analysis,
            technical_analysis=technical_result.analysis,
            relevance_analysis=relevance_result.analysis,
            post_metadata=metadata
        )
        decision_time = (datetime.utcnow() - decision_start).total_seconds()

        # Create decision result
        decision_result = AgentResult(
            agent_name="decision_synthesizer",
            analysis=final_decision,
            processing_time=decision_time
        )

        # Create final screening result
        screening_result = LLMScreeningResult(
            decision=final_decision.decision,
            rationale=final_decision.rationale,
            confidence=final_decision.confidence,
            model_used=f"agentic_pipeline_{self.llm.model_name}"
        )

        total_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Agentic screening completed in {total_time:.2f}s")

        # Return results with detailed agent analyses
        detailed_results = {
            'contract_detector': contract_result,
            'technical_analyst': technical_result,
            'relevance_judge': relevance_result,
            'decision_synthesizer': decision_result,
            'total_processing_time': total_time
        }

        return screening_result, detailed_results

    async def _run_agent_with_timing(
        self,
        agent_name: str,
        agent_method,
        *args,
        **kwargs
    ) -> AgentResult:
        """Run an agent method with timing and error handling."""
        start_time = datetime.utcnow()

        try:
            analysis = await agent_method(*args, **kwargs)
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return AgentResult(
                agent_name=agent_name,
                analysis=analysis,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Agent {agent_name} failed: {str(e)}")

            return AgentResult(
                agent_name=agent_name,
                analysis=self._create_fallback_analysis(agent_name),
                processing_time=processing_time,
                errors=[str(e)]
            )

    def _create_fallback_analysis(self, agent_name: str) -> BaseModel:
        """Create fallback analysis for failed agents."""
        if agent_name == "contract_detector":
            return ContractViolationAnalysis(
                has_violation=False,
                violation_type=None,
                confidence=0.0,
                evidence=["Agent failed"],
                violation_severity="unknown"
            )
        elif agent_name == "technical_analyst":
            return TechnicalErrorAnalysis(
                is_technical_error=False,
                error_category=None,
                root_cause=None,
                api_related=False,
                confidence=0.0,
                error_patterns=["Agent failed"]
            )
        elif agent_name == "relevance_judge":
            return ContextRelevanceAnalysis(
                is_llm_related=False,
                relevance_score=0.0,
                llm_indicators=["Agent failed"],
                context_quality="unknown",
                requires_expert_review=True
            )
        else:
            return FinalDecision(
                decision="Borderline",
                confidence=0.0,
                rationale="Agent failure",
                contract_types_identified=[],
                recommended_action="Manual review required",
                quality_flags=["Agent failure"]
            )

    def _create_fallback_agent_result(self, agent_index: int) -> AgentResult:
        """Create fallback agent result for failed agents."""
        agent_names = ["contract_detector",
                       "technical_analyst", "relevance_judge"]
        agent_name = agent_names[agent_index] if agent_index < len(
            agent_names) else "unknown"

        return AgentResult(
            agent_name=agent_name,
            analysis=self._create_fallback_analysis(agent_name),
            processing_time=0.0,
            errors=["Agent execution failed"]
        )

    async def screen_batch(
        self,
        batch_size: int = 50,
        save_detailed_results: bool = True
    ) -> Dict[str, Any]:
        """Screen a batch of filtered posts using the agentic pipeline."""

        batch_stats = {
            'processed': 0,
            'positive_decisions': 0,
            'negative_decisions': 0,
            'borderline_cases': 0,
            'high_confidence': 0,
            'agent_performance': {},
            'processing_time': 0,
            'errors': 0
        }

        start_time = datetime.utcnow()

        # Get posts for screening
        posts_to_screen = []
        async for filtered_post in self.db.get_posts_for_screening(batch_size, "agentic_screening"):
            posts_to_screen.append(filtered_post)

        logger.info(
            f"Starting agentic screening of {len(posts_to_screen)} posts")

        # Process posts
        for post in posts_to_screen:
            try:
                screening_result, detailed_results = await self.screen_post(post)

                # Update statistics
                batch_stats['processed'] += 1

                if screening_result.decision == 'Y':
                    batch_stats['positive_decisions'] += 1
                elif screening_result.decision == 'N':
                    batch_stats['negative_decisions'] += 1
                else:
                    batch_stats['borderline_cases'] += 1

                if screening_result.confidence >= 0.8:
                    batch_stats['high_confidence'] += 1

                # Save results
                if save_detailed_results:
                    await self._save_agentic_screening_result(
                        post, screening_result, detailed_results
                    )

                # Track agent performance
                for agent_name, agent_result in detailed_results.items():
                    if agent_name not in batch_stats['agent_performance']:
                        batch_stats['agent_performance'][agent_name] = {
                            'total_time': 0.0,
                            'error_count': 0,
                            'success_count': 0
                        }

                    perf = batch_stats['agent_performance'][agent_name]
                    perf['total_time'] += getattr(agent_result,
                                                  'processing_time', 0.0)

                    if hasattr(agent_result, 'errors') and agent_result.errors:
                        perf['error_count'] += 1
                    else:
                        perf['success_count'] += 1

            except Exception as e:
                logger.error(
                    f"Error screening post {post.get('_id')}: {str(e)}")
                batch_stats['errors'] += 1

        batch_stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Agentic screening completed: {batch_stats}")
        return batch_stats

    async def _save_agentic_screening_result(
        self,
        filtered_post: Dict[str, Any],
        screening_result: LLMScreeningResult,
        detailed_results: Dict[str, AgentResult]
    ) -> None:
        """Save detailed agentic screening results."""

        # Convert detailed results to saveable format
        detailed_data = {}
        for agent_name, agent_result in detailed_results.items():
            if hasattr(agent_result, 'analysis'):
                # Use model_dump() for Pydantic models, dict conversion for others
                if hasattr(agent_result.analysis, 'model_dump'):
                    analysis_data = agent_result.analysis.model_dump()
                elif hasattr(agent_result.analysis, '__dict__'):
                    analysis_data = agent_result.analysis.__dict__
                else:
                    analysis_data = str(agent_result.analysis)

                detailed_data[agent_name] = {
                    'analysis': analysis_data,
                    'processing_time': agent_result.processing_time,
                    'errors': agent_result.errors
                }

        # Save to a dedicated collection for agentic results
        agentic_result = {
            'filtered_post_id': str(filtered_post['_id']),
            'screening_result': screening_result.to_dict(),
            'detailed_agent_analyses': detailed_data,
            'timestamp': datetime.utcnow(),
            'pipeline_version': 'agentic_v1.0'
        }

        await self.db.insert_one('agentic_screening_results', agentic_result)

        # Log provenance
        await self.provenance.log_transformation(
            source_id=str(filtered_post['_id']),
            source_collection='filtered_posts',
            target_id=str(agentic_result.get('_id', 'unknown')),
            target_collection='agentic_screening_results',
            transformation_type='agentic_screening',
            metadata={'agent_count': len(detailed_results)}
        )
