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

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class ContractViolationAnalysis(BaseModel):
    """Structured analysis of contract violations."""
    has_violation: bool = Field(
        description="Whether a contract violation is present")
    violation_type: Optional[str] = Field(
        description="Type of contract violation")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    evidence: List[str] = Field(description="Evidence supporting the analysis")
    violation_severity: str = Field(
        description="Severity: low, medium, high, critical")


class TechnicalErrorAnalysis(BaseModel):
    """Structured analysis of technical errors."""
    is_technical_error: bool = Field(
        description="Whether technical error is present")
    error_category: Optional[str] = Field(description="Category of error")
    root_cause: Optional[str] = Field(description="Likely root cause")
    api_related: bool = Field(description="Whether error is API-related")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    error_patterns: List[str] = Field(description="Identified error patterns")


class ContextRelevanceAnalysis(BaseModel):
    """Structured analysis of context relevance."""
    is_llm_related: bool = Field(description="Whether content is LLM-related")
    relevance_score: float = Field(description="Relevance score 0.0-1.0")
    llm_indicators: List[str] = Field(
        description="LLM-related indicators found")
    context_quality: str = Field(
        description="Quality: poor, fair, good, excellent")
    requires_expert_review: bool = Field(
        description="Whether expert review is needed")


class FinalDecision(BaseModel):
    """Final screening decision."""
    decision: str = Field(description="Y, N, or Borderline")
    confidence: float = Field(description="Overall confidence 0.0-1.0")
    rationale: str = Field(description="Decision rationale")
    contract_types_identified: List[str] = Field(
        description="Contract types found")
    recommended_action: str = Field(description="Recommended next action")
    quality_flags: List[str] = Field(description="Quality flags or concerns")


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

    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

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
        # Implementation would depend on your preferred API
        # This is a placeholder for the actual API call
        return self._make_api_call(prompt, **kwargs)

    def _make_api_call(self, prompt: str, **kwargs) -> str:
        """Make actual API call - implement based on your provider."""
        # Placeholder implementation
        import httpx

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        # Return mock response for now - replace with actual API call
        return "Mock response - implement actual API call"


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
        return """You are an expert LLM API contract violation detector. Your task is to analyze posts for explicit or implicit API usage contract violations.

CONTRACT TYPES TO DETECT:

1. PARAMETER CONSTRAINTS:
   - max_tokens: Must be positive integer, within model limits
   - temperature: Must be 0.0-2.0 (typically 0.0-1.0)
   - top_p: Must be 0.0-1.0, mutually exclusive with temperature in some APIs
   - frequency_penalty: Must be -2.0 to 2.0
   - presence_penalty: Must be -2.0 to 2.0

2. RATE LIMITING VIOLATIONS:
   - Requests per minute (RPM) exceeded
   - Tokens per minute (TPM) exceeded  
   - Quota limitations
   - Billing/usage limits

3. INPUT FORMAT VIOLATIONS:
   - JSON schema validation failures
   - Function calling format errors
   - Message format violations
   - Encoding issues

4. CONTEXT LENGTH VIOLATIONS:
   - Token count exceeding model limits
   - Context window overflow
   - Prompt + completion length issues

5. AUTHENTICATION/AUTHORIZATION:
   - Invalid API keys
   - Insufficient permissions
   - Expired tokens
   - Billing issues

6. RESPONSE FORMAT VIOLATIONS:
   - Expected JSON but got text
   - Schema validation failures
   - Missing required fields
   - Type mismatches

VIOLATION INDICATORS:
- Error messages with specific codes (400, 401, 403, 429, 500)
- Parameter validation failures
- "Invalid request" messages
- Rate limit exceeded notifications
- Context length error messages
- Authentication failures
- Schema validation errors

ANALYZE THE FOLLOWING aspects:
1. Identify specific contract violations
2. Assess violation severity
3. Provide evidence from the text
4. Rate confidence in your analysis

Return structured analysis as JSON matching the ContractViolationAnalysis schema.

Format your response as valid JSON only."""

    async def analyze(self, post_content: str, title: str = "") -> ContractViolationAnalysis:
        """Analyze content for contract violations."""
        input_text = f"Title: {title}\n\nContent: {post_content}"

        try:
            result = await self.chain.arun(input_text=input_text)
            return self.parser.parse(result)
        except Exception as e:
            logger.error(f"Contract violation analysis failed: {str(e)}")
            # Return fallback analysis
            return ContractViolationAnalysis(
                has_violation=False,
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
        return """You are a technical error analysis expert specializing in LLM API integration issues. Analyze posts for technical errors, their root causes, and API relationships.

ERROR CATEGORIES TO IDENTIFY:

1. API CONNECTION ERRORS:
   - Network timeouts
   - SSL/TLS handshake failures
   - DNS resolution issues
   - Connection refused
   - Service unavailable

2. AUTHENTICATION ERRORS:
   - Invalid API key format
   - Expired authentication tokens
   - Insufficient permissions
   - Billing account issues
   - Region/endpoint restrictions

3. REQUEST FORMATTING ERRORS:
   - Malformed JSON payloads
   - Missing required headers
   - Incorrect Content-Type
   - Invalid parameter types
   - Encoding issues (UTF-8, etc.)

4. PARAMETER VALIDATION ERRORS:
   - Out-of-range values
   - Type mismatches
   - Missing required parameters
   - Conflicting parameter combinations
   - Invalid enum values

5. RATE LIMITING ERRORS:
   - Too many requests per second/minute
   - Token rate limits exceeded
   - Burst rate limit violations
   - Concurrent request limits

6. RESPONSE PROCESSING ERRORS:
   - JSON parsing failures
   - Unexpected response format
   - Missing expected fields
   - Type conversion errors
   - Encoding/decoding issues

7. MODEL-SPECIFIC ERRORS:
   - Model not found/available
   - Model deprecation
   - Regional availability issues
   - Model capability limitations

ROOT CAUSE ANALYSIS:
- Trace errors to configuration issues
- Identify library/SDK version problems
- Detect environment setup problems
- Assess code implementation issues
- Evaluate API usage patterns

TECHNICAL INDICATORS:
- Stack traces and error messages
- HTTP status codes
- Library/framework error patterns
- Configuration snippets
- Code examples with issues

Analyze the technical aspects and provide structured output as JSON."""

    async def analyze(self, post_content: str, title: str = "") -> TechnicalErrorAnalysis:
        """Analyze content for technical errors."""
        input_text = f"Title: {title}\n\nContent: {post_content}"

        try:
            result = await self.chain.arun(input_text=input_text)
            return self.parser.parse(result)
        except Exception as e:
            logger.error(f"Technical error analysis failed: {str(e)}")
            return TechnicalErrorAnalysis(
                is_technical_error=False,
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
        return """You are a context relevance judge specializing in LLM-related content analysis. Evaluate whether content is relevant to LLM API research and assess its quality.

LLM RELEVANCE INDICATORS:

1. DIRECT LLM MENTIONS:
   - Specific models: GPT-3, GPT-4, Claude, PaLM, LLaMA
   - API providers: OpenAI, Anthropic, Google AI, Cohere
   - Libraries: langchain, transformers, openai-python

2. API-SPECIFIC TERMINOLOGY:
   - Completions, embeddings, fine-tuning
   - Prompts, tokens, context windows
   - Temperature, top-p, max_tokens
   - Function calling, tool use
   - JSON mode, schema validation

3. TECHNICAL CONCEPTS:
   - Prompt engineering
   - Context length limitations
   - Rate limiting and quotas  
   - Model capabilities and limitations
   - API integration patterns

4. ERROR PATTERNS:
   - API-specific error codes
   - LLM provider error messages
   - Authentication issues
   - Usage limit violations

QUALITY ASSESSMENT CRITERIA:

1. CONTENT DEPTH:
   - Superficial mention vs detailed discussion
   - Technical depth and specificity
   - Problem description clarity
   - Solution attempts documented

2. RESEARCH VALUE:
   - Novel contract violation patterns
   - Clear error reproduction steps
   - Community-validated solutions
   - Edge case documentation

3. SIGNAL vs NOISE:
   - Actionable technical content
   - Specific implementation details
   - Clear problem-solution mapping
   - Minimal off-topic content

4. EXPERT REVIEW INDICATORS:
   - Complex multi-factor issues
   - Novel integration patterns
   - Unclear error causation
   - Conflicting information

Evaluate content relevance and quality for LLM contract research."""

    async def analyze(self, post_content: str, title: str = "", metadata: Dict = None) -> ContextRelevanceAnalysis:
        """Analyze content relevance and quality."""
        meta_info = f"Metadata: {json.dumps(metadata)}" if metadata else ""
        input_text = f"Title: {title}\n\nContent: {post_content}\n\n{meta_info}"

        try:
            result = await self.chain.arun(input_text=input_text)
            return self.parser.parse(result)
        except Exception as e:
            logger.error(f"Context relevance analysis failed: {str(e)}")
            return ContextRelevanceAnalysis(
                is_llm_related=False,
                relevance_score=0.0,
                llm_indicators=[],
                context_quality="poor",
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
        return """You are the final decision synthesizer for LLM contract violation screening. Integrate multiple agent analyses to make the final screening decision.

DECISION CRITERIA:

POSITIVE (Y) - Include in research dataset:
1. Clear LLM API contract violations with evidence
2. Technical errors directly related to API usage
3. High-quality discussions of API limitations
4. Novel contract violation patterns
5. Well-documented integration issues

NEGATIVE (N) - Exclude from research dataset:
1. No LLM API relevance
2. Generic programming questions
3. Installation/environment issues unrelated to APIs
4. Conceptual discussions without practical API usage
5. Off-topic content

BORDERLINE - Requires expert review:
1. Ambiguous contract violation evidence
2. Complex multi-factor issues
3. Novel integration patterns needing validation
4. Contradictory analysis results
5. Low confidence from multiple agents

CONFIDENCE CALCULATION:
- Weight agent confidences by reliability
- Consider evidence consistency across agents
- Account for analysis depth and specificity
- Factor in content quality indicators

SYNTHESIS RULES:
1. Contract violation evidence trumps other factors
2. Technical API errors are high value
3. Context relevance is a prerequisite
4. Quality gates prevent noise inclusion
5. Novel patterns may warrant borderline classification

Provide final structured decision with clear rationale."""

    async def synthesize(
        self,
        contract_analysis: ContractViolationAnalysis,
        technical_analysis: TechnicalErrorAnalysis,
        relevance_analysis: ContextRelevanceAnalysis,
        post_metadata: Dict = None
    ) -> FinalDecision:
        """Synthesize all analyses into final decision."""

        analysis_summary = f"""
CONTRACT VIOLATION ANALYSIS:
- Has Violation: {contract_analysis.has_violation}
- Violation Type: {contract_analysis.violation_type}
- Confidence: {contract_analysis.confidence}
- Evidence: {contract_analysis.evidence}
- Severity: {contract_analysis.violation_severity}

TECHNICAL ERROR ANALYSIS:
- Is Technical Error: {technical_analysis.is_technical_error}
- Error Category: {technical_analysis.error_category}
- Root Cause: {technical_analysis.root_cause}
- API Related: {technical_analysis.api_related}
- Confidence: {technical_analysis.confidence}
- Error Patterns: {technical_analysis.error_patterns}

CONTEXT RELEVANCE ANALYSIS:
- Is LLM Related: {relevance_analysis.is_llm_related}
- Relevance Score: {relevance_analysis.relevance_score}
- LLM Indicators: {relevance_analysis.llm_indicators}
- Context Quality: {relevance_analysis.context_quality}
- Requires Expert Review: {relevance_analysis.requires_expert_review}

POST METADATA: {json.dumps(post_metadata) if post_metadata else 'None'}
"""

        try:
            result = await self.chain.arun(analysis_summary=analysis_summary)
            return self.parser.parse(result)
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

        # Simple decision logic
        if not relevance_analysis.is_llm_related:
            decision = "N"
            confidence = 0.8
            rationale = "Not LLM-related content"
        elif contract_analysis.has_violation and contract_analysis.confidence > 0.7:
            decision = "Y"
            confidence = contract_analysis.confidence
            rationale = f"Clear contract violation: {contract_analysis.violation_type}"
        elif technical_analysis.is_technical_error and technical_analysis.api_related:
            decision = "Y"
            confidence = technical_analysis.confidence
            rationale = f"API-related technical error: {technical_analysis.error_category}"
        elif relevance_analysis.requires_expert_review:
            decision = "Borderline"
            confidence = 0.5
            rationale = "Requires expert review"
        else:
            decision = "N"
            confidence = 0.6
            rationale = "Insufficient evidence for inclusion"

        return FinalDecision(
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            contract_types_identified=[],
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
        content = raw_post.get('body_md', '')

        # Prepare metadata
        metadata = {
            'platform': raw_post.get('platform'),
            'tags': raw_post.get('tags', []),
            'labels': raw_post.get('labels', []),
            'score': raw_post.get('score', 0),
            'created_at': raw_post.get('created_at'),
            'filter_confidence': filtered_post.get('filter_confidence', 0.0),
            'matched_keywords': filtered_post.get('matched_keywords', [])
        } if include_metadata else {}

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
                confidence=0.0,
                evidence=["Agent failed"],
                violation_severity="unknown"
            )
        elif agent_name == "technical_analyst":
            return TechnicalErrorAnalysis(
                is_technical_error=False,
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
        async for filtered_post in self.db.get_posts_for_labelling("agentic_screening", batch_size):
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
                detailed_data[agent_name] = {
                    'analysis': asdict(agent_result.analysis) if hasattr(agent_result.analysis, '__dict__') else str(agent_result.analysis),
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
