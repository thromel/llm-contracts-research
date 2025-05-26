#!/usr/bin/env python3
"""
LangChain Agentic LLM Screening Pipeline Demo

Demonstrates the comprehensive multi-agent analysis system for
LLM API contract violation detection and research.

Features:
- 4 specialized agents with domain expertise
- Comprehensive prompting strategies  
- Structured output parsing
- Parallel agent execution
- Detailed provenance tracking
- Quality validation and fallbacks
"""

from pipeline.common.models import RawPost, FilteredPost, Platform
from pipeline.llm_screening.agentic_screener import (
    AgenticScreeningOrchestrator,
    ContractViolationDetectorAgent,
    TechnicalErrorAnalystAgent,
    ContextRelevanceJudgeAgent,
    FinalDecisionSynthesizerAgent,
    CustomLLM,
    ContractViolationAnalysis,
    TechnicalErrorAnalysis,
    ContextRelevanceAnalysis,
    FinalDecision
)
from pipeline.common.database import MongoDBManager
from pipeline.common.config import (
    PipelineConfig,
    get_development_config,
    get_research_config,
    ScreeningMode,
    LLMProvider,
    LLMConfig
)
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import json

# Add the parent directory to the path to import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticScreeningDemo:
    """Demonstration of the agentic screening pipeline."""

    def __init__(self):
        """Initialize the demo."""
        self.config = None
        self.db = None
        self.orchestrator = None

    async def setup(self, use_research_config: bool = True):
        """Setup the demo environment."""
        logger.info("🚀 Setting up Agentic Screening Demo")

        # Load configuration
        if use_research_config:
            self.config = get_research_config()
        else:
            self.config = get_development_config()

        # Validate configuration
        issues = self.config.validate()
        if issues:
            logger.warning(f"Configuration issues: {issues}")
            # For demo, we'll create mock configurations
            await self._setup_mock_config()

        logger.info(
            f"Using screening mode: {self.config.screening_mode.value}")
        logger.info(
            f"Active LLM configs: {list(self.config.get_active_llm_configs().keys())}")

        # Setup database
        self.db = MongoDBManager(
            connection_string=self.config.database.connection_string,
            database_name=self.config.database.database_name
        )

        try:
            await self.db.connect()
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            # For demo purposes, we'll continue with a mock setup
            await self._setup_mock_database()

        # Setup agentic orchestrator
        if self.config.agentic_screening.contract_detector_llm:
            self.orchestrator = AgenticScreeningOrchestrator(
                api_key=self.config.agentic_screening.contract_detector_llm.api_key,
                model_name=self.config.agentic_screening.contract_detector_llm.model_name,
                db_manager=self.db,
                base_url=self.config.agentic_screening.contract_detector_llm.base_url
            )
            logger.info("✅ Agentic orchestrator initialized")
        else:
            logger.warning("⚠️ No agentic LLM configuration found")

    async def _setup_mock_config(self):
        """Setup mock configuration for demo purposes."""
        logger.info("Setting up mock configuration for demo")

        # Create a mock LLM config (this would use a local model or mock API)
        mock_llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,  # Mock provider
            model_name="gpt-4-demo",
            api_key="demo-key-12345",
            base_url="http://localhost:8000/v1",  # Mock local endpoint
            temperature=0.1,
            max_tokens=2000
        )

        # Apply to all agentic agents
        self.config.agentic_screening.contract_detector_llm = mock_llm_config
        self.config.agentic_screening.technical_analyst_llm = mock_llm_config
        self.config.agentic_screening.relevance_judge_llm = mock_llm_config
        self.config.agentic_screening.decision_synthesizer_llm = mock_llm_config

        # Use in-memory database for demo
        self.config.database.connection_string = "mongodb://localhost:27017/"
        self.config.database.database_name = "agentic_demo"

    async def _setup_mock_database(self):
        """Setup mock database for demo purposes."""
        logger.info("Setting up mock database operations")
        # This would implement mock database operations for the demo
        # For now, we'll just log that we're using mock operations
        pass

    async def demo_individual_agents(self):
        """Demonstrate each agent individually."""
        logger.info("🔍 Demonstrating Individual Agents")

        # Sample posts for testing
        sample_posts = self._get_sample_posts()

        for i, (title, content, description) in enumerate(sample_posts):
            logger.info(f"\n📝 Sample Post {i+1}: {description}")
            logger.info(f"Title: {title}")
            logger.info(f"Content: {content[:200]}...")

            if self.orchestrator:
                await self._analyze_with_all_agents(title, content)
            else:
                await self._mock_agent_analysis(title, content)

    async def _analyze_with_all_agents(self, title: str, content: str):
        """Analyze content with all agents."""
        logger.info("Running analysis with all agents...")

        try:
            # Contract Violation Detector
            logger.info("🔍 Contract Violation Detector Analysis:")
            contract_analysis = await self.orchestrator.contract_agent.analyze(content, title)
            self._log_contract_analysis(contract_analysis)

            # Technical Error Analyst
            logger.info("🔧 Technical Error Analyst Analysis:")
            technical_analysis = await self.orchestrator.technical_agent.analyze(content, title)
            self._log_technical_analysis(technical_analysis)

            # Context Relevance Judge
            logger.info("📊 Context Relevance Judge Analysis:")
            relevance_analysis = await self.orchestrator.relevance_agent.analyze(content, title)
            self._log_relevance_analysis(relevance_analysis)

            # Final Decision Synthesizer
            logger.info("⚖️ Final Decision Synthesis:")
            final_decision = await self.orchestrator.decision_agent.synthesize(
                contract_analysis, technical_analysis, relevance_analysis
            )
            self._log_final_decision(final_decision)

        except Exception as e:
            logger.error(f"Agent analysis failed: {str(e)}")
            await self._mock_agent_analysis(title, content)

    async def _mock_agent_analysis(self, title: str, content: str):
        """Mock agent analysis for demo purposes."""
        logger.info("Using mock agent analysis for demo...")

        # Mock Contract Violation Analysis
        logger.info("🔍 Contract Violation Detector Analysis (Mock):")
        mock_contract = ContractViolationAnalysis(
            has_violation=True,
            violation_type="rate_limit_exceeded",
            confidence=0.85,
            evidence=["Rate limit exceeded message", "429 status code"],
            violation_severity="medium"
        )
        self._log_contract_analysis(mock_contract)

        # Mock Technical Error Analysis
        logger.info("🔧 Technical Error Analyst Analysis (Mock):")
        mock_technical = TechnicalErrorAnalysis(
            is_technical_error=True,
            error_category="rate_limiting",
            root_cause="too_many_requests",
            api_related=True,
            confidence=0.90,
            error_patterns=["429 Too Many Requests", "rate_limit_exceeded"]
        )
        self._log_technical_analysis(mock_technical)

        # Mock Context Relevance Analysis
        logger.info("📊 Context Relevance Judge Analysis (Mock):")
        mock_relevance = ContextRelevanceAnalysis(
            is_llm_related=True,
            relevance_score=0.88,
            llm_indicators=["openai", "gpt-4", "api_key"],
            context_quality="good",
            requires_expert_review=False
        )
        self._log_relevance_analysis(mock_relevance)

        # Mock Final Decision
        logger.info("⚖️ Final Decision Synthesis (Mock):")
        mock_decision = FinalDecision(
            decision="Y",
            confidence=0.87,
            rationale="Clear rate limiting contract violation with high-quality technical documentation",
            contract_types_identified=["rate_limit", "authentication"],
            recommended_action="Include in research dataset",
            quality_flags=[]
        )
        self._log_final_decision(mock_decision)

    def _log_contract_analysis(self, analysis: ContractViolationAnalysis):
        """Log contract violation analysis results."""
        logger.info(f"  ├─ Has Violation: {analysis.has_violation}")
        logger.info(f"  ├─ Violation Type: {analysis.violation_type}")
        logger.info(f"  ├─ Confidence: {analysis.confidence:.2f}")
        logger.info(f"  ├─ Severity: {analysis.violation_severity}")
        logger.info(f"  └─ Evidence: {analysis.evidence}")

    def _log_technical_analysis(self, analysis: TechnicalErrorAnalysis):
        """Log technical error analysis results."""
        logger.info(f"  ├─ Is Technical Error: {analysis.is_technical_error}")
        logger.info(f"  ├─ Error Category: {analysis.error_category}")
        logger.info(f"  ├─ Root Cause: {analysis.root_cause}")
        logger.info(f"  ├─ API Related: {analysis.api_related}")
        logger.info(f"  ├─ Confidence: {analysis.confidence:.2f}")
        logger.info(f"  └─ Error Patterns: {analysis.error_patterns}")

    def _log_relevance_analysis(self, analysis: ContextRelevanceAnalysis):
        """Log context relevance analysis results."""
        logger.info(f"  ├─ Is LLM Related: {analysis.is_llm_related}")
        logger.info(f"  ├─ Relevance Score: {analysis.relevance_score:.2f}")
        logger.info(f"  ├─ Context Quality: {analysis.context_quality}")
        logger.info(
            f"  ├─ Requires Expert Review: {analysis.requires_expert_review}")
        logger.info(f"  └─ LLM Indicators: {analysis.llm_indicators}")

    def _log_final_decision(self, decision: FinalDecision):
        """Log final decision results."""
        logger.info(f"  ├─ Decision: {decision.decision}")
        logger.info(f"  ├─ Confidence: {decision.confidence:.2f}")
        logger.info(f"  ├─ Rationale: {decision.rationale}")
        logger.info(
            f"  ├─ Contract Types: {decision.contract_types_identified}")
        logger.info(f"  ├─ Recommended Action: {decision.recommended_action}")
        logger.info(f"  └─ Quality Flags: {decision.quality_flags}")

    def _get_sample_posts(self) -> List[tuple]:
        """Get sample posts for testing."""
        return [
            (
                "OpenAI API Rate Limit Exceeded Error",
                """I'm getting a rate limit error when calling the OpenAI API:

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=150,
    temperature=0.7
)
```

Error: `RateLimitError: You exceeded your current quota, please check your plan and billing details.`

I'm only making a few requests per minute. What could be causing this?""",
                "Rate limiting error with API usage"
            ),
            (
                "JSON Schema Validation Error with Function Calling",
                """Having trouble with OpenAI function calling. My JSON schema keeps failing validation:

```json
{
    "name": "get_weather", 
    "description": "Get weather information",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}
```

Error: `Invalid request: schema validation failed`

The schema looks correct to me. Any ideas?""",
                "JSON schema validation error"
            ),
            (
                "How to install Python",
                """I'm new to programming and want to install Python on my computer. I'm using Windows 10. 
Can someone guide me through the installation process? I've heard about virtual environments 
but I'm not sure what they are or if I need them.

Also, what's the difference between Python 2 and Python 3? Which one should I use?""",
                "Generic programming question (should be filtered out)"
            ),
            (
                "Context Length Exceeded with GPT-4",
                """Getting context length errors with GPT-4:

```
openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens. 
However, your messages resulted in 12543 tokens. Please reduce the length of the messages.
```

I'm trying to process a long document. The input is about 15,000 tokens. 
I know GPT-4 has a context limit, but I thought it was higher than 8K tokens.

Is there a way to chunk the document or use a different approach?""",
                "Context length violation"
            ),
            (
                "Temperature Parameter Out of Range",
                """I'm experimenting with different temperature values for text generation:

```python
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a creative story",
    temperature=2.5,  # This seems to cause issues
    max_tokens=500
)
```

Getting validation error: `temperature must be between 0 and 2`

But I've seen examples online using temperature > 2. What's the actual valid range?""",
                "Parameter constraint violation"
            )
        ]

    async def demo_full_pipeline(self):
        """Demonstrate the full agentic screening pipeline."""
        logger.info("\n🔄 Demonstrating Full Agentic Pipeline")

        if not self.orchestrator:
            logger.warning(
                "⚠️ Orchestrator not available, skipping full pipeline demo")
            return

        # Create mock filtered posts
        mock_posts = self._create_mock_filtered_posts()

        logger.info(
            f"Processing {len(mock_posts)} posts through agentic pipeline...")

        try:
            # Run batch screening
            results = await self.orchestrator.screen_batch(
                batch_size=len(mock_posts),
                save_detailed_results=False  # Don't save to DB for demo
            )

            logger.info("📊 Batch Screening Results:")
            logger.info(f"  ├─ Posts Processed: {results['processed']}")
            logger.info(
                f"  ├─ Positive Decisions: {results['positive_decisions']}")
            logger.info(
                f"  ├─ Negative Decisions: {results['negative_decisions']}")
            logger.info(
                f"  ├─ Borderline Cases: {results['borderline_cases']}")
            logger.info(f"  ├─ High Confidence: {results['high_confidence']}")
            logger.info(
                f"  ├─ Processing Time: {results['processing_time']:.2f}s")
            logger.info(f"  └─ Errors: {results['errors']}")

            # Log agent performance
            if 'agent_performance' in results:
                logger.info("\n🎯 Agent Performance:")
                for agent_name, perf in results['agent_performance'].items():
                    logger.info(f"  {agent_name}:")
                    logger.info(
                        f"    ├─ Total Time: {perf['total_time']:.2f}s")
                    logger.info(
                        f"    ├─ Success Count: {perf['success_count']}")
                    logger.info(f"    └─ Error Count: {perf['error_count']}")

        except Exception as e:
            logger.error(f"Full pipeline demo failed: {str(e)}")

    def _create_mock_filtered_posts(self) -> List[Dict[str, Any]]:
        """Create mock filtered posts for testing."""
        sample_posts = self._get_sample_posts()
        mock_posts = []

        for i, (title, content, description) in enumerate(sample_posts):
            mock_post = {
                '_id': f"mock_filtered_{i}",
                'raw_post_id': f"mock_raw_{i}",
                'passed_keyword_filter': True,
                'matched_keywords': ['openai', 'api', 'error'],
                'filter_confidence': 0.8,
                'relevant_snippets': [content[:200]],
                'potential_contracts': [title]
            }
            mock_posts.append(mock_post)

            # Mock the corresponding raw post
            if hasattr(self, 'mock_raw_posts'):
                self.mock_raw_posts = {}
            self.mock_raw_posts[f"mock_raw_{i}"] = {
                '_id': f"mock_raw_{i}",
                'title': title,
                'body_md': content,
                'platform': 'github',
                'tags': ['api', 'openai'],
                'labels': ['bug'],
                'score': 10,
                'created_at': datetime.utcnow()
            }

        return mock_posts

    async def demo_prompt_engineering(self):
        """Demonstrate the comprehensive prompts used by each agent."""
        logger.info("\n📝 Demonstrating Comprehensive Prompts")

        if not self.orchestrator:
            logger.info("Creating mock agents to show prompts...")
            # Create a mock LLM for demonstration
            mock_llm = CustomLLM("demo-key", "demo-model")

            # Create agents
            contract_agent = ContractViolationDetectorAgent(mock_llm)
            technical_agent = TechnicalErrorAnalystAgent(mock_llm)
            relevance_agent = ContextRelevanceJudgeAgent(mock_llm)
            decision_agent = FinalDecisionSynthesizerAgent(mock_llm)
        else:
            contract_agent = self.orchestrator.contract_agent
            technical_agent = self.orchestrator.technical_agent
            relevance_agent = self.orchestrator.relevance_agent
            decision_agent = self.orchestrator.decision_agent

        # Show prompts for each agent
        logger.info("🔍 Contract Violation Detector Prompt:")
        logger.info(contract_agent._get_system_prompt()[:500] + "...")

        logger.info("\n🔧 Technical Error Analyst Prompt:")
        logger.info(technical_agent._get_system_prompt()[:500] + "...")

        logger.info("\n📊 Context Relevance Judge Prompt:")
        logger.info(relevance_agent._get_system_prompt()[:500] + "...")

        logger.info("\n⚖️ Final Decision Synthesizer Prompt:")
        logger.info(decision_agent._get_system_prompt()[:500] + "...")

    async def demo_configuration_options(self):
        """Demonstrate different configuration options."""
        logger.info("\n⚙️ Configuration Options Demo")

        logger.info("Current Configuration:")
        config_dict = self.config.to_dict()
        for key, value in config_dict.items():
            logger.info(f"  ├─ {key}: {value}")

        logger.info("\nAvailable Screening Modes:")
        for mode in ScreeningMode:
            logger.info(f"  ├─ {mode.value}: {mode.name}")

        logger.info("\nSupported LLM Providers:")
        for provider in LLMProvider:
            logger.info(f"  ├─ {provider.value}: {provider.name}")

    async def cleanup(self):
        """Cleanup demo resources."""
        logger.info("🧹 Cleaning up demo resources")

        if self.db:
            await self.db.disconnect()
            logger.info("✅ Database disconnected")


async def main():
    """Run the agentic screening demo."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║             LangChain Agentic LLM Screening Demo             ║
║                                                              ║
║  Multi-Agent Analysis for LLM API Contract Violations       ║
║  • Contract Violation Detector Agent                        ║
║  • Technical Error Analyst Agent                            ║
║  • Context Relevance Judge Agent                            ║
║  • Final Decision Synthesizer Agent                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    demo = AgenticScreeningDemo()

    try:
        # Setup
        await demo.setup(use_research_config=True)

        # Run demonstrations
        await demo.demo_prompt_engineering()
        await demo.demo_configuration_options()
        await demo.demo_individual_agents()
        await demo.demo_full_pipeline()

        logger.info("\n✅ Demo completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
