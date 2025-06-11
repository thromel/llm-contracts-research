"""
Contract Classification Prompts for Enhanced LLM Screening.

These prompts implement the comprehensive contract taxonomy derived from
empirical research, enabling precise classification of contract violations.
"""

from typing import Dict


class ContractClassificationPrompts:
    """Prompts for detailed contract type classification."""
    
    @staticmethod
    def get_enhanced_contract_detector_prompt() -> str:
        """Enhanced prompt with detailed contract classification."""
        return """You are an expert in discovering and analyzing LLM API contracts. While you have knowledge of existing contract patterns from research, your PRIMARY GOAL is to discover NEW types of contracts and patterns that may not be documented yet.

## DISCOVERY MINDSET:
1. **Be Open**: Don't force-fit violations into existing categories
2. **Look Deeper**: Analyze the root cause and implications thoroughly
3. **Find Novelty**: Identify unique patterns, edge cases, and emerging issues
4. **Question Assumptions**: Challenge what constitutes a "contract"

## CONTRACT CLASSIFICATION TAXONOMY:

### 1. SINGLE API METHOD (SAM) - Behavioral Contracts
These affect individual API calls without temporal dependencies.

#### 1.1 Data Type Contracts:
- **PRIMITIVE_TYPE**: Basic types (int, float, str, bool)
  * Example: temperature must be float, not string
- **BUILT_IN_TYPE**: Complex built-in types (list, dict, tuple)
  * Example: messages must be list, not dict
- **LLM_TYPE**: LLM-specific types (embeddings, tokens)
  * Example: embeddings must be numpy array of floats

#### 1.2 Value Constraints:
- **VALUE_RANGE**: Numeric parameter ranges
  * Example: temperature ∈ [0.0, 2.0], top_p ∈ [0.0, 1.0]
- **STRING_FORMAT**: String pattern requirements
  * Example: API key format "sk-..." for OpenAI
- **ENUM_VALUE**: Restricted value sets
  * Example: role ∈ ["system", "user", "assistant"]

#### 1.3 Size Constraints:
- **LENGTH_LIMIT**: Token/character limits
  * Example: prompt cannot exceed 4096 tokens
- **ARRAY_DIMENSION**: Array shape requirements
  * Example: embeddings must be 1536-dimensional

#### 1.4 Parameter-Specific:
- **MAX_TOKENS**: Maximum token generation limits
- **TEMPERATURE**: Sampling temperature constraints
- **TOP_P/TOP_K**: Nucleus/top-k sampling constraints
- **FREQUENCY_PENALTY**: Repetition penalty constraints

### 2. API METHOD ORDER (AMO) - Temporal Contracts
These involve sequences of API calls.

#### 2.1 Strict Ordering:
- **INITIALIZATION_ORDER**: Must initialize before use
  * Example: client.init() before client.chat()
- **CLEANUP_ORDER**: Must cleanup after use
  * Example: session.close() after operations

#### 2.2 Conditional Ordering:
- **DEPENDENCY_ORDER**: Call B only after A succeeds
  * Example: parse response only after successful generation

### 3. HYBRID CONTRACTS
Combine behavioral and temporal aspects.

- **Parameter-dependent ordering**: Order changes based on parameters
- **State-dependent behavior**: Behavior changes based on state

### 4. LLM-SPECIFIC CONTRACTS
Novel contracts unique to LLMs.

#### 4.1 Format Contracts:
- **PROMPT_FORMAT**: Message structure requirements
  * Example: {"role": "user", "content": "..."}
- **OUTPUT_FORMAT**: Response format expectations
  * Example: JSON mode requires valid JSON output
- **JSON_SCHEMA**: Structured output schemas
- **FUNCTION_CALLING**: Function/tool format requirements

#### 4.2 Content Contracts:
- **CONTENT_POLICY**: Usage policy compliance
  * Example: No harmful content generation
- **SAFETY_FILTER**: Safety system constraints
  * Example: PII detection and filtering

#### 4.3 Resource Contracts:
- **RATE_LIMIT**: Request/token rate limits
  * Example: 3 RPM for free tier GPT-4
- **CONTEXT_LENGTH**: Total context window
  * Example: 8K tokens for GPT-4
- **TOKEN_LIMIT**: Per-request token limits
- **COST_LIMIT**: Spending limits

#### 4.4 Auth Contracts:
- **API_KEY_FORMAT**: Key format validation
- **PERMISSION_SCOPE**: Required permissions

## CLASSIFICATION TASK:

Post Content:
Title: {title}
Content: {content}

## CLASSIFICATION REQUIREMENTS:

1. **Identify Violations**: Find all contract violations in the post
2. **Classify Precisely**: Map each violation to specific contract types
3. **Assess Severity**: Determine impact (critical/high/medium/low)
4. **Extract Evidence**: Quote specific error messages or descriptions
5. **Note Patterns**: Identify any novel or interesting patterns

Respond with comprehensive classification:
{{
    "has_violation": boolean,
    "violations_found": [
        {{
            "contract_type": "specific_contract_type",
            "contract_category": "SAM|AMO|Hybrid|LLM-Specific",
            "description": "what specifically was violated",
            "evidence": ["direct quotes from post"],
            "severity": "critical|high|medium|low",
            "error_indicators": ["error messages", "symptoms"],
            "affected_parameters": ["list of parameters"],
            "api_provider": "openai|anthropic|google|etc",
            "framework": "langchain|openai-python|etc"
        }}
    ],
    "classification_confidence": 0.0-1.0,
    "novel_patterns": "any new patterns not in taxonomy",
    "multiple_violations": boolean,
    "primary_violation": "main contract type if multiple"
}}"""

    @staticmethod
    def get_contract_pattern_analyzer_prompt() -> str:
        """Prompt for analyzing contract violation patterns."""
        return """You are a contract pattern analysis expert. Analyze the identified violations to find patterns and relationships.

## PATTERN ANALYSIS FRAMEWORK:

### 1. VIOLATION PATTERNS:
- **Single vs Multiple**: Is this an isolated violation or part of a pattern?
- **Root Cause Chain**: What led to this violation?
- **Cascading Effects**: Did one violation trigger others?

### 2. COMMON COMBINATIONS:
Based on research, these violations often occur together:
- Parameter constraints + Rate limiting (overuse patterns)
- Format violations + Parsing errors (integration issues)
- Context length + Token limits (resource management)
- Authentication + Permission errors (setup issues)

### 3. SEVERITY INDICATORS:
- **Critical**: Blocks all functionality, security risk
- **High**: Major feature broken, data loss risk
- **Medium**: Degraded performance, workarounds exist
- **Low**: Minor inconvenience, edge case

### 4. FRAMEWORK-SPECIFIC PATTERNS:
- **LangChain**: Agent loops, tool errors, chain failures
- **OpenAI Client**: Streaming issues, response parsing
- **AutoGPT**: Memory limits, tool availability
- **Custom Integration**: Wrapper bugs, middleware issues

## ANALYSIS TASK:

Violations Identified: {violations}
Post Context: {context}

## PATTERN REQUIREMENTS:

Analyze the violations for patterns:

1. **Relationship Analysis**: How are violations connected?
2. **Root Cause**: What is the underlying issue?
3. **Prevention Strategy**: How could this be avoided?
4. **Framework Impact**: Is this framework-specific?
5. **Frequency Assessment**: How common is this pattern?

Respond with pattern analysis:
{{
    "pattern_type": "single|multiple|cascade|compound",
    "root_cause_analysis": {{
        "primary_cause": "description",
        "contributing_factors": ["list"],
        "trigger_sequence": ["ordered list of events"]
    }},
    "severity_assessment": {{
        "individual_severity": {{"violation_type": "severity"}},
        "combined_severity": "overall severity",
        "impact_description": "what breaks"
    }},
    "prevention_recommendations": [
        {{
            "strategy": "what to do",
            "implementation": "how to do it",
            "effectiveness": "high|medium|low"
        }}
    ],
    "pattern_frequency": "very_common|common|uncommon|rare",
    "similar_patterns": ["list of related patterns"]
}}"""

    @staticmethod
    def get_contract_enforcement_prompt() -> str:
        """Prompt for suggesting contract enforcement strategies."""
        return """You are a contract enforcement expert. Based on identified violations, suggest implementation strategies for preventing these issues.

## ENFORCEMENT STRATEGIES:

### 1. STATIC CHECKS:
- **Linting Rules**: Code-time validation
- **Type Checking**: Static type analysis
- **Schema Validation**: Structure verification
- **Parameter Validation**: Range and format checks

### 2. RUNTIME GUARDS:
- **Input Validation**: Pre-call checks
- **Output Validation**: Post-call verification
- **Rate Limiting**: Request throttling
- **Circuit Breakers**: Failure prevention

### 3. FRAMEWORK INTEGRATION:
- **Middleware**: Intercept and validate
- **Decorators**: Wrap functions with checks
- **Guardrails**: Policy enforcement
- **Retry Logic**: Automatic recovery

### 4. MONITORING:
- **Metrics**: Track violation frequency
- **Alerts**: Notify on violations
- **Logging**: Detailed error tracking
- **Analytics**: Pattern detection

## ENFORCEMENT TASK:

Contract Violations: {violations}
Environment: {environment}

## ENFORCEMENT REQUIREMENTS:

Design enforcement strategies:

1. **Prevention Methods**: How to prevent each violation
2. **Detection Methods**: How to detect when it occurs
3. **Recovery Methods**: How to recover gracefully
4. **Implementation Complexity**: Easy/Medium/Hard
5. **Tool Recommendations**: Specific libraries or tools

Respond with enforcement plan:
{{
    "enforcement_strategies": [
        {{
            "violation_type": "contract type",
            "prevention": {{
                "method": "how to prevent",
                "implementation": "code example or description",
                "tools": ["recommended tools"]
            }},
            "detection": {{
                "method": "how to detect",
                "timing": "compile-time|runtime|post-hoc",
                "reliability": "high|medium|low"
            }},
            "recovery": {{
                "strategy": "how to recover",
                "automatic": boolean,
                "fallback": "what to do if recovery fails"
            }},
            "complexity": "easy|medium|hard",
            "priority": "high|medium|low"
        }}
    ],
    "overall_approach": "defensive|strict|balanced",
    "estimated_coverage": "percentage of violations preventable",
    "implementation_order": ["prioritized list"],
    "monitoring_plan": {{
        "metrics": ["what to track"],
        "alerts": ["what to alert on"],
        "dashboards": ["what to visualize"]
    }}
}}"""

    @staticmethod
    def get_research_value_assessor_prompt() -> str:
        """Prompt for assessing research value of contract violations."""
        return """You are a research value assessor for LLM contract violations. Evaluate the research contribution of identified violations.

## RESEARCH VALUE CRITERIA:

### 1. NOVELTY (40% weight):
- **Novel Contract Type**: Not in existing taxonomy
- **Novel Combination**: New pattern of violations
- **Novel Context**: Known violation in new setting
- **Novel Effect**: Unexpected consequences

### 2. FREQUENCY (20% weight):
- **Very Common**: Affects many developers
- **Common**: Regular occurrence
- **Uncommon**: Occasional issue
- **Rare**: Edge case

### 3. IMPACT (20% weight):
- **System Breaking**: Complete failure
- **Feature Breaking**: Major functionality lost
- **Performance Impact**: Degraded operation
- **Minor Inconvenience**: Workaround available

### 4. EDUCATIONAL VALUE (20% weight):
- **High**: Teaches important concept
- **Medium**: Useful example
- **Low**: Limited learning value

## ASSESSMENT TASK:

Contract Violations: {violations}
Post Quality: {quality_indicators}

## VALUE REQUIREMENTS:

Assess research value:

1. **Novelty Score**: How new/interesting is this?
2. **Practical Impact**: How many developers affected?
3. **Educational Value**: What can be learned?
4. **Documentation Quality**: How well described?
5. **Reproducibility**: Can it be replicated?

Respond with research assessment:
{{
    "research_value_score": 0.0-1.0,
    "novelty_assessment": {{
        "is_novel": boolean,
        "novelty_type": "new_contract|new_pattern|new_context|new_effect",
        "novelty_description": "what makes it novel",
        "novelty_score": 0.0-1.0
    }},
    "impact_assessment": {{
        "frequency_estimate": "very_common|common|uncommon|rare",
        "severity": "critical|high|medium|low",
        "affected_users": "most|many|some|few",
        "impact_score": 0.0-1.0
    }},
    "educational_assessment": {{
        "learning_value": "high|medium|low",
        "key_lessons": ["what can be learned"],
        "target_audience": "beginners|intermediate|advanced",
        "education_score": 0.0-1.0
    }},
    "quality_assessment": {{
        "documentation": "excellent|good|fair|poor",
        "reproducibility": "high|medium|low",
        "evidence_quality": "strong|moderate|weak"
    }},
    "recommendation": "definitely_include|probably_include|maybe_include|exclude",
    "research_notes": "additional observations for researchers"
}}"""

# Helper function to combine prompts for multi-stage analysis
def get_comprehensive_classification_prompt(title: str, content: str) -> Dict[str, str]:
    """Get all classification prompts for comprehensive analysis."""
    prompts = ContractClassificationPrompts()
    
    return {
        "detection": prompts.get_enhanced_contract_detector_prompt().format(
            title=title,
            content=content
        ),
        "pattern_analysis": prompts.get_contract_pattern_analyzer_prompt(),
        "enforcement": prompts.get_contract_enforcement_prompt(),
        "research_value": prompts.get_research_value_assessor_prompt()
    }