"""
Borderline Screening Prompts for LLM Contract Violation Detection.

Specialized prompts for re-evaluating posts with uncertain confidence scores.
Focuses on detailed analysis and expert-level evaluation of edge cases.
"""


class BorderlineScreeningPrompts:
    """Enhanced borderline screening prompts for detailed analysis."""

    @staticmethod
    def get_borderline_screening_prompt() -> str:
        """Get the main borderline screening prompt for expert-level evaluation."""
        return """You are a senior researcher conducting expert-level analysis of potentially borderline LLM API contract violation cases. This post was flagged as uncertain by initial screening and requires detailed evaluation.

EXPERT ANALYSIS FRAMEWORK:

## EVIDENCE HIERARCHY (Rate evidence strength 1-5):

### Level 5 Evidence (Definitive Violations):
- Direct API error responses with specific codes and messages
- Exact parameter values shown to violate documented limits
- Complete code examples demonstrating contract violations
- Step-by-step reproduction of contract violation scenarios
- Official error documentation references

### Level 4 Evidence (Strong Violations):
- Clear error descriptions matching known contract patterns
- Specific technical details about failed API calls
- Well-documented workarounds for contract limitations
- Multiple community confirmations of the same issue
- Integration framework errors caused by API contracts

### Level 3 Evidence (Moderate Violations):  
- Technical discussions of API limitations with some specifics
- Partial code examples showing potential contract issues
- Error messages without full context but clear API relation
- Community discussions of best practices to avoid violations
- Performance issues clearly tied to contract parameters

### Level 2 Evidence (Weak Violations):
- Vague references to API problems without specifics
- General discussions of API behavior without clear violations
- Incomplete error descriptions that might be contract-related
- Speculation about API limitations without evidence
- Mixed technical discussions with some relevant content

### Level 1 Evidence (Insufficient):
- No specific technical details about API usage
- General programming questions without API focus
- Installation/setup issues unrelated to API contracts
- Conceptual discussions without practical implementation
- Off-topic content with minimal API relevance

## DETAILED CONTRACT ANALYSIS:

Examine the post for these specific patterns based on your research taxonomy:

### Parameter Constraint Violations:
- Are specific parameter names and values mentioned?
- Do the values fall outside documented API limits?
- Is there evidence of parameter validation errors?
- Are there discussions of parameter combinations that fail?

### Rate Limiting Violations:
- Are HTTP status codes 429 or rate limit messages mentioned?
- Is there evidence of request frequency issues?
- Are there discussions of quota or billing limitations?
- Is there mention of throttling or backoff strategies?

### Content Policy Violations:
- Are safety filters or content moderation mentioned?
- Is there evidence of prompt rejection or output filtering?
- Are there discussions of content policy compliance?
- Is there mention of harmful content detection?

### Format Violations:
- Are JSON parsing errors or schema validation failures mentioned?
- Is there evidence of message format issues?
- Are function calling or tool use problems described?
- Is there mention of response format mismatches?

### Context Length Violations:
- Are token limits or context length errors mentioned?
- Is there evidence of conversation truncation issues?
- Are there discussions of prompt size optimization?
- Is there mention of model-specific context limits?

### Authentication Violations:
- Are API key errors or authentication failures mentioned?
- Is there evidence of permission or access issues?
- Are there discussions of billing or subscription problems?
- Is there mention of organization-level restrictions?

## QUALITY ASSESSMENT CRITERIA:

### Technical Depth (1-5):
- Specificity of technical details provided
- Accuracy of API usage understanding
- Completeness of problem description
- Quality of proposed solutions or workarounds

### Research Value (1-5):
- Novelty of the contract violation pattern
- Potential to inform better API design or documentation
- Educational value for other developers
- Contribution to understanding of API contracts

### Community Validation (1-5):
- Number and quality of community responses
- Confirmation from multiple independent sources
- Upvotes, acceptance, or other quality indicators
- Expert validation or official responses

POST FOR EXPERT ANALYSIS:

Title: {title}

Content (includes original post and comments): {content}

Previous Analysis Summary:
{previous_analysis}

DETAILED EXPERT EVALUATION:

**IMPORTANT**: This content includes both the original post and community comments. Pay special attention to:
- Comments that provide additional technical context or solutions
- Community responses that confirm or clarify the issue
- Expert feedback that validates or disputes the reported problem
- Follow-up discussions that reveal the actual cause or resolution

Based on your expert analysis, provide a comprehensive assessment:

1. Evidence Level (1-5): Rate the strength of contract violation evidence
2. Contract Categories: Identify all applicable contract violation types
3. Technical Quality: Assess the technical depth and accuracy
4. Research Value: Evaluate potential contribution to API contract research
5. Confidence Factors: List factors that increase or decrease confidence
6. Edge Case Analysis: Identify any unusual or novel aspects
7. Recommendation: Final decision with detailed justification

Respond in this structured format:
DECISION: [Y/N/Borderline]
CONFIDENCE: [0.0-1.0]
EVIDENCE_LEVEL: [1-5]
CONTRACT_CATEGORIES: [List specific types identified]
TECHNICAL_QUALITY: [1-5 with explanation]
RESEARCH_VALUE: [1-5 with explanation]
CONFIDENCE_FACTORS: [Positive and negative factors]
EDGE_CASE_NOTES: [Any unusual patterns or novel aspects]
RATIONALE: [Comprehensive justification for decision]
RECOMMENDED_ACTION: [Next steps for this post]"""

    @staticmethod
    def get_comparative_analysis_prompt() -> str:
        """Get prompt for comparing borderline cases against known examples."""
        return """You are analyzing a borderline case by comparing it to established patterns of LLM API contract violations.

COMPARISON FRAMEWORK:

Compare this post against these validated contract violation patterns:

### Established Positive Patterns:
1. "InvalidRequestError: max_tokens must be ≤ 4096" - Parameter constraint violation
2. "RateLimitError: Rate limit reached for requests" - Rate limiting violation  
3. "This request has been flagged as potentially violating our usage policy" - Content policy violation
4. "JSONDecodeError: Could not parse LLM output" - Format violation
5. "ContextLengthExceededError: Maximum context length exceeded" - Context length violation

### Established Negative Patterns:
1. "How do I install the openai library?" - Installation question
2. "Why is GPT-4 not as accurate as expected?" - Quality complaint
3. "What's the difference between GPT-3 and GPT-4?" - Conceptual question
4. "Can you help me debug this Python error?" - General programming

POST TO COMPARE:
Title: {title}
Content: {content}

COMPARATIVE ANALYSIS:
- How similar is this to established positive patterns?
- What specific similarities/differences do you observe?
- Does this represent a novel pattern worth studying?
- How confident are you in the comparison?

DECISION: [Y/N/Borderline]
CONFIDENCE: [0.0-1.0]
SIMILARITY_SCORE: [0.0-1.0 to established patterns]
RATIONALE: [Detailed comparison analysis]"""

    @staticmethod
    def get_multi_factor_analysis_prompt() -> str:
        """Get prompt for comprehensive multi-factor analysis."""
        return """You are conducting a comprehensive multi-factor analysis of a borderline LLM API contract violation case.

MULTI-FACTOR EVALUATION:

### Factor 1: Technical Specificity
Score: __/10
- Are specific API endpoints, parameters, or error codes mentioned?
- Is there evidence of actual API usage (not just theoretical discussion)?
- How detailed are the technical descriptions?

### Factor 2: Contract Violation Evidence  
Score: __/10
- Is there clear evidence of violating documented API usage rules?
- Are the violations explicitly described or just implied?
- How verifiable are the reported violations?

### Factor 3: Problem-Solution Mapping
Score: __/10
- Is there a clear cause-effect relationship between API usage and problems?
- Are solutions or workarounds provided that relate to contracts?
- How well does the discussion map to known contract types?

### Factor 4: Community Validation
Score: __/10
- How many responses or confirmations are there?
- Is there expert or official validation of the issue?
- What quality indicators (votes, acceptance) are present?

### Factor 5: Research Novelty
Score: __/10
- Does this represent a novel contract violation pattern?
- How much does this contribute to understanding of LLM API contracts?
- Is this a common issue or an edge case worth documenting?

POST FOR MULTI-FACTOR ANALYSIS:
Title: {title}
Content: {content}
Platform: {platform}
Score: {score}
Tags: {tags}

Previous Analysis:
{previous_analysis}

COMPREHENSIVE EVALUATION:

Provide scores for each factor and overall assessment:

FACTOR_SCORES: [Technical:__/10, Evidence:__/10, Mapping:__/10, Validation:__/10, Novelty:__/10]
TOTAL_SCORE: __/50
DECISION: [Y/N/Borderline]
CONFIDENCE: [0.0-1.0]
DETAILED_ANALYSIS: [Factor-by-factor breakdown]
RATIONALE: [Overall justification based on multi-factor analysis]"""
