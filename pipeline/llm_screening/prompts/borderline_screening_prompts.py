"""
Borderline Screening Prompts for LLM Contract Violation Detection.

Specialized prompts for re-evaluating posts with uncertain confidence scores.
Focuses on detailed analysis and expert-level evaluation of edge cases.
"""


class BorderlineScreeningPrompts:
    """Enhanced borderline screening prompts for detailed analysis."""

    @staticmethod
    def get_borderline_screening_prompt() -> str:
        """Get the comprehensive LLM contract violation screening prompt for expert-level evaluation."""
        return """You are a senior researcher specializing in LLM API contract violations conducting comprehensive analysis for academic research. You are analyzing posts from GitHub (closed issues with community discussion) and Stack Overflow (answered questions with community engagement) to identify real-world contract violations between developers and LLM API providers.

## RESEARCH CONTEXT & OBJECTIVES:

**Research Goal**: Systematically identify and categorize LLM API contract violations to understand:
- Common failure patterns in LLM API usage
- Developer pain points and misunderstandings
- API design issues and documentation gaps
- Community solutions and workarounds

**Data Sources**: 
- **GitHub**: Closed issues with comments from LLM provider repositories (OpenAI, Anthropic, LangChain, etc.)
- **Stack Overflow**: Answered questions with comments from LLM-specific tags (openai-api, langchain, etc.)

**Why This Data**: We focus on closed/answered posts with comments because they provide:
- ✅ Confirmed problem-solution pairs
- ✅ Community validation and expert feedback
- ✅ Multiple perspectives on the same issue
- ✅ Resolution context and workarounds

## COMPREHENSIVE ANALYSIS FRAMEWORK:

## EVIDENCE HIERARCHY (Rate evidence strength 1-5):

### Level 5 Evidence (Definitive Contract Violations):
**Examples**: 
- `"InvalidRequestError: max_tokens (5000) exceeds maximum (4096)"`
- `"RateLimitError: Rate limit reached for requests"`
- `"ContextLengthExceededError: This model's maximum context length is 8192 tokens"`

**Criteria**:
- Direct API error responses with specific HTTP codes (400, 429, etc.)
- Exact parameter values violating documented limits
- Complete reproducible code examples
- Official API documentation violations
- Clear cause-effect relationship between input and error

### Level 4 Evidence (Strong Contract Violations):
**Examples**:
- Detailed error descriptions matching documented API limits
- Code examples showing parameter constraint failures
- Multiple community members confirming same violation pattern
- Workarounds that directly address contract limitations

**Criteria**:
- Specific technical details about API call failures
- Clear violation patterns with partial code/parameters
- Community validation of the reported issue
- Error patterns consistent with known API contracts

### Level 3 Evidence (Moderate Contract Violations):  
**Examples**:
- Discussions of API behavior limits with some technical details
- Partial error messages clearly related to API constraints
- Best practice discussions to avoid known violations
- Performance issues tied to specific API parameters

**Criteria**:
- Some technical specifics about API limitations
- Partial context about contract-related issues
- Community discussions of constraint workarounds
- Clear API relationship but incomplete violation evidence

### Level 2 Evidence (Weak/Possible Violations):
**Examples**:
- Vague references to "API not working" with minimal details
- General discussions about API behavior without specifics
- Mixed content with some contract-relevant elements

**Criteria**:
- Limited technical details about API usage
- Unclear relationship between problem and API contracts
- Speculation without concrete evidence
- Tangential contract relevance

### Level 1 Evidence (No Contract Violation):
**Examples**:
- "How do I install the openai library?"
- "What's the difference between GPT-3 and GPT-4?"
- General programming errors unrelated to API contracts
- Installation, setup, or conceptual questions

**Criteria**:
- No API contract violation evidence
- General programming or setup questions
- Conceptual discussions without implementation
- Off-topic or irrelevant content

## CONTRACT VIOLATION TAXONOMY:

Based on empirical analysis of LLM API violations, examine the post for these specific patterns:

### 1. Parameter Constraint Violations (28% of violations)
**Look for**:
- `max_tokens`, `temperature`, `top_p`, `frequency_penalty` with specific values
- Parameter validation errors: "must be ≤", "out of range", "invalid value"
- Model-specific limits: GPT-4 vs GPT-3.5 constraints
- Combination failures: conflicting parameter settings

**Examples**: 
- `max_tokens=5000` when model limit is 4096
- `temperature=2.5` when max is 2.0
- Invalid `response_format` specifications

### 2. Rate Limiting Violations (22% of violations)
**Look for**:
- HTTP 429 status codes, "Rate limit exceeded" messages
- Quota exhaustion: "insufficient_quota", billing issues
- Request frequency problems: bursts, concurrent requests
- Throttling discussions and backoff strategies

**Examples**:
- "Rate limit reached for requests"
- "You exceeded your current quota"
- TPM/RPM (tokens/requests per minute) violations

### 3. Context Length Violations (18% of violations)
**Look for**:
- Token limit errors: "maximum context length", "context_length_exceeded"
- Model-specific limits: 4K, 8K, 32K, 128K token discussions
- Truncation issues and conversation management
- Prompt engineering to fit context windows

**Examples**:
- "This model's maximum context length is 8192 tokens"
- Conversation truncation problems
- Input too long errors

### 4. Format/Schema Violations (15% of violations)
**Look for**:
- JSON parsing errors: "JSONDecodeError", malformed responses
- Function calling issues: tool use, schema validation
- Response format problems: structured output failures
- Message format violations: role/content structure

**Examples**:
- Invalid JSON schema in function calls
- Response format not matching specified structure
- Tool calling parameter mismatches

### 5. Authentication/Authorization Violations (10% of violations)
**Look for**:
- API key errors: "invalid_api_key", "unauthorized"
- Billing/subscription issues: account limitations
- Organization/team access problems
- Permission errors for specific models or features

**Examples**:
- "Invalid API key provided"
- Model access restrictions
- Billing account issues

### 6. Content Policy Violations (7% of violations)
**Look for**:
- Safety filter triggers: content moderation, policy violations
- Prompt rejections: harmful content detection
- Output filtering: response censoring or refusal
- Usage policy compliance issues

**Examples**:
- "This request violates our usage policies"
- Content flagged as inappropriate
- Safety system interventions

## POSTS TO AUTOMATICALLY EXCLUDE:

**❌ REJECT these post types immediately** (mark as "N" with high confidence):

### 1. Bug Reports (Not Contract Violations)
**Indicators**:
- "Bug:", "Issue:", "Unexpected behavior" in titles
- Reports of software defects, incorrect outputs, or functional problems
- SDK/library bugs unrelated to API usage limits
- Performance issues without constraint violations

**Examples to REJECT**:
- "Bug: Model generates incorrect JSON format"
- "ChatCompletion returns wrong response structure"
- "SDK crashes when using function calling"
- "Inconsistent responses between identical requests"

### 2. Feature Requests (Not Contract Violations)
**Indicators**:
- "Feature Request:", "Enhancement:", "Add support for" in titles
- Requests for new functionality, capabilities, or improvements
- Suggestions for API design changes
- "Would be nice if", "Can you add", "Please implement"

**Examples to REJECT**:
- "Feature Request: Add support for streaming function calls"
- "Enhancement: Allow custom stop sequences"
- "Please add GPT-5 model support"
- "Request: Batch processing endpoints"

### 3. General SDK/Library Issues (Not API Contract Violations)
**Indicators**:
- Installation problems, dependency conflicts
- Documentation requests, examples, tutorials
- Environment setup, configuration issues
- Version compatibility questions

**Examples to REJECT**:
- "How to install openai library on Windows?"
- "Requirements.txt missing dependency"
- "Documentation unclear about parameter X"
- "Example code doesn't work in Python 3.9"

### 4. Quality Complaints Without Contract Violations
**Indicators**:
- Subjective quality assessments without specific violations
- "Model is not good enough", "Results are poor"
- Expectation mismatches without documented contract breaches

**Examples to REJECT**:
- "GPT-4 doesn't understand my domain"
- "Model responses are too generic"
- "Quality decreased after update"

**CRITICAL**: Only mark as contract violations if there are specific, documented API usage constraints being violated (parameters, limits, quotas, formats, policies).

## COMMENT ANALYSIS PRIORITY:

**Critical**: This content includes both original post AND community comments. Comments often contain:

🔍 **Resolution Context**: How the problem was actually solved
🔍 **Expert Validation**: Community confirmation or dispute of the reported issue  
🔍 **Root Cause Analysis**: Technical explanations beyond the original post
🔍 **Workaround Strategies**: Community-developed solutions for contract limitations
🔍 **False Positive Detection**: Comments that clarify misunderstandings

**Comment Analysis Guidelines**:
- **Resolution Comments**: Look for accepted answers, marked solutions, "this worked" responses
- **Expert Feedback**: Official responses, maintainer comments, highly-voted explanations
- **Problem Clarification**: Additional context that changes the interpretation
- **Community Consensus**: Multiple people confirming or denying the same issue

## DECISION CRITERIA:

### ✅ **YES (Definitive Contract Violation)** - Evidence Level 4-5
**Requirements**:
- Clear API contract violation with specific technical details
- Reproducible error pattern or explicit API constraint violation
- Strong evidence from original post OR community validation in comments

**Examples**:
- Specific error messages with parameter violations
- Documented API limit breaches with values
- Community-confirmed rate limiting or quota issues

### ❌ **NO (Not a Contract Violation)** - Evidence Level 1-2  
**Requirements**:
- No clear API contract violation evidence
- Posts in the "AUTOMATICALLY EXCLUDE" categories above
- General programming, installation, or conceptual questions
- Issues unrelated to API usage constraints

**Examples**:
- Bug reports and feature requests (see exclusion list above)
- Library installation problems
- General "how to use" questions
- Quality complaints without contract violations
- SDK issues unrelated to API constraints

### 🤔 **BORDERLINE (Uncertain/Ambiguous)** - Evidence Level 3
**Requirements**:
- Some contract-related content but insufficient evidence
- Unclear technical details or incomplete information
- Mixed signals from community discussion

**Use sparingly**: Only when evidence is genuinely ambiguous and could go either way.

---

## POST FOR ANALYSIS:

**Title**: {title}

**Content** (includes original post and community comments): 
{content}

**Previous Analysis** (if any): {previous_analysis}

---

## YOUR ANALYSIS TASK:

As a senior researcher, provide a comprehensive evaluation following this structured approach:

### STEP 1: Evidence Assessment
- **FIRST**: Is this a bug report, feature request, or general SDK issue? (If yes → immediate "N")
- What evidence level (1-5) does this post demonstrate?
- Are there specific error messages, parameter values, or technical details?
- How do comments enhance or clarify the evidence?

### STEP 2: Contract Violation Detection  
- Which violation categories (if any) are present?
- Are there reproducible patterns or clear API constraint violations?
- Does the community discussion confirm or dispute the violation?

### STEP 3: Comment Integration
- What additional context do comments provide?
- Are there resolution confirmations or expert validations?
- Do comments reveal the actual root cause?

### STEP 4: Research Value Assessment
- Does this contribute to understanding LLM API contract patterns?
- Is this a novel issue or common problem?
- Would this help other developers or API designers?

## REQUIRED RESPONSE FORMAT:

DECISION: [Y/N/Borderline]
CONFIDENCE: [0.0-1.0]
EVIDENCE_LEVEL: [1-5]
CONTRACT_CATEGORIES: [List specific violation types, or "None" if not a violation]
COMMUNITY_VALIDATION: [How comments support or dispute the finding]
TECHNICAL_DETAILS: [Specific API elements involved: parameters, errors, limits, etc.]
RESOLUTION_CONTEXT: [How the issue was resolved based on comments]
RESEARCH_VALUE: [High/Medium/Low - contribution to understanding LLM API contracts]
RATIONALE: [Comprehensive 2-3 sentence justification citing specific evidence from post and comments]

**Remember**: You are the primary screener. Be thorough but decisive. Focus on clear contract violations with good evidence, and don't hesitate to mark non-violations as "N" even if they're LLM-related. **IMMEDIATELY REJECT bug reports, feature requests, and general SDK issues** - these are not contract violations."""

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
