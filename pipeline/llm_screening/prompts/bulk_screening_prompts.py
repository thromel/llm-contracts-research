"""
Bulk Screening Prompts for LLM Contract Violation Detection.

Based on empirical analysis of 600+ real-world contract violations from:
- OpenAI GPT APIs (most common)
- Anthropic Claude APIs  
- Google PaLM/Gemini APIs
- Meta LLaMA APIs
- Integration frameworks (LangChain, AutoGPT, etc.)

Research findings show the most common violation categories:
1. Parameter Constraints (28% of violations)
2. Rate Limiting (22% of violations) 
3. Content Policy (18% of violations)
4. Input/Output Format (16% of violations)
5. Context Length (12% of violations)
6. Authentication (4% of violations)
"""


class BulkScreeningPrompts:
    """Improved bulk screening prompts based on empirical research."""

    @staticmethod
    def get_screening_prompt() -> str:
        """Get the main bulk screening prompt."""
        return """You are an expert analyst identifying LLM API contract violations for academic research. You have analyzed 600+ real-world contract violations and understand the systematic patterns of how developers violate LLM API contracts.

RESEARCH-BASED CONTRACT VIOLATION TAXONOMY:

## 1. PARAMETER CONSTRAINTS (28% of violations)
**Most Common Category - Strict validation required**

### 1.1 Numeric Range Violations:
- max_tokens: Must be ≤ model limits (GPT-3.5: 4K, GPT-4: 8K/32K/128K, Claude: 200K)
- temperature: Must be 0.0-2.0 (values outside cause API errors)
- top_p: Must be 0.0-1.0 (cannot exceed 1.0)
- frequency_penalty/presence_penalty: Must be -2.0 to 2.0
- n (number of completions): Must be ≥1, usually ≤100

### 1.2 Invalid Parameter Combinations:
- Using temperature AND top_p simultaneously (some APIs discourage)
- Setting temperature=0 with top_p≠1 (conflicting determinism)
- Invalid model names or deprecated model versions

## 2. RATE LIMITING VIOLATIONS (22% of violations) 
**Second Most Common - Often critical production issues**

### 2.1 Request Rate Limits:
- HTTP 429 "Too Many Requests" errors
- RPM (Requests Per Minute) exceeded: Free tier ~3 RPM, Paid ~3000+ RPM
- Burst rate limits: Too many requests in short timeframe

### 2.2 Token Rate Limits:
- TPM (Tokens Per Minute) exceeded: Varies by model and tier
- Daily/monthly quota exhaustion
- Organization-level billing limits reached

## 3. CONTENT POLICY VIOLATIONS (18% of violations)
**Critical for safety - Often causes silent failures**

### 3.1 Input Content Violations:
- Prompts flagged as potentially harmful, illegal, or explicit
- Personal data or PII in prompts (GDPR/privacy violations)
- Attempts to jailbreak or bypass safety measures

### 3.2 Output Content Filtering:
- Generated content filtered by safety systems
- "I cannot provide information about..." responses
- Empty or truncated outputs due to policy violations

## 4. INPUT/OUTPUT FORMAT VIOLATIONS (16% of violations)
**Especially common in function calling and structured outputs**

### 4.1 Message Format Violations:
- Messages array missing required 'role' and 'content' fields
- Invalid role types (must be 'system', 'user', 'assistant', 'function', 'tool')
- Conversation missing required system message
- Message content type mismatches (string vs object)

### 4.2 Function Calling Format Violations:
- Function definitions missing required schema fields
- Tool calls with invalid JSON schemas
- Function responses not matching expected format
- Missing tool_choice or function_call parameters when required

### 4.3 Response Format Violations:
- Requesting JSON output but model returns plain text
- Schema validation failures in structured outputs
- Parsing errors: "Could not parse LLM output as JSON"

## 5. CONTEXT LENGTH VIOLATIONS (12% of violations)
**Increasingly common with longer conversations**

### 5.1 Token Count Exceeded:
- Total tokens (prompt + completion) > model maximum
- Conversation history too long for context window
- Single prompt exceeding input token limits

### 5.2 Context Management Issues:
- Not truncating old messages in long conversations
- Including full conversation history unnecessarily
- Inefficient prompt construction causing waste

## 6. AUTHENTICATION/AUTHORIZATION (4% of violations)
**Usually initial setup issues**

### 6.1 API Key Issues:
- Invalid, missing, or expired API keys (HTTP 401)
- Wrong API key format or encoding
- API key not properly set in environment/headers

### 6.2 Permission Issues:
- Insufficient permissions for requested model (HTTP 403)
- Organization access restrictions
- Beta feature access not granted

POST ANALYSIS TASK:

Title: {title}

Content: {content}

Based on this empirical taxonomy, analyze if this post discusses LLM API contract violations.

POSITIVE INDICATORS (Y):
✅ Specific error codes: 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 429 (Rate Limited), 500 (Server Error)
✅ Parameter validation errors with exact values/limits mentioned
✅ Rate limiting messages: "Rate limit exceeded", "Too many requests", "Quota exceeded"
✅ Content policy rejections: "flagged as potentially violating", "content filter", "safety"
✅ Format errors: "JSON parse error", "invalid message format", "schema validation failed"
✅ Context length errors: "context length exceeded", "too many tokens", "maximum context"
✅ Function calling errors: "invalid function definition", "tool call failed"
✅ Authentication failures: "invalid API key", "unauthorized", "billing issue"

NEGATIVE INDICATORS (N):
❌ General programming questions unrelated to LLM APIs
❌ Installation/environment setup not involving API usage
❌ Model quality/accuracy discussions (not contract violations)
❌ Conceptual questions about LLM capabilities
❌ Unrelated library errors (numpy, pandas, etc.)
❌ Network issues unrelated to API calls

QUALITY REQUIREMENTS FOR POSITIVE CLASSIFICATION:
- Must contain specific technical details or error messages
- Should show actual API usage scenarios (code, parameters, responses)
- Needs clear cause-effect relationship between violation and outcome
- Should demonstrate real-world developer experiences

Respond EXACTLY in this format:
DECISION: [Y/N]
CONFIDENCE: [0.0-1.0]
RATIONALE: [Specific contract category and evidence - be precise about which of the 6 categories applies]"""

    @staticmethod
    def get_high_precision_prompt() -> str:
        """Get a more precise prompt for critical screening tasks."""
        return """You are a senior researcher specializing in LLM API contract violations. You must maintain extremely high precision to avoid false positives in research data.

STRICT EVIDENCE REQUIREMENTS:

ACCEPT ONLY IF POST CONTAINS:
1. Specific LLM API error messages with codes/details
2. Actual parameter values that violated documented limits  
3. Real code examples showing API misuse
4. Clear technical cause-effect relationships
5. Verifiable contract violation patterns

STRICT REJECTION CRITERIA:
- Vague or unclear technical descriptions
- Hypothetical scenarios without real examples
- Quality/accuracy complaints (not contract issues)  
- General debugging without API specifics
- Installation/setup issues unrelated to API contracts

POST TO ANALYZE:
Title: {title}
Content: {content}

DECISION: [Y/N]
CONFIDENCE: [0.0-1.0]  
RATIONALE: [Be extremely specific about evidence]"""

    @staticmethod
    def get_context_aware_prompt() -> str:
        """Get prompt that considers context and metadata."""
        return """You are analyzing LLM API posts with additional context to improve accuracy.

POST ANALYSIS:
Title: {title}
Content: {content}

CONTEXT METADATA:
Platform: {platform}
Tags: {tags}
Score: {score}
Matched Keywords: {keywords}

Use this context to better understand the post's relevance and quality.
High scores and relevant tags increase confidence.
Technical tags (api, openai, error, rate-limit) are strong positive signals.

DECISION: [Y/N]
CONFIDENCE: [0.0-1.0]
RATIONALE: [Include how context influenced your decision]"""
