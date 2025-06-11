"""
LLM API Contract Taxonomy based on empirical research.

This module implements the comprehensive taxonomy of LLM API contracts
derived from analysis of 600+ real-world contract violations.

Based on the research extending Khairunnesa et al.'s work on ML API contracts
to the domain of LLM APIs.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field


class ContractCategory(str, Enum):
    """High-level contract categories."""
    SINGLE_API_METHOD = "single_api_method"  # Behavioral contracts (SAM)
    API_METHOD_ORDER = "api_method_order"    # Temporal contracts (AMO)
    HYBRID = "hybrid"                        # Combined behavioral and temporal
    LLM_SPECIFIC = "llm_specific"           # Novel LLM-specific contracts


class SingleAPIMethodContract(str, Enum):
    """Behavioral contracts - constraints on single API calls."""
    # Data Type (DT) subcategories
    PRIMITIVE_TYPE = "primitive_type"        # int, float, str, bool
    BUILT_IN_TYPE = "built_in_type"         # list, dict, tuple
    REFERENCE_TYPE = "reference_type"        # object references
    ML_TYPE = "ml_type"                     # tensors, arrays
    LLM_TYPE = "llm_type"                   # messages, embeddings
    
    # Boolean Expression Type (BET) subcategories
    INTRA_ARGUMENT = "intra_argument"        # Single parameter constraints
    INTER_ARGUMENT = "inter_argument"        # Multi-parameter dependencies
    
    # Value constraints
    VALUE_RANGE = "value_range"              # Numeric ranges
    STRING_FORMAT = "string_format"          # String patterns
    ENUM_VALUE = "enum_value"                # Allowed values
    
    # Size constraints
    LENGTH_LIMIT = "length_limit"            # Token/character limits
    ARRAY_DIMENSION = "array_dimension"      # Array shape constraints


class APIMethodOrderContract(str, Enum):
    """Temporal contracts - constraints on API call sequences."""
    ALWAYS = "always"                        # Must always follow order (G)
    EVENTUALLY = "eventually"                # Must eventually be called (F)
    NEVER = "never"                         # Never call in sequence
    CONDITIONAL = "conditional"              # Order depends on state


class HybridContract(str, Enum):
    """Combined behavioral and temporal contracts."""
    SAM_AMO_INTERDEPENDENCY = "sam_amo_interdependency"  # SAI
    SELECTION = "selection"                               # SL - choice of contracts
    STATE_DEPENDENT_ORDER = "state_dependent_order"       # Order based on state
    PARAMETER_DEPENDENT_ORDER = "parameter_dependent_order"  # Order based on params


class LLMSpecificContract(str, Enum):
    """Novel contract types specific to LLM APIs."""
    # Prompt/Output format contracts
    PROMPT_FORMAT = "prompt_format"          # Message structure requirements
    OUTPUT_FORMAT = "output_format"          # Expected response format
    JSON_SCHEMA = "json_schema"              # Structured output schema
    FUNCTION_CALLING = "function_calling"    # Function/tool calling format
    
    # Content policy contracts
    CONTENT_POLICY = "content_policy"        # Usage policy compliance
    SAFETY_FILTER = "safety_filter"          # Safety system constraints
    PII_HANDLING = "pii_handling"           # Personal info handling
    
    # Multi-turn interaction contracts
    CONTEXT_MANAGEMENT = "context_management"  # Conversation state
    MEMORY_CONSTRAINT = "memory_constraint"    # Context window limits
    TOOL_AVAILABILITY = "tool_availability"    # Available tools/functions
    
    # Streaming and async contracts
    STREAM_FORMAT = "stream_format"          # Streaming response format
    ASYNC_HANDLING = "async_handling"        # Async operation constraints


class ParameterConstraint(str, Enum):
    """Specific parameter constraint types."""
    # Token/length constraints
    MAX_TOKENS = "max_tokens"
    MAX_INPUT_TOKENS = "max_input_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    CONTEXT_LENGTH = "context_length"
    
    # Sampling parameters
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    TOP_K = "top_k"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    REPETITION_PENALTY = "repetition_penalty"
    
    # Model selection
    MODEL_NAME = "model_name"
    MODEL_VERSION = "model_version"
    ENDPOINT = "endpoint"
    
    # Special parameters
    SEED = "seed"
    STOP_SEQUENCES = "stop_sequences"
    LOGIT_BIAS = "logit_bias"
    RESPONSE_FORMAT = "response_format"


class RateLimitType(str, Enum):
    """Types of rate limiting contracts."""
    REQUESTS_PER_MINUTE = "rpm"
    REQUESTS_PER_DAY = "rpd"
    TOKENS_PER_MINUTE = "tpm"
    TOKENS_PER_DAY = "tpd"
    CONCURRENT_REQUESTS = "concurrent"
    COST_LIMIT = "cost_limit"


class ViolationSeverity(str, Enum):
    """Severity levels for contract violations."""
    CRITICAL = "critical"  # Immediate failure
    HIGH = "high"         # Significant impact
    MEDIUM = "medium"     # Moderate impact
    LOW = "low"          # Minor impact
    WARNING = "warning"   # Potential issue


@dataclass
class ContractDefinition:
    """Complete definition of a contract type."""
    id: str
    name: str
    category: ContractCategory
    subcategory: Optional[str] = None
    description: str = ""
    
    # Contract details
    parameters: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)
    
    # Examples and documentation
    examples: List[Dict[str, Any]] = field(default_factory=list)
    common_violations: List[str] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)
    
    # Metadata
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    frequency: float = 0.0  # Percentage from empirical data
    affects_frameworks: List[str] = field(default_factory=list)
    api_providers: List[str] = field(default_factory=list)


class LLMContractTaxonomy:
    """Comprehensive taxonomy of LLM API contracts."""
    
    def __init__(self):
        self.contracts = self._initialize_contracts()
        self._build_indices()
    
    def _initialize_contracts(self) -> Dict[str, ContractDefinition]:
        """Initialize all contract definitions based on research."""
        contracts = {}
        
        # Parameter constraint contracts (28% of violations)
        contracts["max_tokens_limit"] = ContractDefinition(
            id="max_tokens_limit",
            name="Maximum Tokens Limit",
            category=ContractCategory.SINGLE_API_METHOD,
            subcategory=SingleAPIMethodContract.VALUE_RANGE,
            description="Constraint on maximum tokens in request/response",
            parameters=["max_tokens", "max_completion_tokens"],
            constraints={
                "gpt-3.5-turbo": {"min": 1, "max": 4096},
                "gpt-4": {"min": 1, "max": 8192},
                "gpt-4-32k": {"min": 1, "max": 32768},
                "claude-2": {"min": 1, "max": 100000},
                "claude-3": {"min": 1, "max": 200000}
            },
            error_patterns=[
                "maximum context length",
                "max_tokens exceeds",
                "InvalidRequestError",
                "context_length_exceeded"
            ],
            severity=ViolationSeverity.HIGH,
            frequency=0.12,
            api_providers=["openai", "anthropic"]
        )
        
        contracts["temperature_range"] = ContractDefinition(
            id="temperature_range",
            name="Temperature Parameter Range",
            category=ContractCategory.SINGLE_API_METHOD,
            subcategory=SingleAPIMethodContract.VALUE_RANGE,
            description="Valid range for temperature sampling parameter",
            parameters=["temperature"],
            constraints={"min": 0.0, "max": 2.0},
            error_patterns=[
                "temperature must be between",
                "InvalidParameterError",
                "ValidationError"
            ],
            common_violations=[
                "Setting temperature > 2.0",
                "Using negative temperature values",
                "Setting temperature with top_p simultaneously"
            ],
            severity=ViolationSeverity.MEDIUM,
            frequency=0.08
        )
        
        # Rate limiting contracts (22% of violations)
        contracts["rate_limit_rpm"] = ContractDefinition(
            id="rate_limit_rpm",
            name="Requests Per Minute Limit",
            category=ContractCategory.LLM_SPECIFIC,
            subcategory=RateLimitType.REQUESTS_PER_MINUTE,
            description="API request rate limits per minute",
            constraints={
                "free_tier": {"gpt-3.5": 20, "gpt-4": 3},
                "paid_tier": {"gpt-3.5": 3500, "gpt-4": 500}
            },
            error_patterns=[
                "Rate limit reached",
                "429 Too Many Requests",
                "quota exceeded",
                "RateLimitError"
            ],
            prevention_strategies=[
                "Implement exponential backoff",
                "Use request queuing",
                "Monitor rate limit headers"
            ],
            severity=ViolationSeverity.HIGH,
            frequency=0.15
        )
        
        # Content policy contracts (18% of violations)
        contracts["content_policy"] = ContractDefinition(
            id="content_policy",
            name="Content Policy Compliance",
            category=ContractCategory.LLM_SPECIFIC,
            subcategory=LLMSpecificContract.CONTENT_POLICY,
            description="Compliance with provider content policies",
            error_patterns=[
                "content policy violation",
                "flagged as potentially harmful",
                "I cannot provide",
                "safety filter triggered"
            ],
            common_violations=[
                "Requesting harmful content",
                "Including PII in prompts",
                "Attempting jailbreaks"
            ],
            severity=ViolationSeverity.CRITICAL,
            frequency=0.18
        )
        
        # Format contracts (16% of violations)
        contracts["message_format"] = ContractDefinition(
            id="message_format",
            name="Message Format Structure",
            category=ContractCategory.LLM_SPECIFIC,
            subcategory=LLMSpecificContract.PROMPT_FORMAT,
            description="Required structure for chat messages",
            parameters=["messages"],
            constraints={
                "required_fields": ["role", "content"],
                "valid_roles": ["system", "user", "assistant", "function", "tool"],
                "structure": "array of message objects"
            },
            error_patterns=[
                "messages must be an array",
                "missing required field",
                "invalid role",
                "malformed message"
            ],
            severity=ViolationSeverity.HIGH,
            frequency=0.10
        )
        
        contracts["json_output_format"] = ContractDefinition(
            id="json_output_format",
            name="JSON Output Format",
            category=ContractCategory.LLM_SPECIFIC,
            subcategory=LLMSpecificContract.OUTPUT_FORMAT,
            description="Structured JSON output requirements",
            parameters=["response_format", "functions", "tools"],
            error_patterns=[
                "Could not parse LLM output",
                "Invalid JSON",
                "JSONDecodeError",
                "Schema validation failed"
            ],
            prevention_strategies=[
                "Use response_format parameter",
                "Provide clear JSON examples",
                "Implement retry with clarification"
            ],
            severity=ViolationSeverity.MEDIUM,
            frequency=0.06
        )
        
        # Context length contracts (12% of violations)
        contracts["context_window"] = ContractDefinition(
            id="context_window",
            name="Context Window Limit",
            category=ContractCategory.LLM_SPECIFIC,
            subcategory=LLMSpecificContract.MEMORY_CONSTRAINT,
            description="Total token limit for input + output",
            constraints={
                "gpt-3.5-turbo-16k": 16384,
                "gpt-4": 8192,
                "gpt-4-turbo": 128000,
                "claude-3-opus": 200000,
                "gemini-pro": 1048576
            },
            error_patterns=[
                "context length exceeded",
                "conversation too long",
                "maximum context",
                "token limit"
            ],
            severity=ViolationSeverity.HIGH,
            frequency=0.12
        )
        
        # Authentication contracts (4% of violations)
        contracts["api_key_auth"] = ContractDefinition(
            id="api_key_auth",
            name="API Key Authentication",
            category=ContractCategory.SINGLE_API_METHOD,
            subcategory=SingleAPIMethodContract.STRING_FORMAT,
            description="Valid API key format and permissions",
            parameters=["api_key", "authorization"],
            error_patterns=[
                "Invalid API key",
                "401 Unauthorized",
                "403 Forbidden",
                "insufficient permissions"
            ],
            severity=ViolationSeverity.CRITICAL,
            frequency=0.04
        )
        
        # Temporal contracts
        contracts["session_initialization"] = ContractDefinition(
            id="session_initialization",
            name="Session Initialization Order",
            category=ContractCategory.API_METHOD_ORDER,
            subcategory=APIMethodOrderContract.ALWAYS,
            description="Required initialization before API calls",
            error_patterns=[
                "client not initialized",
                "session not started",
                "missing configuration"
            ],
            affects_frameworks=["langchain", "openai"],
            severity=ViolationSeverity.HIGH,
            frequency=0.03
        )
        
        # Hybrid contracts
        contracts["tool_calling_order"] = ContractDefinition(
            id="tool_calling_order",
            name="Tool Calling Order Dependency",
            category=ContractCategory.HYBRID,
            subcategory=HybridContract.PARAMETER_DEPENDENT_ORDER,
            description="Tool/function calling requires proper setup and format",
            parameters=["tools", "functions", "tool_choice"],
            error_patterns=[
                "tool not found",
                "invalid tool call",
                "function not defined",
                "agent stopped due to"
            ],
            affects_frameworks=["langchain", "autogpt"],
            severity=ViolationSeverity.MEDIUM,
            frequency=0.05
        )
        
        return contracts
    
    def _build_indices(self):
        """Build indices for efficient lookups."""
        self.by_category = {}
        self.by_severity = {}
        self.by_frequency = {}
        self.by_provider = {}
        self.by_framework = {}
        
        for contract_id, contract in self.contracts.items():
            # By category
            cat = contract.category.value
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(contract_id)
            
            # By severity
            sev = contract.severity.value
            if sev not in self.by_severity:
                self.by_severity[sev] = []
            self.by_severity[sev].append(contract_id)
            
            # By provider
            for provider in contract.api_providers:
                if provider not in self.by_provider:
                    self.by_provider[provider] = []
                self.by_provider[provider].append(contract_id)
            
            # By framework
            for framework in contract.affects_frameworks:
                if framework not in self.by_framework:
                    self.by_framework[framework] = []
                self.by_framework[framework].append(contract_id)
    
    def get_contract(self, contract_id: str) -> Optional[ContractDefinition]:
        """Get a specific contract definition."""
        return self.contracts.get(contract_id)
    
    def get_contracts_by_category(self, category: ContractCategory) -> List[ContractDefinition]:
        """Get all contracts in a category."""
        contract_ids = self.by_category.get(category.value, [])
        return [self.contracts[cid] for cid in contract_ids]
    
    def get_contracts_by_severity(self, severity: ViolationSeverity) -> List[ContractDefinition]:
        """Get all contracts of a given severity."""
        contract_ids = self.by_severity.get(severity.value, [])
        return [self.contracts[cid] for cid in contract_ids]
    
    def get_contracts_by_provider(self, provider: str) -> List[ContractDefinition]:
        """Get all contracts affecting a specific API provider."""
        contract_ids = self.by_provider.get(provider, [])
        return [self.contracts[cid] for cid in contract_ids]
    
    def get_contracts_by_framework(self, framework: str) -> List[ContractDefinition]:
        """Get all contracts affecting a specific framework."""
        contract_ids = self.by_framework.get(framework, [])
        return [self.contracts[cid] for cid in contract_ids]
    
    def identify_violations(self, error_text: str) -> List[Tuple[ContractDefinition, float]]:
        """Identify potential contract violations from error text."""
        matches = []
        error_lower = error_text.lower()
        
        for contract in self.contracts.values():
            score = 0.0
            matched_patterns = 0
            
            # Check error patterns
            for pattern in contract.error_patterns:
                if pattern.lower() in error_lower:
                    matched_patterns += 1
            
            if matched_patterns > 0:
                score = matched_patterns / len(contract.error_patterns)
                matches.append((contract, score))
        
        # Sort by confidence score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get taxonomy statistics."""
        total_contracts = len(self.contracts)
        
        # Category distribution
        category_dist = {}
        for cat in ContractCategory:
            contracts = self.get_contracts_by_category(cat)
            category_dist[cat.value] = {
                "count": len(contracts),
                "percentage": len(contracts) / total_contracts * 100
            }
        
        # Severity distribution
        severity_dist = {}
        for sev in ViolationSeverity:
            contracts = self.get_contracts_by_severity(sev)
            severity_dist[sev.value] = {
                "count": len(contracts),
                "percentage": len(contracts) / total_contracts * 100
            }
        
        # Top violations by frequency
        top_violations = sorted(
            self.contracts.values(),
            key=lambda c: c.frequency,
            reverse=True
        )[:5]
        
        return {
            "total_contracts": total_contracts,
            "category_distribution": category_dist,
            "severity_distribution": severity_dist,
            "top_violations": [
                {
                    "id": c.id,
                    "name": c.name,
                    "frequency": c.frequency
                }
                for c in top_violations
            ]
        }


# Singleton instance
taxonomy = LLMContractTaxonomy()