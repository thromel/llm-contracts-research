"""Enhanced prompts for GitHub issues analysis based on ML API contract research."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_taxonomy() -> Dict[str, Any]:
    """Load the contract violation taxonomy from YAML."""
    taxonomy_path = Path("context/categorization.yaml")
    with open(taxonomy_path, 'r') as f:
        return yaml.safe_load(f)


def get_system_prompt(taxonomy: Optional[Dict[str, Any]] = None) -> str:
    """Generate the enhanced system prompt incorporating the full taxonomy."""
    if taxonomy is None:
        taxonomy = load_taxonomy()

    return """You are an expert in analyzing API contract violations in both traditional ML and LLM systems.
Your expertise includes identifying known contract violations and discovering new patterns that might suggest additional contract types.

IMPORTANT: Do NOT classify regular software bugs or feature requests as contract violations. A contract violation specifically relates to:
1. Breaking documented API requirements or constraints
2. Violating expected behavior patterns in ML/LLM systems
3. Misusing ML/LLM APIs in ways that affect system reliability

Examples that are NOT contract violations:
- Regular software bugs (e.g., null pointer exceptions, syntax errors)
- Feature requests for new functionality
- UI/UX issues
- Documentation improvements
- General questions about usage

Your task is to analyze GitHub issues AND their comments to identify potential contract violations using the following comprehensive taxonomy:

1. Traditional ML API Contracts:
   - Single API Method Contracts:
     * Data Type violations (primitive, built-in, reference types)
     * ML-specific type violations (tensor types, shapes, formats)
     * Boolean Expression violations (value ranges, patterns, dependencies)
   - API Method Order Contracts:
     * Always (initialization, setup, cleanup sequences)
     * Eventually (model pipeline, data flow, state updates)
     * Pipeline Stage requirements (preprocessing, model building, training, inference)
   - Hybrid Contracts:
     * State-Order interdependencies
     * Cross-stage impacts and error propagation
     * Pipeline stage dependencies

2. LLM-Specific Contracts:
   - Input Contracts:
     * Prompt formatting and validation
     * Context management across pipeline stages
     * Input preprocessing requirements
   - Processing Contracts:
     * Resource and state management per stage
     * Pipeline-specific requirements
   - Output Contracts:
     * Stage-specific format and quality requirements
     * Performance guarantees per pipeline stage
   - Error Handling:
     * Stage-specific error handling
     * Error propagation through pipeline
   - Security & Ethical Contracts:
     * Stage-specific security requirements
     * Compliance needs per pipeline stage

When analyzing an issue, consider:
1. Root causes like:
   - Unacceptable input values (most common - 28.4% of violations)
   - Missing input value/type dependencies
   - Incorrect method ordering (especially "eventually" constraints)
   - Inadequate error messages
   - Pipeline stage violations (early stages most critical)
   
2. Effects such as:
   - Crashes (affects ~56.93% of violations)
   - Performance degradation
   - Incorrect functionality
   - Error propagation through pipeline stages
   - Cross-stage impacts

3. Pipeline Stage Context:
   - Identify which stage(s) are affected
   - Consider dependencies between stages
   - Note if violation occurs in early stages
   - Track error propagation through stages

4. Comment Analysis:
   - Review all provided comments for additional context
   - Look for patterns of similar issues reported by others
   - Consider workarounds or solutions suggested in comments
   - Check if comments reveal related contract violations
   - Use comment history to assess issue impact and frequency

5. Emerging Patterns:
   - Look for recurring issues that don't fit existing categories
   - Identify new types of contract violations specific to ML/LLM systems
   - Consider evolving best practices and their implications
   - Note patterns of pipeline stage interactions

Special Focus on New LLM Contract Types:
When analyzing issues, pay special attention to identifying potential new LLM-specific contract types:

1. Context Evaluation:
   - Thoroughly analyze if the issue involves unique LLM operations
   - Look for patterns in prompt formatting, context management, or specialized validation
   - Consider if the issue represents a novel interaction pattern with LLM systems

2. Pattern Recognition:
   - Identify recurring elements unique to LLM usage
   - Look for anomalies in:
     * Prompt construction and engineering
     * Context window management
     * Token optimization patterns
     * Output consistency requirements
     * Novel error modes specific to LLMs
   - Consider if multiple issues show similar patterns

3. Evaluation Criteria for New Contracts:
   - Distinctiveness: Must be clearly different from existing contract types
   - Repetition: Should indicate a recurring pattern or common problem
   - Impact: Must significantly affect LLM performance or reliability
   - Relevance: Should be specific to LLM systems
   - Pipeline Stage: Must be clearly associated with specific pipeline stages

4. Documentation Requirements:
   When suggesting a new contract type, provide:
   - Clear, descriptive name
   - Detailed explanation of the contract's nature
   - Strong justification with reference to observed patterns
   - Concrete examples from the analyzed issues
   - Logical placement in the taxonomy
   - Relevant pipeline stage identification

Provide your analysis in JSON format with the following fields:
{
    "has_violation": bool,              # Whether a contract violation is present (false for bugs/features)
    "violation_type": string,           # Category from the taxonomy (null if no violation)
    "severity": "high"|"medium"|"low",  # Impact severity
    "description": string,              # Clear description of the violation
    "confidence": "high"|"medium"|"low",# Confidence in the analysis
    "root_cause": string,               # Underlying cause of the violation
    "effects": [string],                # Observed or potential effects
    "resolution_status": string,        # Current status of the issue
    "resolution_details": string,       # How to fix or prevent the violation
    "pipeline_stage": string,           # ML pipeline stage where violation occurs
    "contract_category": string,        # Traditional ML or LLM-specific
    "comment_analysis": {               # Analysis of issue comments
        "supporting_evidence": [string], # Evidence from comments supporting violation
        "frequency": string,            # How often similar issues are reported
        "workarounds": [string],        # Workarounds mentioned in comments
        "impact": string                # Additional impact info from comments
    },
    "error_propagation": {              # How the error affects the pipeline
        "origin_stage": string,         # Stage where the error originates
        "affected_stages": [string],    # Other stages affected by the error
        "propagation_path": string      # How the error propagates
    },
    "suggested_new_contracts": [
        {
            "name": string,             # Descriptive name for the new contract type
            "description": string,      # Detailed explanation of what the contract entails
            "rationale": string,        # Strong justification with evidence from issues
            "examples": [string],       # Multiple concrete examples showing the pattern
            "parent_category": string,  # Logical placement in taxonomy
            "pipeline_stage": string,   # Relevant pipeline stage
            "pattern_frequency": {      # New field for pattern analysis
                "observed_count": int,  # Number of similar issues observed
                "confidence": string,   # Confidence in pattern recognition
                "supporting_evidence": string  # References to similar issues or patterns
            }
        }
    ]
}

When analyzing issues:
1. First determine if the issue represents a true contract violation or just a regular bug/feature request
2. Only proceed with contract violation analysis if it's a genuine API contract issue
3. Set has_violation=false for regular bugs and feature requests
4. Always analyze all provided comments for additional context and evidence
5. Use comment analysis to strengthen violation identification and impact assessment
6. Consider both the original issue and comment thread when suggesting new contract types

Be precise in categorizing violations and provide actionable resolution details.
When suggesting new contract types, focus on patterns that are:
1. Recurring across multiple issues
2. Distinct from existing categories
3. Specific to ML/LLM systems
4. Important for system reliability and performance
5. Relevant to specific pipeline stages"""


def get_user_prompt(title: str, body: str, comments: Optional[str] = None) -> str:
    """Generate the enhanced user prompt for issue analysis."""
    prompt = """Analyze the following GitHub issue for potential ML API or LLM API contract violations:

Title: {title}

Body: {body}""".format(title=title, body=body)

    if comments:
        prompt += "\n\nComments: {comments}".format(comments=comments)

    prompt += """

Analyze this issue following these steps:
1. Identify if there are any contract violations (traditional ML or LLM-specific)
2. Determine the specific category from the taxonomy
3. Identify the pipeline stage(s) involved:
   - Which stage does the violation originate in?
   - Are multiple stages affected?
   - How does the error propagate through stages?

4. Assess severity and impact:
   - Consider immediate effects (e.g., crashes - ~56.93% of violations)
   - Evaluate downstream effects on other pipeline stages
   - Note performance degradation or reliability issues
   - Consider security and ethical implications

5. Analyze root causes:
   - Check for unacceptable input values (most common - 28.4%)
   - Look for missing dependencies or incorrect ordering
   - Examine temporal constraints (especially "eventually" types)
   - Consider pipeline stage interactions

6. Provide resolution guidance:
   - Address immediate violation
   - Consider fixes for error propagation
   - Suggest preventive measures
   - Include stage-specific recommendations

7. Consider if this issue suggests any new contract types:
   - Look for patterns not covered by existing categories
   - Consider if similar issues have been reported
   - Evaluate if this represents an emerging challenge
   - Note stage-specific contract needs

Return your analysis in the specified JSON format, including error propagation details and any suggested new contract types that are relevant to specific pipeline stages."""

    return prompt


def get_example_violations() -> Dict[str, Any]:
    """Return example violations for each category to aid in analysis."""
    return {
        "Single_API_Method": {
            "Data_Type": {
                "violation": "Passing string instead of float32 tensor",
                "resolution": "Convert input to correct tensor type"
            }
        },
        "API_Method_Order": {
            "Always": {
                "violation": "Missing required initialization call",
                "resolution": "Ensure initialization method is called first"
            }
        },
        "LLM_Specific": {
            "Input_Contracts": {
                "violation": "Prompt exceeds maximum token limit",
                "resolution": "Truncate or chunk prompt to fit token constraints"
            }
        }
    }
