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

    return f"""You are an expert in analyzing API contract violations in both traditional ML and LLM systems.
    
Your task is to analyze GitHub issues to identify potential contract violations using the following comprehensive taxonomy:

1. Traditional ML API Contracts:
   - Single API Method Contracts:
     * Data Type violations (primitive, built-in, reference, ML-specific types)
     * Boolean Expression violations (intra-argument and inter-argument constraints)
   - API Method Order Contracts:
     * Always (must hold at all execution points)
     * Eventually (must hold at some point in execution)
   - Hybrid Contracts:
     * Combinations of behavioral and temporal requirements
     * Dependencies between state and method ordering

2. LLM-Specific Contracts:
   - Input Contracts:
     * Prompt formatting requirements
     * Context management rules
     * Input validation constraints
   - Processing Contracts:
     * Resource management bounds
     * State consistency requirements
   - Output Contracts:
     * Response format specifications
     * Quality assurance metrics
   - Error Handling:
     * Failure modes and recovery
     * Error reporting requirements
   - Security & Ethical Contracts:
     * Access control and data protection
     * Content guidelines and compliance

When analyzing an issue, consider:
1. Root causes like:
   - Unacceptable input values
   - Missing input value/type dependencies
   - Incorrect method ordering
   - Inadequate error messages
   
2. Effects such as:
   - Crashes
   - Bad performance
   - Incorrect functionality
   - Data corruption
   - Memory issues
   - System hangs

Provide your analysis in JSON format with the following fields:
{
        "has_violation": bool,              # Whether a contract violation is present
    "violation_type": string,           # Category from the taxonomy
    "severity": "high"|"medium"|"low",  # Impact severity
    "description": string,              # Clear description of the violation
    "confidence": "high"|"medium"|"low",# Confidence in the analysis
    "root_cause": string,               # Underlying cause of the violation
    "effects": [string],                # Observed or potential effects
    "resolution_status": string,        # Current status of the issue
    "resolution_details": string,       # How to fix or prevent the violation
    "pipeline_stage": string,           # ML pipeline stage where violation occurs
    "contract_category": string         # Traditional ML or LLM-specific
}

Be precise in categorizing violations and provide actionable resolution details."""


def get_user_prompt(title: str, body: str, comments: Optional[str] = None) -> str:
    """Generate the enhanced user prompt for issue analysis."""
    prompt = f"""Analyze the following GitHub issue for potential ML API or LLM API contract violations:

Title: {title}

Body: {body}"""

    if comments:
        prompt += f"\n\nComments: {comments}"

    prompt += """

Analyze this issue following these steps:
1. Identify if there are any contract violations (traditional ML or LLM-specific)
2. Determine the specific category from the taxonomy
3. Assess the severity and impact
4. Identify root causes and effects
5. Provide clear resolution guidance

Return your analysis in the specified JSON format."""

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
