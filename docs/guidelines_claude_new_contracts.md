# Guidelines for Identifying New LLM-Specific Contract Types

## Introduction
This document provides detailed instructions for Claude to identify and suggest new LLM-specific contract types when analyzing GitHub issues. These guidelines ensure that any emerging contract patterns unique to LLM functionalities are captured, documented, and suggested for inclusion in the taxonomy.

## Step-by-Step Instructions

1. **Context Evaluation**
   - Read the entire GitHub issue, including the title, body, and comments, to understand the context.
   - Determine if the issue is related to LLM operations (e.g., prompt formatting, context management, specialized input validation, or output processing).

2. **Mapping to Existing Taxonomy**
   - Attempt to categorize the issue using the current taxonomy under the LLM_Specific category (e.g., Input_Contracts, Processing_Contracts, Output_Contracts, Error_Handling, Security_Contracts, Ethical_Contracts).
   - If the issue does not clearly fit into any existing subcategory or presents novel characteristics, consider it as a candidate for a new contract type.

3. **Pattern Identification**
   - Look for recurring elements or error patterns that are unique to LLM usage. Focus on anomalies in prompt construction, context propagation, or output inconsistencies that are not addressed by current contract types.
   - Evaluate if similar issues have been reported or if multiple instances suggest a common underlying contract violation.

4. **Criteria for Suggesting a New Contract Type**
   - **Distinctiveness:** The observed pattern must be distinct from any existing contract type in the taxonomy.
   - **Repetition:** The pattern should appear in multiple issues or be strongly indicative of a recurring problem.
   - **Impact:** The violation should have a significant impact on LLM performance or reliability.
   - **Relevance:** The pattern should be specific to LLM systems, such as novel prompt handling issues or context management challenges.
   - **Pipeline Stage:** Identify the specific pipeline stage (e.g., Input Generation, Inference) where the violation occurs or propagates.

5. **Documenting the New Contract Suggestion**
   When a new contract type is identified, create a detailed suggestion with the following structure:
   - **Name:** A concise and descriptive name for the new contract type.
   - **Description:** A detailed explanation of what the contract entails and the nature of the violation.
   - **Rationale:** Justification for why this should be considered a new contract type, including reference to recurring issues or observed gaps.
   - **Examples:** One or more examples or scenarios that illustrate the violation.
   - **Parent Category:** The higher-level category under the current taxonomy where the new type logically fits (e.g., LLM_Specific.Input_Contracts).
   - **Pipeline Stage:** The relevant stage of the ML/LLM pipeline where this contract is applicable (e.g., Input Generation, Inference).

6. **Integration into the Analysis Output**
   - Include the new contract suggestion in the JSON output under the `suggested_new_contracts` field. Ensure that the structure adheres to the expected schema.
   - If no new pattern is detected, return an empty list for `suggested_new_contracts`.

## Example JSON Structure for New Contract Suggestions
```json
"suggested_new_contracts": [
  {
    "name": "LLM_Prompt_Context_Contract",
    "description": "Ensures that prompt construction includes all necessary contextual elements to avoid ambiguity in model responses.",
    "rationale": "Recurring issues have been observed where insufficient context in the prompt leads to vague or incorrect outputs.",
    "examples": ["Prompt missing key context details resulting in ambiguous responses."],
    "parent_category": "LLM_Specific.Input_Contracts",
    "pipeline_stage": "Input Generation"
  }
]
```

## Final Notes
- Always ensure clarity and precision in your analysis; avoid overgeneralizing or speculating without evidence from the issue content.
- Use lower confidence ratings if the new contract type suggestion is less certain, but still document the possibility.
- The goal is to capture emerging LLM-specific contract patterns that are not yet covered by the current taxonomy, improving overall system robustness.

Follow these guidelines rigorously when analyzing issues for potential new LLM-specific contracts. 