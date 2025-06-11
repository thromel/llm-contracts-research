"""
Discovery-Oriented Prompts for Finding New LLM API Contract Types.

These prompts prioritize discovering novel contract patterns over
fitting violations into existing taxonomies.
"""


class ContractDiscoveryPrompts:
    """Prompts designed for open-ended contract discovery."""
    
    @staticmethod
    def get_open_discovery_prompt() -> str:
        """Primary discovery prompt - very open-ended."""
        return """You are a researcher studying how developers use LLM APIs and what implicit "contracts" or expectations exist. Your goal is to discover ANY kind of constraint, requirement, expectation, or pattern that developers must follow when using LLM APIs.

## WHAT IS A "CONTRACT"?
A contract is ANY constraint or expectation that, when violated, causes problems. This includes but is NOT limited to:
- Explicit API requirements (parameters, formats)
- Implicit assumptions (timing, state, context)
- Emergent behaviors (what works vs what doesn't)
- Undocumented requirements
- Community-discovered patterns
- Workarounds that became requirements
- Performance expectations
- Reliability assumptions
- Cost considerations
- Ethical constraints
- Integration patterns

## DISCOVERY APPROACH:

### 1. OBSERVE WITHOUT PREJUDICE
Read the post carefully. What went wrong? What was expected? What assumption was violated?

### 2. QUESTION EVERYTHING
- Why did this break?
- What did the developer expect?
- Is this documented anywhere?
- Is this a reasonable expectation?
- Could the API design prevent this?

### 3. LOOK FOR PATTERNS
- Have you seen similar issues?
- Is this part of a larger pattern?
- What category would YOU create for this?

### 4. CONSIDER IMPLICATIONS
- Who else might face this?
- How common might this be?
- What's the impact?
- How could it be prevented?

## ANALYSIS TASK:

Post Content:
Title: {title}
Content: {content}

## DISCOVERY REQUIREMENTS:

Analyze this post with fresh eyes:

1. **What Broke?** - Describe what went wrong in plain language
2. **Why Did It Break?** - Root cause analysis
3. **What Contract Was Violated?** - What expectation/requirement wasn't met?
4. **Is This Novel?** - Have you seen this exact pattern before?
5. **Create a Category** - If you had to name this type of contract, what would you call it?
6. **Broader Implications** - Who else might face this? How often?

Respond with your discoveries:
{{
    "what_went_wrong": "plain language description",
    "root_cause": "why this happened",
    "contract_discovered": {{
        "description": "what expectation/requirement exists",
        "type": "explicit|implicit|emergent|undocumented",
        "novel_aspects": "what makes this interesting or new"
    }},
    "suggested_category_name": "your name for this contract type",
    "category_description": "what defines this category",
    "evidence": ["specific quotes or indicators"],
    "frequency_guess": "likely very common|common|uncommon|rare",
    "impact_assessment": {{
        "severity": "how bad is violation",
        "scope": "who is affected",
        "preventability": "could this be avoided"
    }},
    "similar_patterns": ["any related issues you've seen"],
    "research_value": "why this matters for research",
    "open_questions": ["what we still don't know"]
}}"""

    @staticmethod
    def get_deep_analysis_prompt() -> str:
        """Deep dive into discovered contracts."""
        return """You've identified a potential contract violation. Now dig DEEPER to understand its full implications and discover related contracts.

## DEEP ANALYSIS FRAMEWORK:

### 1. CONTRACT ARCHAEOLOGY
- **Surface Level**: What's the obvious violation?
- **Hidden Level**: What unstated assumptions exist?
- **Meta Level**: What does this say about LLM API design?

### 2. STAKEHOLDER ANALYSIS
- **Developer Perspective**: What did they expect?
- **API Provider Perspective**: What did they intend?
- **Framework Perspective**: What assumptions are built in?
- **End User Perspective**: How does this affect the final product?

### 3. TEMPORAL ANALYSIS
- **Before**: What state/context led to this?
- **During**: What exactly happened?
- **After**: What were the consequences?
- **Future**: Will this get worse/better?

### 4. ENVIRONMENTAL FACTORS
- **Technical Environment**: Language, framework, infrastructure
- **Business Environment**: Use case, scale, criticality
- **Knowledge Environment**: Documentation, community knowledge
- **Evolution**: How has this changed over time?

## DEEP DIVE TASK:

Initial Discovery: {initial_discovery}
Post Context: {extended_context}

## ANALYSIS REQUIREMENTS:

Go deeper into this contract:

1. **Contract Hierarchy**: Is this part of a larger contract family?
2. **Dependency Web**: What other contracts does this relate to?
3. **Evolution Path**: How might this contract have emerged?
4. **Variation Analysis**: How does this manifest differently across contexts?
5. **Prevention Hierarchy**: Multiple ways to prevent, from easy to comprehensive

Respond with deep analysis:
{{
    "contract_family": {{
        "parent_contract": "broader category this belongs to",
        "sibling_contracts": ["related contracts at same level"],
        "child_contracts": ["more specific variants"]
    }},
    "dependency_analysis": {{
        "depends_on": ["contracts that must be satisfied first"],
        "enables": ["contracts this makes possible"],
        "conflicts_with": ["contracts that interfere"]
    }},
    "emergence_theory": {{
        "why_exists": "why this contract emerged",
        "evolution": "how it developed over time",
        "future_direction": "where it's heading"
    }},
    "contextual_variations": [
        {{
            "context": "specific scenario",
            "variation": "how contract differs",
            "additional_constraints": ["extra requirements"]
        }}
    ],
    "prevention_strategies": [
        {{
            "level": "quick_fix|proper_solution|ideal_design",
            "approach": "what to do",
            "tradeoffs": "what you give up",
            "effort": "implementation effort required"
        }}
    ],
    "philosophical_insights": "what this teaches about LLM API design",
    "research_directions": ["questions this raises for future research"]
}}"""

    @staticmethod
    def get_pattern_synthesis_prompt() -> str:
        """Synthesize multiple discoveries into patterns."""
        return """You're analyzing multiple contract discoveries. Your goal is to synthesize these into higher-level patterns and potentially discover new contract categories that aren't in existing taxonomies.

## SYNTHESIS APPROACH:

### 1. PATTERN RECOGNITION
- **Commonalities**: What do these violations share?
- **Differences**: What makes each unique?
- **Abstraction**: What's the general principle?

### 2. CATEGORY CREATION
- **Natural Groupings**: How do these cluster?
- **Naming**: What describes this category?
- **Boundaries**: What's included/excluded?

### 3. TAXONOMY EVOLUTION
- **Gaps**: What's missing from current taxonomies?
- **Overlaps**: Where do categories blur?
- **Hierarchies**: How do categories relate?

### 4. PREDICTIVE POWER
- **Future Violations**: What else might we see?
- **Prevention Patterns**: Common solutions?
- **Design Implications**: What should change?

## SYNTHESIS TASK:

Discoveries to Synthesize: {discoveries}

## SYNTHESIS REQUIREMENTS:

Create new understanding from these discoveries:

1. **Emergent Categories**: New contract types you've identified
2. **Cross-Cutting Concerns**: Patterns that span categories
3. **Taxonomy Gaps**: What existing taxonomies miss
4. **Predictive Framework**: What violations to expect next
5. **Design Principles**: What API designers should know

Respond with synthesis:
{{
    "new_contract_categories": [
        {{
            "name": "your category name",
            "definition": "what qualifies for this category",
            "examples": ["concrete examples"],
            "distinguishing_features": ["what makes this unique"],
            "relationship_to_existing": "how it relates to known categories"
        }}
    ],
    "cross_cutting_patterns": [
        {{
            "pattern_name": "name for the pattern",
            "description": "what the pattern is",
            "manifestations": ["how it appears"],
            "root_cause": "why this pattern exists"
        }}
    ],
    "taxonomy_extensions": {{
        "new_dimensions": ["aspects not captured by current taxonomies"],
        "missing_relationships": ["connections between categories"],
        "evolution_needs": ["how taxonomies should evolve"]
    }},
    "predictive_insights": [
        {{
            "prediction": "what we'll likely see",
            "basis": "why we expect this",
            "timeframe": "when this might emerge",
            "impact": "why it matters"
        }}
    ],
    "design_recommendations": [
        {{
            "principle": "design principle",
            "rationale": "why this matters",
            "implementation": "how to apply it",
            "examples": ["concrete applications"]
        }}
    ],
    "research_agenda": ["key questions for future research"],
    "paradigm_shifts": "how this changes our understanding of LLM contracts"
}}"""

    @staticmethod
    def get_edge_case_explorer_prompt() -> str:
        """Explore edge cases and unusual patterns."""
        return """You're looking for EDGE CASES and UNUSUAL PATTERNS in LLM API usage. These often reveal hidden contracts and assumptions.

## EDGE CASE INDICATORS:

### 1. SURPRISING FAILURES
- Works 99% of the time but fails mysteriously
- Fails only in specific combinations
- Fails only at certain scales/loads
- Fails only with certain content

### 2. WORKAROUND PATTERNS
- "Everyone knows you have to..."
- "The trick is to..."
- "For some reason, this works..."
- "Don't ask why, but..."

### 3. UNDEFINED BEHAVIOR
- Documentation says nothing about this
- Different behavior across versions
- Platform-specific differences
- Time-dependent behavior

### 4. EMERGENT CONSTRAINTS
- Limits that appear only in practice
- Community-discovered boundaries
- Interaction effects
- Cascade failures

## EXPLORATION TASK:

Post Content: {content}
Known Patterns: {known_patterns}

## EXPLORATION REQUIREMENTS:

Hunt for the unusual:

1. **Edge Case Type**: What kind of edge case is this?
2. **Discovery Story**: How was this discovered?
3. **Hidden Contract**: What implicit rule exists?
4. **Reproducibility**: How consistent is this?
5. **Workaround Analysis**: What solutions exist?
6. **System Boundaries**: What limits does this reveal?

Respond with edge case analysis:
{{
    "edge_case_classification": {{
        "type": "surprising_failure|undefined_behavior|emergent_limit|interaction_effect",
        "rarity": "how uncommon this is",
        "discovery_difficulty": "how hard to find"
    }},
    "hidden_contract": {{
        "rule": "the implicit requirement",
        "evidence": ["how we know this exists"],
        "consistency": "how reliable this rule is",
        "documentation_status": "documented|undocumented|contradicted"
    }},
    "reproduction_recipe": {{
        "conditions": ["what must be true"],
        "steps": ["how to reproduce"],
        "success_rate": "how often it reproduces"
    }},
    "workaround_ecology": [
        {{
            "workaround": "what people do",
            "effectiveness": "how well it works",
            "side_effects": ["what else happens"],
            "community_adoption": "how widespread"
        }}
    ],
    "boundary_discovery": {{
        "limit_type": "what kind of boundary",
        "measured_value": "specific limit if known",
        "variation": "how it varies",
        "business_impact": "why this matters"
    }},
    "theoretical_implications": {{
        "what_this_reveals": "about LLM API design",
        "design_assumptions": ["what designers assumed"],
        "reality_gap": "difference from intended behavior"
    }},
    "future_research": ["what to investigate next"]
}}"""


def get_multi_stage_discovery_pipeline(title: str, content: str) -> Dict[str, str]:
    """Get prompts for multi-stage discovery pipeline."""
    prompts = ContractDiscoveryPrompts()
    
    return {
        "initial_discovery": prompts.get_open_discovery_prompt().format(
            title=title,
            content=content
        ),
        "deep_analysis": prompts.get_deep_analysis_prompt(),
        "pattern_synthesis": prompts.get_pattern_synthesis_prompt(),
        "edge_exploration": prompts.get_edge_case_explorer_prompt()
    }