"""
Contract Classification Analysis Module.

This module provides tools for analyzing and classifying LLM API contract
violations, with support for both known patterns and novel discovery.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import json
from datetime import datetime

from pipeline.domain.models import (
    ContractType, RootCause, Effect, PipelineStage,
    LLMScreeningResult, FilteredPost
)
from pipeline.llm_screening.contract_taxonomy import (
    LLMContractTaxonomy, ContractDefinition, ViolationSeverity,
    ContractCategory, SingleAPIMethodContract, LLMSpecificContract
)


@dataclass
class ContractViolationAnalysis:
    """Detailed analysis of a contract violation."""
    
    # Core classification
    contract_id: Optional[str] = None  # From taxonomy if matched
    contract_type: Optional[ContractType] = None
    contract_category: Optional[str] = None
    is_novel: bool = False
    
    # Novel contract details (if not in taxonomy)
    novel_name: Optional[str] = None
    novel_description: Optional[str] = None
    novel_category_suggestion: Optional[str] = None
    
    # Evidence and context
    evidence: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    affected_parameters: List[str] = field(default_factory=list)
    
    # Classification confidence
    confidence: float = 0.0
    match_score: float = 0.0  # How well it matches known patterns
    
    # Relationships
    related_contracts: List[str] = field(default_factory=list)
    root_cause: Optional[RootCause] = None
    effects: List[Effect] = field(default_factory=list)
    severity: Optional[ViolationSeverity] = None
    
    # Context
    api_provider: Optional[str] = None
    framework: Optional[str] = None
    pipeline_stage: Optional[PipelineStage] = None


@dataclass
class ContractAnalysisResult:
    """Complete analysis result for a post."""
    
    post_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Violations found
    violations: List[ContractViolationAnalysis] = field(default_factory=list)
    has_violations: bool = False
    
    # Overall classification
    primary_violation: Optional[ContractViolationAnalysis] = None
    total_violations: int = 0
    novel_violations: int = 0
    
    # Pattern analysis
    violation_pattern: Optional[str] = None  # single, multiple, cascade
    pattern_description: Optional[str] = None
    
    # Research value
    research_value_score: float = 0.0
    novelty_score: float = 0.0
    educational_value: str = "low"  # low, medium, high
    
    # Quality metrics
    evidence_quality: str = "weak"  # weak, moderate, strong
    reproducibility: str = "low"  # low, medium, high
    
    # Recommendations
    include_in_dataset: bool = False
    requires_expert_review: bool = False
    notes: str = ""


class ContractAnalyzer:
    """Analyzes posts for contract violations with novel discovery support."""
    
    def __init__(self):
        self.taxonomy = LLMContractTaxonomy()
        self.provider_patterns = self._init_provider_patterns()
        self.framework_patterns = self._init_framework_patterns()
        self.novel_contracts_found = []  # Track novel discoveries
    
    def _init_provider_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting API providers."""
        return {
            "openai": ["openai", "gpt-3", "gpt-4", "chatgpt", "dall-e", "whisper"],
            "anthropic": ["anthropic", "claude", "claude-2", "claude-3"],
            "google": ["google", "palm", "gemini", "bard", "vertex"],
            "cohere": ["cohere", "command", "generate"],
            "huggingface": ["huggingface", "transformers", "inference api"],
            "azure": ["azure", "azure openai", "cognitive services"],
            "aws": ["bedrock", "sagemaker", "aws ai"],
            "meta": ["llama", "llama2", "meta ai"]
        }
    
    def _init_framework_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting frameworks."""
        return {
            "langchain": ["langchain", "chain", "agent", "retriever", "memory"],
            "llamaindex": ["llamaindex", "llama_index", "gpt_index"],
            "openai-python": ["openai.ChatCompletion", "openai.Completion"],
            "autogpt": ["autogpt", "auto-gpt", "autonomous"],
            "semantic-kernel": ["semantic kernel", "sk.", "kernel"],
            "guidance": ["guidance", "handlebars", "{{gen}}"],
            "dspy": ["dspy", "demonstrate", "signature"],
            "custom": ["wrapper", "client", "integration", "sdk"]
        }
    
    def analyze_post(self, post: FilteredPost, llm_result: Optional[LLMScreeningResult] = None) -> ContractAnalysisResult:
        """Analyze a post for contract violations."""
        result = ContractAnalysisResult(post_id=post.id)
        
        # Combine all text for analysis
        text = f"{post.raw_post_id} {' '.join(post.relevant_snippets)} {' '.join(post.potential_contracts)}"
        if llm_result and llm_result.rationale:
            text += f" {llm_result.rationale}"
        
        # Detect provider and framework
        provider = self._detect_provider(text)
        framework = self._detect_framework(text)
        
        # Look for known contract violations
        known_violations = self._find_known_violations(text, provider, framework)
        
        # Look for novel patterns
        novel_violations = self._discover_novel_violations(
            text, 
            known_violations,
            llm_result
        )
        
        # Combine all violations
        all_violations = known_violations + novel_violations
        result.violations = all_violations
        result.has_violations = len(all_violations) > 0
        result.total_violations = len(all_violations)
        result.novel_violations = len(novel_violations)
        
        # Analyze patterns
        if result.has_violations:
            result.violation_pattern = self._analyze_pattern(all_violations)
            result.primary_violation = self._identify_primary_violation(all_violations)
            
            # Calculate research value
            result.research_value_score = self._calculate_research_value(
                all_violations,
                novel_violations,
                post
            )
            result.novelty_score = len(novel_violations) / len(all_violations) if all_violations else 0
            
            # Set recommendations
            result.include_in_dataset = result.research_value_score > 0.5
            result.requires_expert_review = (
                result.novel_violations > 0 or 
                result.research_value_score > 0.8 or
                any(v.severity == ViolationSeverity.CRITICAL for v in all_violations)
            )
        
        return result
    
    def _detect_provider(self, text: str) -> Optional[str]:
        """Detect API provider from text."""
        text_lower = text.lower()
        for provider, patterns in self.provider_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return provider
        return None
    
    def _detect_framework(self, text: str) -> Optional[str]:
        """Detect framework from text."""
        text_lower = text.lower()
        for framework, patterns in self.framework_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return framework
        return None
    
    def _find_known_violations(self, text: str, provider: Optional[str], framework: Optional[str]) -> List[ContractViolationAnalysis]:
        """Find violations matching known patterns."""
        violations = []
        
        # Check against taxonomy
        matches = self.taxonomy.identify_violations(text)
        
        for contract_def, score in matches:
            if score > 0.3:  # Confidence threshold
                violation = ContractViolationAnalysis(
                    contract_id=contract_def.id,
                    contract_type=self._map_to_contract_type(contract_def),
                    contract_category=contract_def.category.value,
                    is_novel=False,
                    confidence=score,
                    match_score=score,
                    severity=contract_def.severity,
                    api_provider=provider,
                    framework=framework
                )
                
                # Extract evidence
                violation.evidence = self._extract_evidence(text, contract_def.error_patterns)
                violation.affected_parameters = contract_def.parameters
                
                violations.append(violation)
        
        return violations
    
    def _discover_novel_violations(self, 
                                  text: str, 
                                  known_violations: List[ContractViolationAnalysis],
                                  llm_result: Optional[LLMScreeningResult]) -> List[ContractViolationAnalysis]:
        """Discover novel contract violations not in taxonomy."""
        novel_violations = []
        
        # Look for error patterns not matched by known contracts
        error_indicators = [
            r"error[:\s]*(.*?)(?:\n|$)",
            r"failed[:\s]*(.*?)(?:\n|$)",
            r"exception[:\s]*(.*?)(?:\n|$)",
            r"invalid[:\s]*(.*?)(?:\n|$)",
            r"violated[:\s]*(.*?)(?:\n|$)",
            r"constraint[:\s]*(.*?)(?:\n|$)",
            r"limit[:\s]*(.*?)(?:\n|$)",
            r"required[:\s]*(.*?)(?:\n|$)",
            r"must[:\s]*(.*?)(?:\n|$)",
            r"cannot[:\s]*(.*?)(?:\n|$)",
            r"unsupported[:\s]*(.*?)(?:\n|$)",
            r"deprecated[:\s]*(.*?)(?:\n|$)"
        ]
        
        unmatched_errors = []
        for pattern in error_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                error_text = match.group(0)
                # Check if this error is already covered by known violations
                if not any(error_text.lower() in ' '.join(v.evidence).lower() for v in known_violations):
                    unmatched_errors.append(error_text)
        
        # Analyze unmatched errors for novel contracts
        if unmatched_errors:
            novel_violation = self._create_novel_violation(
                unmatched_errors,
                text,
                llm_result
            )
            if novel_violation:
                novel_violations.append(novel_violation)
                self.novel_contracts_found.append(novel_violation)
        
        # Check LLM-identified violations not in taxonomy
        if llm_result and llm_result.contract_violations:
            for llm_violation in llm_result.contract_violations:
                if not self._is_known_pattern(llm_violation):
                    novel = ContractViolationAnalysis(
                        is_novel=True,
                        novel_name=llm_violation.get("name", "Unknown Novel Contract"),
                        novel_description=llm_violation.get("description", ""),
                        evidence=[llm_violation.get("evidence", "")],
                        confidence=llm_violation.get("confidence", 0.5)
                    )
                    novel_violations.append(novel)
        
        return novel_violations
    
    def _create_novel_violation(self, 
                               error_texts: List[str], 
                               full_text: str,
                               llm_result: Optional[LLMScreeningResult]) -> Optional[ContractViolationAnalysis]:
        """Create a novel violation analysis from unmatched errors."""
        if not error_texts:
            return None
        
        # Try to categorize the novel violation
        novel_category = self._guess_novel_category(error_texts, full_text)
        
        violation = ContractViolationAnalysis(
            is_novel=True,
            novel_name=f"Novel_{novel_category}_{len(self.novel_contracts_found)}",
            novel_description=f"Undocumented constraint: {error_texts[0][:100]}...",
            novel_category_suggestion=novel_category,
            evidence=error_texts[:3],  # Top 3 examples
            confidence=0.6,  # Medium confidence for novel
            severity=self._estimate_severity(error_texts, full_text)
        )
        
        return violation
    
    def _guess_novel_category(self, errors: List[str], text: str) -> str:
        """Guess category for novel violation."""
        error_text = ' '.join(errors).lower()
        
        # Category heuristics
        if any(word in error_text for word in ["rate", "limit", "quota", "throttle"]):
            return "NovelRateLimit"
        elif any(word in error_text for word in ["format", "structure", "schema", "json"]):
            return "NovelFormat"
        elif any(word in error_text for word in ["policy", "safety", "content", "filter"]):
            return "NovelPolicy"
        elif any(word in error_text for word in ["state", "context", "memory", "history"]):
            return "NovelState"
        elif any(word in error_text for word in ["cost", "price", "billing", "budget"]):
            return "NovelCost"
        elif any(word in error_text for word in ["version", "deprecated", "compatibility"]):
            return "NovelCompatibility"
        elif any(word in error_text for word in ["performance", "latency", "timeout", "slow"]):
            return "NovelPerformance"
        else:
            return "NovelUnknown"
    
    def _estimate_severity(self, errors: List[str], text: str) -> ViolationSeverity:
        """Estimate severity of novel violation."""
        error_text = ' '.join(errors).lower()
        
        # Severity heuristics
        if any(word in error_text for word in ["crash", "fatal", "critical", "security"]):
            return ViolationSeverity.CRITICAL
        elif any(word in error_text for word in ["error", "failed", "invalid", "denied"]):
            return ViolationSeverity.HIGH
        elif any(word in error_text for word in ["warning", "deprecated", "slow"]):
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def _is_known_pattern(self, llm_violation: Dict[str, Any]) -> bool:
        """Check if LLM-identified violation matches known patterns."""
        violation_text = str(llm_violation).lower()
        
        # Check against all known contracts
        for contract in self.taxonomy.contracts.values():
            for pattern in contract.error_patterns:
                if pattern.lower() in violation_text:
                    return True
        
        return False
    
    def _extract_evidence(self, text: str, patterns: List[str]) -> List[str]:
        """Extract evidence snippets for violations."""
        evidence = []
        text_lines = text.split('\n')
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            for i, line in enumerate(text_lines):
                if pattern_lower in line.lower():
                    # Get context (line before and after)
                    start = max(0, i - 1)
                    end = min(len(text_lines), i + 2)
                    context = '\n'.join(text_lines[start:end])
                    evidence.append(context.strip())
                    break
        
        return evidence[:3]  # Limit to 3 pieces of evidence
    
    def _map_to_contract_type(self, contract_def: ContractDefinition) -> Optional[ContractType]:
        """Map taxonomy contract to domain model contract type."""
        # This is a simplified mapping - could be more sophisticated
        mapping = {
            "max_tokens_limit": ContractType.MAX_TOKENS,
            "temperature_range": ContractType.TEMPERATURE,
            "rate_limit_rpm": ContractType.RATE_LIMIT,
            "content_policy": ContractType.CONTENT_POLICY,
            "message_format": ContractType.PROMPT_FORMAT,
            "json_output_format": ContractType.JSON_SCHEMA,
            "context_window": ContractType.CONTEXT_LENGTH,
            "api_key_auth": ContractType.API_KEY_FORMAT
        }
        
        return mapping.get(contract_def.id)
    
    def _analyze_pattern(self, violations: List[ContractViolationAnalysis]) -> str:
        """Analyze pattern of violations."""
        if len(violations) == 0:
            return "none"
        elif len(violations) == 1:
            return "single"
        elif len(violations) == 2:
            return "double"
        else:
            # Check for cascade pattern (one violation causing others)
            if self._is_cascade_pattern(violations):
                return "cascade"
            else:
                return "multiple"
    
    def _is_cascade_pattern(self, violations: List[ContractViolationAnalysis]) -> bool:
        """Check if violations show cascade pattern."""
        # Simple heuristic: rate limit often causes other failures
        has_rate_limit = any(
            v.contract_type == ContractType.RATE_LIMIT or 
            "rate" in (v.novel_name or "").lower() 
            for v in violations
        )
        
        return has_rate_limit and len(violations) > 2
    
    def _identify_primary_violation(self, violations: List[ContractViolationAnalysis]) -> ContractViolationAnalysis:
        """Identify the primary/root violation."""
        if not violations:
            return None
        
        # Sort by severity and confidence
        sorted_violations = sorted(
            violations,
            key=lambda v: (
                v.severity.value if v.severity else 0,
                v.confidence,
                1.0 if v.is_novel else 0.0  # Prefer novel
            ),
            reverse=True
        )
        
        return sorted_violations[0]
    
    def _calculate_research_value(self, 
                                 all_violations: List[ContractViolationAnalysis],
                                 novel_violations: List[ContractViolationAnalysis],
                                 post: FilteredPost) -> float:
        """Calculate research value score."""
        score = 0.0
        
        # Novelty weight (40%)
        novelty_score = len(novel_violations) / len(all_violations) if all_violations else 0
        score += novelty_score * 0.4
        
        # Severity weight (20%)
        max_severity = max(
            (v.severity.value if v.severity else 0 for v in all_violations),
            default=0
        )
        severity_score = max_severity / 4.0  # Normalize to 0-1
        score += severity_score * 0.2
        
        # Evidence quality (20%)
        avg_confidence = sum(v.confidence for v in all_violations) / len(all_violations) if all_violations else 0
        score += avg_confidence * 0.2
        
        # Post quality (20%)
        post_quality = post.quality_score
        score += post_quality * 0.2
        
        return min(1.0, score)
    
    def get_novel_contracts_summary(self) -> Dict[str, Any]:
        """Get summary of all novel contracts discovered."""
        if not self.novel_contracts_found:
            return {"count": 0, "categories": {}}
        
        # Group by category
        categories = defaultdict(list)
        for violation in self.novel_contracts_found:
            category = violation.novel_category_suggestion or "Unknown"
            categories[category].append({
                "name": violation.novel_name,
                "description": violation.novel_description,
                "evidence_count": len(violation.evidence),
                "severity": violation.severity.value if violation.severity else "unknown"
            })
        
        return {
            "count": len(self.novel_contracts_found),
            "categories": dict(categories),
            "top_patterns": self._identify_top_novel_patterns()
        }
    
    def _identify_top_novel_patterns(self) -> List[Dict[str, Any]]:
        """Identify most common novel patterns."""
        # Count evidence patterns
        pattern_counter = Counter()
        
        for violation in self.novel_contracts_found:
            for evidence in violation.evidence:
                # Extract key terms
                terms = re.findall(r'\b\w+\b', evidence.lower())
                for term in terms:
                    if len(term) > 3:  # Skip short words
                        pattern_counter[term] += 1
        
        # Get top patterns
        top_patterns = []
        for term, count in pattern_counter.most_common(10):
            top_patterns.append({
                "pattern": term,
                "frequency": count,
                "percentage": count / len(self.novel_contracts_found) * 100
            })
        
        return top_patterns


# Utility function for batch analysis
def analyze_post_batch(posts: List[FilteredPost], 
                      llm_results: Optional[Dict[str, LLMScreeningResult]] = None) -> List[ContractAnalysisResult]:
    """Analyze a batch of posts."""
    analyzer = ContractAnalyzer()
    results = []
    
    for post in posts:
        llm_result = llm_results.get(post.id) if llm_results else None
        result = analyzer.analyze_post(post, llm_result)
        results.append(result)
    
    # Print novel contracts summary
    novel_summary = analyzer.get_novel_contracts_summary()
    if novel_summary["count"] > 0:
        print(f"\nDiscovered {novel_summary['count']} novel contract patterns:")
        for category, contracts in novel_summary["categories"].items():
            print(f"  {category}: {len(contracts)} patterns")
    
    return results