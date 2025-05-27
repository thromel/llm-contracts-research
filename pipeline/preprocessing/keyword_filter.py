"""
Advanced Keyword Pre-Filter for LLM Contracts Research Pipeline.

Multi-stage filtering system with semantic categories, context analysis,
and intelligent scoring for high-precision LLM contract violation detection.
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..common.models import RawPost, FilteredPost
from ..common.database import MongoDBManager, ProvenanceTracker

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of keyword filtering."""
    passed: bool
    confidence: float
    matched_keywords: List[str]
    relevant_snippets: List[str]
    potential_contracts: List[str]
    filter_metadata: Dict[str, Any]


class AdvancedKeywordFilter:
    """
    Advanced keyword-based pre-filter for LLM contract research.

    Features:
    - Multi-category semantic filtering
    - Context-aware pattern matching
    - Confidence scoring with multiple signals
    - Contract violation pattern detection
    - Noise reduction through negative filtering
    """

    def __init__(self, db_manager: MongoDBManager):
        """Initialize advanced keyword filter.

        Args:
            db_manager: MongoDB manager for storage
        """
        self.db = db_manager
        self.provenance = ProvenanceTracker(db_manager)

        # === POSITIVE SIGNALS - Organized by semantic categories ===

        # Core LLM API contract violation indicators (HIGH VALUE)
        self.contract_violation_keywords = {
            # Parameter constraint violations
            'max_tokens', 'temperature', 'top_p', 'top_k', 'frequency_penalty',
            'presence_penalty', 'logprobs', 'n_completions', 'best_of',
            'context_length', 'context_window', 'sequence_length',

            # Rate limiting and quota violations
            'rate_limit', 'rate_limited', 'quota_exceeded', 'quota_limit',
            'rpm', 'tpm', 'requests_per_minute', 'tokens_per_minute',
            'billing_exceeded', 'usage_limit', 'throttling', 'throttled',

            # Format and schema violations
            'json_schema', 'response_format', 'function_calling', 'tool_calling',
            'structured_output', 'json_mode', 'schema_validation',
            'invalid_json', 'malformed_json', 'parse_error', 'json_decode_error',

            # Content policy violations
            'content_policy', 'policy_violation', 'content_filter',
            'safety_filter', 'moderation_filter', 'flagged_content',
            'inappropriate_content', 'blocked_content',

            # Authentication and authorization
            'api_key', 'invalid_api_key', 'unauthorized', 'forbidden',
            'authentication_failed', 'access_denied', 'permission_denied',

            # Token and context issues
            'token_limit', 'token_count', 'context_overflow', 'input_too_long',
            'prompt_too_long', 'context_length_exceeded', 'truncated',
        }

        # LLM API and framework terms (MEDIUM VALUE)
        self.llm_api_keywords = {
            # Core LLM providers
            'openai', 'anthropic', 'claude', 'gpt', 'chatgpt', 'gpt-4', 'gpt-3.5',
            'gemini', 'palm', 'bard', 'azure_openai', 'bedrock',

            # Popular frameworks
            'langchain', 'llamaindex', 'llama_index', 'semantic_kernel',
            'guidance', 'autogen', 'crewai', 'haystack',

            # Vector databases and embeddings
            'embeddings', 'vector_database', 'pinecone', 'chroma', 'weaviate',
            'qdrant', 'milvus', 'faiss', 'pgvector',

            # Application frameworks
            'streamlit', 'gradio', 'chainlit', 'modal', 'vercel_ai',
        }

        # Error and problem indicators (MEDIUM VALUE)
        self.error_keywords = {
            'error', 'exception', 'failed', 'failure', 'bug', 'issue',
            'problem', 'trouble', 'broken', 'not_working', 'doesnt_work',
            'timeout', 'connection_error', 'network_error', 'server_error',
            'http_error', 'api_error', 'request_failed', 'response_error',
            'invalid_request', 'bad_request', 'status_code', '400', '401',
            '403', '429', '500', '502', '503', '504',
        }

        # Technical implementation terms (LOW VALUE - context dependent)
        self.technical_keywords = {
            'api', 'sdk', 'client', 'library', 'package', 'module',
            'implementation', 'integration', 'configuration', 'setup',
            'parameters', 'arguments', 'response', 'request', 'headers',
            'payload', 'endpoint', 'url', 'method', 'get', 'post',
        }

        # === NEGATIVE SIGNALS - Filter out non-contract-related content ===

        self.negative_keywords = {
            # General installation/setup (unless combined with errors)
            'how_to_install', 'installation_guide', 'getting_started',
            'tutorial', 'beginner_guide', 'basic_usage', 'hello_world',

            # General conceptual questions
            'what_is', 'difference_between', 'comparison', 'vs', 'versus',
            'pros_and_cons', 'advantages', 'disadvantages', 'best_practices',

            # Non-API related content
            'training_data', 'fine_tuning_guide', 'model_architecture',
            'research_paper', 'academic_study', 'thesis', 'publication',

            # General programming help (unless API-specific)
            'python_basics', 'javascript_basics', 'coding_help',
            'programming_tutorial', 'syntax_help', 'variable_declaration',
        }

        # === CONTEXT PATTERNS - Regex patterns for complex matching ===

        self.contract_patterns = [
            # Parameter validation errors
            r'(?i)(max_tokens|temperature|top_p).*(?:must|should|cannot|exceeds?|limit|invalid|error)',
            r'(?i)(rate.?limit|quota).*(?:exceeded|error|reached|violation|denied)',
            r'(?i)(context.?length|token.?limit).*(?:exceeded|too.?long|maximum|overflow)',

            # API error patterns
            r'(?i)(json|schema).*(?:invalid|error|malformed|parse.*fail|validation.*fail)',
            r'(?i)(api.?key|token).*(?:invalid|expired|missing|unauthorized|forbidden)',
            r'(?i)(status.?code|http).*(?:400|401|403|429|500|502|503)',

            # Content policy patterns
            r'(?i)(content|usage).?policy.*(?:violation|flagged|denied|blocked)',
            r'(?i)(safety|moderation).*(?:filter|blocked|flagged|violation)',

            # Functional errors
            r'(?i)(function.?call|tool.?use).*(?:error|failed|invalid|malformed)',
            r'(?i)(stream|streaming).*(?:error|failed|interrupted|timeout)',
        ]

        # === QUALITY INDICATORS ===

        self.quality_indicators = {
            # High-quality content signals
            'positive': {
                'error_message', 'stack_trace', 'traceback', 'exception_details',
                'code_example', 'minimal_reproduction', 'steps_to_reproduce',
                'expected_behavior', 'actual_behavior', 'workaround', 'solution',
                'resolved', 'fixed', 'working_solution',
            },

            # Low-quality content signals
            'negative': {
                'urgent', 'please_help', 'asap', 'desperate', 'stuck',
                'no_idea', 'completely_lost', 'beginner', 'newbie', 'noob',
                'simple_question', 'quick_question', 'dumb_question',
            }
        }

    async def filter_batch(
        self,
        batch_size: int = 1000,
        confidence_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """Filter a batch of raw posts using advanced filtering.

        Args:
            batch_size: Number of posts to process in this batch
            confidence_threshold: Minimum confidence to pass filter

        Returns:
            Batch processing statistics
        """
        stats = {
            'processed': 0,
            'passed': 0,
            'failed': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'categories_matched': {},
            'patterns_matched': {},
            'processing_time': 0
        }

        start_time = datetime.utcnow()

        # Get unfiltered posts
        async for raw_post_dict in self.db.get_posts_for_filtering(batch_size=batch_size):
            try:
                # Convert to RawPost object
                raw_post = RawPost(
                    _id=str(raw_post_dict['_id']),
                    platform=raw_post_dict['platform'],
                    source_id=raw_post_dict['source_id'],
                    title=raw_post_dict.get('title', ''),
                    body_md=raw_post_dict.get('body_md', ''),
                    tags=raw_post_dict.get('tags', []),
                    labels=raw_post_dict.get('labels', [])
                )

                # Apply advanced filter
                filter_result = self.apply_filter(raw_post)

                # Only save if confidence meets threshold
                if filter_result.confidence >= confidence_threshold:
                    # Create FilteredPost
                    filtered_post = FilteredPost(
                        raw_post_id=raw_post._id,
                        passed_keyword_filter=filter_result.passed,
                        matched_keywords=filter_result.matched_keywords,
                        filter_confidence=filter_result.confidence,
                        relevant_snippets=filter_result.relevant_snippets,
                        potential_contracts=filter_result.potential_contracts,
                        filter_timestamp=datetime.utcnow(),
                        filter_version="2.0.0"
                    )

                    # Save to database
                    filtered_id = await self.db.save_filtered_post(filtered_post.to_dict())

                    # Log provenance
                    await self.provenance.log_transformation(
                        source_id=raw_post._id,
                        source_collection='raw_posts',
                        target_id=filtered_id,
                        target_collection='filtered_posts',
                        transformation_type='advanced_keyword_filter',
                        metadata=filter_result.filter_metadata
                    )

                # Update stats
                stats['processed'] += 1
                if filter_result.passed:
                    stats['passed'] += 1
                    if filter_result.confidence >= 0.8:
                        stats['high_confidence'] += 1
                    elif filter_result.confidence >= 0.6:
                        stats['medium_confidence'] += 1
                    else:
                        stats['low_confidence'] += 1
                else:
                    stats['failed'] += 1

                # Track category usage
                categories = filter_result.filter_metadata.get(
                    'categories_matched', {})
                for category, count in categories.items():
                    stats['categories_matched'][category] = stats['categories_matched'].get(
                        category, 0) + count

                # Track pattern usage
                patterns = filter_result.filter_metadata.get(
                    'patterns_matched', 0)
                stats['patterns_matched']['total'] = stats['patterns_matched'].get(
                    'total', 0) + patterns

            except Exception as e:
                logger.error(
                    f"Error filtering post {raw_post_dict.get('_id', 'unknown')}: {str(e)}")
                continue

        stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()
        return stats

    def apply_filter(self, raw_post: RawPost) -> FilterResult:
        """Apply advanced multi-stage filter to a single post.

        Args:
            raw_post: Raw post to filter

        Returns:
            FilterResult with pass/fail decision and detailed metadata
        """
        # Combine all text content
        combined_text = f"{raw_post.title} {raw_post.body_md}"
        combined_text_lower = combined_text.lower()

        # Stage 1: Category-based keyword matching
        contract_matches = self._find_keywords(
            combined_text_lower, self.contract_violation_keywords)
        llm_api_matches = self._find_keywords(
            combined_text_lower, self.llm_api_keywords)
        error_matches = self._find_keywords(
            combined_text_lower, self.error_keywords)
        technical_matches = self._find_keywords(
            combined_text_lower, self.technical_keywords)

        # Stage 2: Negative signal detection
        negative_matches = self._find_keywords(
            combined_text_lower, self.negative_keywords)

        # Stage 3: Pattern matching for complex contract violations
        pattern_matches = self._find_contract_patterns(combined_text)

        # Stage 4: Tag and label analysis
        tag_matches = self._check_tags_and_labels(
            raw_post.tags + raw_post.labels)

        # Stage 5: Quality assessment
        quality_score = self._assess_quality(combined_text_lower)

        # Stage 6: Calculate weighted confidence score
        confidence = self._calculate_advanced_confidence(
            contract_matches, llm_api_matches, error_matches, technical_matches,
            negative_matches, pattern_matches, tag_matches, quality_score,
            combined_text_lower
        )

        # Stage 7: Make pass/fail decision with sophisticated logic
        passed = self._make_decision(
            contract_matches, llm_api_matches, error_matches,
            negative_matches, pattern_matches, confidence
        )

        # Stage 8: Extract relevant snippets and potential contracts
        all_matches = contract_matches + llm_api_matches + error_matches
        relevant_snippets = self._extract_snippets(
            combined_text, all_matches, context_size=150)
        potential_contracts = self._identify_potential_contracts(
            combined_text, all_matches, pattern_matches)

        # Comprehensive metadata
        filter_metadata = {
            'categories_matched': {
                'contract_violations': len(contract_matches),
                'llm_api_terms': len(llm_api_matches),
                'error_indicators': len(error_matches),
                'technical_terms': len(technical_matches),
                'negative_signals': len(negative_matches),
                'tag_matches': len(tag_matches)
            },
            'patterns_matched': len(pattern_matches),
            'quality_score': quality_score,
            'text_length': len(combined_text),
            'word_count': len(combined_text.split()),
            'decision_factors': {
                'has_contract_terms': len(contract_matches) > 0,
                'has_error_context': len(error_matches) > 0,
                'has_negative_signals': len(negative_matches) > 0,
                'has_pattern_matches': len(pattern_matches) > 0,
                'quality_threshold_met': quality_score > 0.3
            }
        }

        return FilterResult(
            passed=passed,
            confidence=confidence,
            matched_keywords=all_matches,
            relevant_snippets=relevant_snippets,
            potential_contracts=potential_contracts,
            filter_metadata=filter_metadata
        )

    def _find_keywords(self, text: str, keyword_set: Set[str]) -> List[str]:
        """Find matching keywords with word boundary checking."""
        matches = []
        for keyword in keyword_set:
            # Create flexible pattern for compound keywords
            pattern = keyword.replace('_', '[_\\s-]?')
            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                matches.append(keyword)
        return matches

    def _find_contract_patterns(self, text: str) -> List[Tuple[str, int, int]]:
        """Find complex contract violation patterns using regex."""
        matches = []
        for pattern in self.contract_patterns:
            for match in re.finditer(pattern, text):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def _check_tags_and_labels(self, tags_and_labels: List[str]) -> List[str]:
        """Check tags and labels for relevant terms."""
        matches = []
        all_keywords = (self.contract_violation_keywords |
                        self.llm_api_keywords |
                        self.error_keywords)

        for tag in tags_and_labels:
            tag_lower = tag.lower().replace('-', '_')
            for keyword in all_keywords:
                if keyword in tag_lower or tag_lower in keyword:
                    matches.append(tag)
                    break
        return matches

    def _assess_quality(self, text: str) -> float:
        """Assess content quality based on positive and negative indicators."""
        positive_matches = self._find_keywords(
            text, self.quality_indicators['positive'])
        negative_matches = self._find_keywords(
            text, self.quality_indicators['negative'])

        # Base quality score
        quality = 0.5

        # Boost for positive indicators
        quality += len(positive_matches) * 0.1

        # Penalty for negative indicators
        quality -= len(negative_matches) * 0.15

        # Bonus for having code blocks, error messages, etc.
        if re.search(r'```|`[^`]+`|error:|exception:|traceback:', text):
            quality += 0.2

        # Bonus for having specific numbers/values (often in error messages)
        if re.search(r'\b\d+\b', text):
            quality += 0.1

        return max(0.0, min(1.0, quality))

    def _calculate_advanced_confidence(
        self,
        contract_matches: List[str],
        llm_api_matches: List[str],
        error_matches: List[str],
        technical_matches: List[str],
        negative_matches: List[str],
        pattern_matches: List[Tuple[str, int, int]],
        tag_matches: List[str],
        quality_score: float,
        text: str
    ) -> float:
        """Calculate sophisticated confidence score with multiple signals."""

        # Base confidence from different categories (weighted)
        # High weight for contract terms
        contract_score = min(len(contract_matches) * 0.3, 0.6)
        # Medium weight for API terms
        llm_api_score = min(len(llm_api_matches) * 0.15, 0.3)
        # Lower weight for generic errors
        error_score = min(len(error_matches) * 0.1, 0.2)
        pattern_score = min(len(pattern_matches) * 0.25,
                            0.5)   # High weight for patterns
        # Bonus for relevant tags
        tag_score = min(len(tag_matches) * 0.1, 0.2)

        # Combination bonuses
        combination_bonus = 0.0
        if contract_matches and error_matches:  # Contract terms + errors = strong signal
            combination_bonus += 0.2
        # API + issues = good signal
        if llm_api_matches and (contract_matches or error_matches):
            combination_bonus += 0.15
        # Patterns + context = very strong
        if pattern_matches and (contract_matches or error_matches):
            combination_bonus += 0.25

        # Quality adjustment
        quality_adjustment = (quality_score - 0.5) * 0.3  # -0.15 to +0.15

        # Negative signal penalty
        negative_penalty = min(len(negative_matches) * 0.2, 0.4)

        # Calculate final confidence
        confidence = (contract_score + llm_api_score + error_score + pattern_score +
                      tag_score + combination_bonus + quality_adjustment - negative_penalty)

        return max(0.0, min(1.0, confidence))

    def _make_decision(
        self,
        contract_matches: List[str],
        llm_api_matches: List[str],
        error_matches: List[str],
        negative_matches: List[str],
        pattern_matches: List[Tuple[str, int, int]],
        confidence: float
    ) -> bool:
        """Make sophisticated pass/fail decision."""

        # High confidence threshold
        if confidence >= 0.7:
            return True

        # Strong pattern matches (even with lower confidence)
        if len(pattern_matches) >= 2:
            return True

        # Strong contract violation signals
        if len(contract_matches) >= 2 and len(error_matches) >= 1:
            return True

        # Multiple LLM API terms with some error context
        if len(llm_api_matches) >= 2 and len(error_matches) >= 1 and len(negative_matches) == 0:
            return True

        # Medium confidence with good signal quality
        if confidence >= 0.5 and len(negative_matches) == 0:
            return True

        # Default threshold
        return confidence >= 0.4

    def _extract_snippets(self, text: str, keywords: List[str], context_size: int = 150) -> List[str]:
        """Extract relevant text snippets around matched keywords."""
        snippets = []
        text_lower = text.lower()

        for keyword in keywords[:5]:  # Limit to top 5 keywords
            keyword_lower = keyword.lower().replace('_', '[_\\s-]?')
            match = re.search(keyword_lower, text_lower)
            if match:
                start = max(0, match.start() - context_size)
                end = min(len(text), match.end() + context_size)
                snippet = text[start:end].strip()
                if len(snippet) > 50 and snippet not in snippets:
                    snippets.append(snippet)

        return snippets[:3]  # Return top 3 snippets

    def _identify_potential_contracts(
        self,
        text: str,
        keywords: List[str],
        pattern_matches: List[Tuple[str, int, int]]
    ) -> List[str]:
        """Identify potential contract violations from context."""
        contracts = []

        # Extract contracts from pattern matches
        for pattern_match, start, end in pattern_matches:
            contracts.append(f"Pattern: {pattern_match.strip()}")

        # Look for specific parameter mentions
        param_patterns = [
            r'(?i)(max_tokens|temperature|top_p)\s*[=:]\s*[\d.]+',
            r'(?i)(rate.?limit|quota).*?(\d+)',
            r'(?i)(context.?length).*?(\d+)',
        ]

        for pattern in param_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    contracts.append(
                        f"Parameter constraint: {' '.join(match)}")
                else:
                    contracts.append(f"Parameter constraint: {match}")

        return contracts[:5]  # Return top 5 potential contracts

    async def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        # Implementation for getting statistics from database
        pass  # Would implement database queries for stats


# Maintain backward compatibility
KeywordPreFilter = AdvancedKeywordFilter
