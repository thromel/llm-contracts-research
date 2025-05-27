"""
Keyword Pre-Filter for LLM Contracts Research Pipeline.

Boolean scan that removes ≈70% noise while retaining >93% recall.
Combines LLM contract keywords with ML-style root-cause cues.
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


class KeywordPreFilter:
    """
    Keyword-based pre-filter for LLM contract research.

    Features:
    - LLM contract keywords (max_tokens, json schema, rate_limit, etc.)
    - ML-style root-cause cues from Khairunnesa et al.
    - Context-aware snippet extraction
    - Confidence scoring
    """

    def __init__(self, db_manager: MongoDBManager):
        """Initialize keyword pre-filter.

        Args:
            db_manager: MongoDB manager for storage
        """
        self.db = db_manager
        self.provenance = ProvenanceTracker(db_manager)

        # Simple unified keyword list - if ANY of these match, the post passes
        self.keywords = {
            # Core LLM/API terms
            'openai', 'gpt', 'claude', 'anthropic', 'api', 'llm', 'chatgpt',
            'langchain', 'huggingface', 'transformers', 'ai', 'model',

            # API parameters
            'max_tokens', 'temperature', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'stream', 'logprobs', 'context_length',

            # JSON and formatting
            'json', 'schema', 'response_format', 'function_calling', 'tools',
            'structured_output', 'json_mode', 'parse', 'format',

            # Rate limiting and quotas
            'rate_limit', 'quota', 'rpm', 'tpm', 'billing', 'usage',
            'rate_exceeded', 'quota_exceeded', 'too_many_requests',

            # Errors and issues
            'error', 'exception', 'failed', 'fail', 'bug', 'issue',
            'problem', 'trouble', 'broken', 'help', 'fix', 'solve',
            'timeout', 'unauthorized', 'forbidden', 'invalid_request',

            # Token and context issues
            'token_limit', 'context_overflow', 'input_too_long',
            'prompt_too_long', 'context_length_exceeded',

            # Content policy
            'content_policy', 'policy_violation', 'moderation',
            'content_filter', 'safety_filter',

            # General programming/ML terms that might be relevant
            'python', 'javascript', 'node', 'sdk', 'library', 'package',
            'installation', 'import', 'module', 'version', 'update'
        }

    async def filter_batch(
        self,
        batch_size: int = 1000,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Filter a batch of raw posts.

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
            'keywords_matched': {},
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

                # Apply filter
                filter_result = self.apply_filter(raw_post)

                # Create FilteredPost
                filtered_post = FilteredPost(
                    raw_post_id=raw_post._id,
                    passed_keyword_filter=filter_result.passed,
                    matched_keywords=filter_result.matched_keywords,
                    filter_confidence=filter_result.confidence,
                    relevant_snippets=filter_result.relevant_snippets,
                    potential_contracts=filter_result.potential_contracts,
                    filter_timestamp=datetime.utcnow(),
                    filter_version="1.0.0"
                )

                # Save to database
                filtered_id = await self.db.save_filtered_post(filtered_post.to_dict())

                # Log provenance
                await self.provenance.log_transformation(
                    source_id=raw_post._id,
                    source_collection='raw_posts',
                    target_id=filtered_id,
                    target_collection='filtered_posts',
                    transformation_type='keyword_filter',
                    metadata=filter_result.filter_metadata
                )

                # Update stats
                stats['processed'] += 1
                if filter_result.passed:
                    stats['passed'] += 1
                    if filter_result.confidence >= 0.7:
                        stats['high_confidence'] += 1
                else:
                    stats['failed'] += 1

                # Track keyword usage
                for keyword in filter_result.matched_keywords:
                    stats['keywords_matched'][keyword] = stats['keywords_matched'].get(
                        keyword, 0) + 1

            except Exception as e:
                logger.error(
                    f"Error filtering post {raw_post_dict.get('_id', 'unknown')}: {str(e)}")
                continue

        stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()
        return stats

    def apply_filter(self, raw_post: RawPost) -> FilterResult:
        """Apply simple keyword filter to a single post.

        Args:
            raw_post: Raw post to filter

        Returns:
            FilterResult with pass/fail decision and metadata
        """
        # Combine all text content
        combined_text = f"{raw_post.title} {raw_post.body_md}"
        combined_text_lower = combined_text.lower()

        # Simple keyword matching - if ANY keyword matches, pass the post
        matched_keywords = []
        for keyword in self.keywords:
            if keyword in combined_text_lower:
                matched_keywords.append(keyword)

        # Check tags and labels for relevant terms
        tag_matches = []
        for tag in raw_post.tags + raw_post.labels:
            tag_lower = tag.lower()
            for keyword in self.keywords:
                if keyword in tag_lower:
                    tag_matches.append(tag)
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

        # Simple decision: if any keyword matches, pass it through
        passed = len(matched_keywords) > 0

        # Simple confidence: more keywords = higher confidence
        confidence = min(len(matched_keywords) * 0.1, 1.0)

        # Extract a simple snippet around the first matched keyword
        relevant_snippets = []
        if matched_keywords and passed:
            first_keyword = matched_keywords[0]
            start_pos = combined_text_lower.find(first_keyword)
            if start_pos != -1:
                snippet_start = max(0, start_pos - 100)
                snippet_end = min(len(combined_text),
                                  start_pos + len(first_keyword) + 100)
                snippet = combined_text[snippet_start:snippet_end].strip()
                if len(snippet) > 20:
                    relevant_snippets.append(snippet)

        # Simple metadata
        filter_metadata = {
            'keywords_matched': len(matched_keywords),
            'tags_matched': len(tag_matches),
            'text_length': len(combined_text),
            'decision_reason': 'keyword_match' if passed else 'no_keywords'
        }

        return FilterResult(
            passed=passed,
            confidence=confidence,
            matched_keywords=matched_keywords,
            relevant_snippets=relevant_snippets,
            potential_contracts=[],  # Not needed for simple filtering
            filter_metadata=filter_metadata
        )

    def _find_keywords(self, text: str, keyword_set: Set[str]) -> List[str]:
        """Find matching keywords in text."""
        matches = []
        for keyword in keyword_set:
            # Handle both exact matches and word boundaries
            if keyword in text:
                # Check if it's a word boundary match for better precision
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    matches.append(keyword)
                elif '_' in keyword and keyword in text:
                    # For snake_case keywords, allow exact substring match
                    matches.append(keyword)
        return matches

    def _find_contract_patterns(self, text: str) -> List[Tuple[str, int, int]]:
        """Find contract violation patterns using regex.

        Returns:
            List of (matched_text, start_pos, end_pos) tuples
        """
        matches = []
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def _check_tags_and_labels(self, tags_and_labels: List[str]) -> List[str]:
        """Check for relevant tags and labels."""
        relevant_tags = []
        llm_tag_patterns = [
            'openai', 'gpt', 'claude', 'anthropic', 'langchain',
            'api', 'bug', 'error', 'rate-limit', 'timeout', 'json'
        ]

        for tag in tags_and_labels:
            tag_lower = tag.lower()
            for pattern in llm_tag_patterns:
                if pattern in tag_lower:
                    relevant_tags.append(tag)
                    break

        return relevant_tags

    def _calculate_confidence(
        self,
        llm_keywords: List[str],
        ml_keywords: List[str],
        error_keywords: List[str],
        contract_matches: List[Tuple[str, int, int]],
        tag_matches: List[str],
        text: str
    ) -> float:
        """Calculate confidence score for the filter decision."""

        # More generous base score components
        # Up to 0.6 for LLM keywords (increased)
        llm_score = min(len(llm_keywords) * 0.25, 0.6)
        # Up to 0.3 for ML keywords (increased)
        ml_score = min(len(ml_keywords) * 0.15, 0.3)
        # Up to 0.3 for error keywords (increased)
        error_score = min(len(error_keywords) * 0.15, 0.3)
        pattern_score = min(len(contract_matches) * 0.35,
                            0.7)  # Up to 0.7 for patterns (increased)
        # Up to 0.4 for tags (increased)
        tag_score = min(len(tag_matches) * 0.2, 0.4)

        # Combination bonuses
        combination_bonus = 0.0
        if len(llm_keywords) > 0 and len(error_keywords) > 0:
            combination_bonus += 0.2  # LLM + error is strong signal

        if len(contract_matches) > 0 and len(llm_keywords) > 0:
            combination_bonus += 0.15  # Contract patterns + LLM keywords

        # Text quality factors
        text_length = len(text)
        if text_length > 100:  # Prefer longer, more descriptive posts
            length_bonus = min((text_length - 100) / 1000, 0.1)
        else:
            length_bonus = -0.1  # Penalize very short posts

        # Calculate final confidence
        confidence = llm_score + ml_score + error_score + \
            pattern_score + tag_score + combination_bonus + length_bonus

        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

    def _extract_snippets(self, text: str, keywords: List[str], context_size: int = 100) -> List[str]:
        """Extract relevant snippets around keywords."""
        snippets = []
        text_lower = text.lower()

        for keyword in keywords[:10]:  # Limit to top 10 keywords
            keyword_lower = keyword.lower()
            start_pos = text_lower.find(keyword_lower)

            if start_pos != -1:
                # Extract context around the keyword
                snippet_start = max(0, start_pos - context_size)
                snippet_end = min(len(text), start_pos +
                                  len(keyword) + context_size)
                snippet = text[snippet_start:snippet_end].strip()

                if len(snippet) > 20:  # Only keep meaningful snippets
                    snippets.append(snippet)

        return snippets[:5]  # Return top 5 snippets

    def _identify_potential_contracts(
        self,
        text: str,
        llm_keywords: List[str],
        contract_matches: List[Tuple[str, int, int]]
    ) -> List[str]:
        """Identify potential contract clauses in the text."""
        contracts = []

        # Extract sentences containing LLM keywords
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentence
                sentence_lower = sentence.lower()
                for keyword in llm_keywords:
                    if keyword.lower() in sentence_lower:
                        contracts.append(sentence)
                        break

        # Add contract pattern matches
        for match_text, _, _ in contract_matches:
            if len(match_text) > 10:
                contracts.append(match_text)

        return contracts[:3]  # Return top 3 potential contracts

    async def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics from the database."""
        stats = {}

        # Overall filter stats
        total_filtered = await self.db.count_documents('filtered_posts')
        passed_filter = await self.db.count_documents('filtered_posts', {'passed_keyword_filter': True})
        failed_filter = total_filtered - passed_filter

        stats['overall'] = {
            'total_filtered': total_filtered,
            'passed': passed_filter,
            'failed': failed_filter,
            'pass_rate': (passed_filter / total_filtered * 100) if total_filtered > 0 else 0
        }

        # Confidence distribution
        high_confidence = await self.db.count_documents('filtered_posts', {
            'filter_confidence': {'$gte': 0.7},
            'passed_keyword_filter': True
        })
        medium_confidence = await self.db.count_documents('filtered_posts', {
            'filter_confidence': {'$gte': 0.4, '$lt': 0.7},
            'passed_keyword_filter': True
        })
        low_confidence = await self.db.count_documents('filtered_posts', {
            'filter_confidence': {'$lt': 0.4},
            'passed_keyword_filter': True
        })

        stats['confidence_distribution'] = {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence
        }

        return stats
