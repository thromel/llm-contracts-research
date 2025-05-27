"""
Borderline LLM Screener using GPT-4.1.

Re-evaluates posts with uncertain confidence scores from bulk screening
for higher accuracy on edge cases.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import httpx

from ..common.models import FilteredPost, LLMScreeningResult
from ..common.database import MongoDBManager, ProvenanceTracker
from .prompts.borderline_screening_prompts import BorderlineScreeningPrompts

logger = logging.getLogger(__name__)


class BorderlineScreener:
    """
    Borderline case screener using GPT-4.1.

    Features:
    - High-accuracy processing of uncertain cases
    - Detailed analysis for borderline confidence scores
    - Enhanced prompting for edge case detection
    - Quality-focused evaluation
    """

    def __init__(
        self,
        api_key: str,
        db_manager: MongoDBManager,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4-1106-preview",
        rate_limit_delay: float = 2.0,
        max_concurrent_requests: int = 10
    ):
        """Initialize borderline screener.

        Args:
            api_key: OpenAI API key
            db_manager: MongoDB manager for storage
            base_url: OpenAI API base URL
            model: GPT-4 model to use for screening
            rate_limit_delay: Delay between API calls in seconds
            max_concurrent_requests: Maximum concurrent API requests
        """
        self.api_key = api_key
        self.db = db_manager
        self.base_url = base_url
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.max_concurrent_requests = max_concurrent_requests
        self.provenance = ProvenanceTracker(db_manager)

        # Semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # HTTP client configuration
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Use improved prompts from research-based prompt system
        self.prompts = BorderlineScreeningPrompts()

    async def screen_all_unscreened_posts(
        self,
        batch_size: int = 25
    ) -> Dict[str, Any]:
        """Screen all unscreened posts directly (when no bulk screener is available).

        Args:
            batch_size: Number of posts to process

        Returns:
            Batch processing statistics
        """
        stats = {
            'processed': 0,
            'positive_decisions': 0,
            'negative_decisions': 0,
            'borderline_cases': 0,
            'high_confidence': 0,
            'api_calls': 0,
            'processing_time': 0,
            'errors': 0
        }

        start_time = datetime.utcnow()

        # Get unscreened posts that passed keyword filter
        unscreened_posts = []
        async for filtered_post in self.db.find_many(
            'filtered_posts',
            {
                'passed_keyword_filter': True,
                'llm_screened': {'$ne': True}
            },
            limit=batch_size
        ):
            unscreened_posts.append(filtered_post)

        logger.info(
            f"Starting concurrent screening of {len(unscreened_posts)} unscreened posts (max {self.max_concurrent_requests} concurrent)")

        # Create concurrent tasks for all posts
        tasks = [self._process_single_post_with_stats(post, stats)
                 for post in unscreened_posts]

        # Process all posts concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Direct screening completed: {stats}")

        return stats

    async def screen_borderline_cases(
        self,
        confidence_min: float = 0.3,
        confidence_max: float = 0.7,
        batch_size: int = 25
    ) -> Dict[str, Any]:
        """Screen posts with borderline confidence scores.

        Args:
            confidence_min: Minimum confidence to be considered borderline
            confidence_max: Maximum confidence to be considered borderline
            batch_size: Number of posts to process

        Returns:
            Batch processing statistics
        """
        stats = {
            'processed': 0,
            'positive_decisions': 0,
            'negative_decisions': 0,
            'borderline_cases': 0,
            'high_confidence': 0,
            'api_calls': 0,
            'processing_time': 0,
            'errors': 0,
            'confidence_improvements': 0
        }

        start_time = datetime.utcnow()

        # Get posts that need borderline re-evaluation
        borderline_posts = []
        async for result in self.db.find_many(
            'llm_screening_results',
            {
                'confidence': {
                    '$gte': confidence_min,
                    '$lte': confidence_max
                },
                'borderline_reviewed': {'$ne': True}
            },
            limit=batch_size
        ):
            borderline_posts.append(result)

        logger.info(
            f"Starting concurrent borderline screening of {len(borderline_posts)} posts (max {self.max_concurrent_requests} concurrent)")

        # Create concurrent tasks for borderline posts
        tasks = [self._process_borderline_post_with_stats(result, stats)
                 for result in borderline_posts]

        # Process all posts concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Borderline screening completed: {stats}")

        return stats

    async def _screen_single_post(
        self,
        filtered_post: Dict[str, Any],
        previous_analysis: Dict[str, Any]
    ) -> LLMScreeningResult:
        """Screen a single post with detailed analysis.

        Args:
            filtered_post: Filtered post document
            previous_analysis: Previous screening result

        Returns:
            Enhanced LLMScreeningResult
        """
        # Get the original raw post content
        raw_post = await self.db.find_one('raw_posts', {'_id': filtered_post['raw_post_id']})

        if not raw_post:
            raise ValueError(
                f"Raw post not found for filtered post {filtered_post['_id']}")

        # Prepare content for screening
        title = raw_post.get('title', '')
        content = raw_post.get('body_md', '')

        # Include metadata context
        metadata_context = f"""
Platform: {raw_post.get('platform', 'unknown')}
Tags: {raw_post.get('tags', [])}
Score: {raw_post.get('score', 0)}
Matched Keywords: {filtered_post.get('matched_keywords', [])}
Filter Confidence: {filtered_post.get('filter_confidence', 0.0)}
"""

        # Format previous analysis
        previous_summary = f"""
Original Decision: {previous_analysis.get('decision', 'unknown')}
Original Confidence: {previous_analysis.get('confidence', 0.0)}
Original Rationale: {previous_analysis.get('rationale', 'none')}
"""

        # Create enhanced prompt using improved research-based prompts
        prompt = self.prompts.get_borderline_screening_prompt().format(
            title=title,
            content=content[:4000],  # Larger context for GPT-4
            previous_analysis=previous_summary + metadata_context
        )

        # Make API call
        response = await self._call_openai_api(prompt)

        # Parse response
        result = self._parse_screening_response(response)
        result.model_used = f"borderline_{self.model}"

        return result

    async def _screen_unscreened_post(
        self,
        filtered_post: Dict[str, Any]
    ) -> LLMScreeningResult:
        """Screen a post that hasn't been through bulk screening.

        Args:
            filtered_post: Filtered post document

        Returns:
            LLMScreeningResult
        """
        # Get the original raw post content
        raw_post = await self.db.find_one('raw_posts', {'_id': filtered_post['raw_post_id']})

        if not raw_post:
            raise ValueError(
                f"Raw post not found for filtered post {filtered_post['_id']}")

        # Prepare content for screening
        title = raw_post.get('title', '')
        content = raw_post.get('body_md', '')

        # Include metadata context
        metadata_context = f"""
Platform: {raw_post.get('platform', 'unknown')}
Tags: {raw_post.get('tags', [])}
Score: {raw_post.get('score', 0)}
Matched Keywords: {filtered_post.get('matched_keywords', [])}
Filter Confidence: {filtered_post.get('filter_confidence', 0.0)}
"""

        # Create prompt for comprehensive screening (primary screener mode)
        prompt = self.prompts.get_borderline_screening_prompt().format(
            title=title,
            # Larger context for comprehensive analysis
            content=content[:6000],
            previous_analysis="Primary screening analysis - comprehensive evaluation" + metadata_context
        )

        # Make API call
        response = await self._call_openai_api(prompt)

        # Parse response
        result = self._parse_screening_response(response)
        result.model_used = f"direct_{self.model}"

        return result

    async def _process_single_post_with_stats(
        self,
        filtered_post: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """Process a single post with concurrent rate limiting and stats tracking.

        Args:
            filtered_post: Filtered post document
            stats: Shared statistics dictionary (thread-safe due to asyncio)
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Rate limiting before API call
                await asyncio.sleep(self.rate_limit_delay)

                # Screen post directly without previous analysis
                screening_result = await self._screen_unscreened_post(filtered_post)

                # Update stats (safe in asyncio)
                stats['processed'] += 1
                stats['api_calls'] += 1

                # Classify result
                if screening_result.decision == 'Y':
                    stats['positive_decisions'] += 1
                    if screening_result.confidence >= 0.8:
                        stats['high_confidence'] += 1
                elif screening_result.decision == 'N':
                    stats['negative_decisions'] += 1
                else:
                    stats['borderline_cases'] += 1

                # Save the screening result as a new document
                await self._save_initial_screening_result(filtered_post, screening_result)

                # Mark filtered post as screened
                await self.db.update_one('filtered_posts',
                                         {'_id': filtered_post['_id']},
                                         {'$set': {'llm_screened': True}})

                logger.info(
                    f"✅ Screened post {filtered_post.get('_id', 'unknown')}: {screening_result.decision} (confidence: {screening_result.confidence:.3f})")

            except Exception as e:
                logger.error(
                    f"❌ Error screening post {filtered_post.get('_id', 'unknown')}: {str(e)}")
                stats['errors'] += 1

    async def _process_borderline_post_with_stats(
        self,
        screening_result: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """Process a single borderline post with concurrent rate limiting and stats tracking.

        Args:
            screening_result: Previous screening result document
            stats: Shared statistics dictionary
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Rate limiting before API call
                await asyncio.sleep(self.rate_limit_delay)

                # Get the original filtered post
                filtered_post = await self.db.find_one(
                    'filtered_posts',
                    {'_id': screening_result['filtered_post_id']}
                )

                if not filtered_post:
                    logger.warning(
                        f"Filtered post not found for screening result {screening_result['_id']}")
                    return

                # Re-screen with detailed analysis
                enhanced_result = await self._screen_single_post(
                    filtered_post,
                    previous_analysis=screening_result
                )

                # Update stats
                stats['processed'] += 1
                stats['api_calls'] += 1

                # Compare confidence improvement
                original_confidence = screening_result.get('confidence', 0.0)
                new_confidence = enhanced_result.confidence

                if abs(new_confidence - original_confidence) > 0.2:
                    stats['confidence_improvements'] += 1

                # Classify result
                if enhanced_result.decision == 'Y':
                    stats['positive_decisions'] += 1
                    if enhanced_result.confidence >= 0.8:
                        stats['high_confidence'] += 1
                elif enhanced_result.decision == 'N':
                    stats['negative_decisions'] += 1
                else:
                    stats['borderline_cases'] += 1

                # Update the screening result with borderline analysis
                await self._update_screening_result(screening_result['_id'], enhanced_result)

                logger.info(
                    f"🔄 Re-screened borderline post {screening_result['_id']}: {enhanced_result.decision} (confidence: {enhanced_result.confidence:.3f})")

            except Exception as e:
                logger.error(
                    f"❌ Error in borderline screening {screening_result.get('_id', 'unknown')}: {str(e)}")
                stats['errors'] += 1

    async def _save_initial_screening_result(
        self,
        filtered_post: Dict[str, Any],
        screening_result: LLMScreeningResult
    ) -> None:
        """Save initial screening result to database."""

        screening_doc = {
            'filtered_post_id': str(filtered_post['_id']),
            'decision': screening_result.decision,
            'confidence': screening_result.confidence,
            'rationale': screening_result.rationale,
            'model_used': screening_result.model_used,
            'screening_type': 'direct',
            'created_at': datetime.utcnow(),
            'borderline_reviewed': False
        }

        # Add classification data if available
        if screening_result.contract_violations:
            screening_doc['contract_violations'] = screening_result.contract_violations
        if screening_result.novel_patterns:
            screening_doc['novel_patterns'] = screening_result.novel_patterns
        if screening_result.research_value:
            screening_doc['research_value'] = screening_result.research_value
        if screening_result.verification_notes:
            screening_doc['verification_notes'] = screening_result.verification_notes

        await self.db.insert_one('llm_screening_results', screening_doc)

        # Log provenance
        await self.provenance.log_transformation(
            source_id=str(filtered_post['_id']),
            source_collection='filtered_posts',
            target_id=str(screening_doc.get('_id', 'unknown')),
            target_collection='llm_screening_results',
            transformation_type='direct_screening',
            metadata={'confidence': screening_result.confidence}
        )

    async def _call_openai_api(self, prompt: str) -> str:
        """Make API call to OpenAI with retry logic and exponential backoff.

        Args:
            prompt: The screening prompt

        Returns:
            API response text
        """
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    payload = {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert researcher analyzing LLM API contract violations with high accuracy and detailed reasoning."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    }

                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        logger.debug(
                            f"API response content (first 200 chars): {content[:200]}...")
                        return content
                    elif response.status_code == 429:  # Rate limit
                        # Extract retry-after from headers if available
                        retry_after = response.headers.get('retry-after')
                        if retry_after:
                            delay = float(retry_after)
                        else:
                            # Exponential backoff: 1, 2, 4, 8, 16 seconds
                            delay = base_delay * (2 ** attempt)

                        logger.warning(
                            f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response.raise_for_status()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

        raise Exception(
            f"Failed to get API response after {max_retries} attempts")

    def _parse_screening_response(self, response: str) -> LLMScreeningResult:
        """Parse the API response into structured result.

        Args:
            response: Raw API response text

        Returns:
            LLMScreeningResult object
        """
        # Default values for fallback
        decision = "Borderline"
        confidence = 0.5
        rationale = "Failed to parse response"
        contract_violations = None
        novel_patterns = None
        research_value = None
        verification_notes = None

        # Strategy 1: Try to parse simple text format first (preferred)
        lines = response.strip().split('\n')
        text_parsed = False

        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision = line.split(':', 1)[1].strip()
                text_parsed = True
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5
            elif line.startswith('RATIONALE:'):
                rationale = line.split(':', 1)[1].strip()

        # Strategy 2: If text parsing failed, try JSON format as fallback
        if not text_parsed:
            try:
                # Look for JSON in response
                json_start = response.find('{')
                if json_start != -1:
                    brace_count = 0
                    json_end = json_start
                    in_string = False
                    escape_next = False

                    for i, char in enumerate(response[json_start:], json_start):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break

                    if brace_count == 0 and json_end > json_start:
                        json_content = response[json_start:json_end]
                        import json
                        parsed = json.loads(json_content)

                        # Map from JSON format to our variables
                        contains_violation = parsed.get(
                            'contains_violation', False)
                        decision = 'Y' if contains_violation else 'N'

                        # Map confidence from text to float
                        confidence_str = parsed.get('confidence', 'medium')
                        if confidence_str == 'high':
                            confidence = 0.9
                        elif confidence_str == 'medium':
                            confidence = 0.7
                        elif confidence_str == 'low':
                            confidence = 0.4
                        else:
                            confidence = 0.5

                        rationale = parsed.get(
                            'verification_notes', 'JSON parsed response')
                        text_parsed = True

            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                logger.warning(
                    f"Failed to parse JSON response: {str(e)[:100]}...")

        # Strategy 3: If both failed, try to extract meaningful information from partial responses
        if not text_parsed:
            logger.debug(f"Raw response: {response[:300]}...")
            response_lower = response.lower()

            # Look for explicit decisions in the response
            if 'contains_violation": true' in response_lower or 'violation": true' in response_lower or '"true"' in response_lower:
                decision = 'Y'
                confidence = 0.7
                rationale = "Contract violation detected in response"
            elif 'contains_violation": false' in response_lower or 'violation": false' in response_lower or '"false"' in response_lower:
                decision = 'N'
                confidence = 0.7
                rationale = "No contract violation found"
            elif 'contains_violation' in response_lower:
                # If we see the field name but can't determine value, analyze content
                if any(word in response_lower for word in ['error', 'invalid', 'limit', 'exceeded', 'failed', 'denied']):
                    decision = 'Y'
                    confidence = 0.6
                    rationale = "Partial response with violation indicators"
                else:
                    decision = 'N'
                    confidence = 0.5
                    rationale = "Partial response, no clear violation indicators"
            elif any(word in response_lower for word in ['violation', 'contract', 'error', 'invalid', 'limit', 'exceeded']):
                decision = 'Y'
                confidence = 0.6
                rationale = "Contract violation indicators detected in response"
            elif any(word in response_lower for word in ['no violation', 'not a contract', 'general question', 'installation', 'how to']):
                decision = 'N'
                confidence = 0.6
                rationale = "No contract violation indicators found"
            else:
                # Final fallback - conservative decision
                decision = 'N'
                confidence = 0.3
                rationale = f"Unable to parse response format, defaulting to no violation. Response: {response[:100]}..."

        # Ensure decision is valid
        if decision not in ['Y', 'N', 'Borderline']:
            if 'yes' in decision.lower() or 'positive' in decision.lower():
                decision = 'Y'
            elif 'no' in decision.lower() or 'negative' in decision.lower():
                decision = 'N'
            else:
                decision = 'Borderline'

        return LLMScreeningResult(
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            model_used=f"borderline_{self.model}",
            contract_violations=contract_violations,
            novel_patterns=novel_patterns,
            research_value=research_value,
            verification_notes=verification_notes
        )

    async def _update_screening_result(
        self,
        result_id: str,
        enhanced_result: LLMScreeningResult
    ) -> None:
        """Update existing screening result with borderline analysis.

        Args:
            result_id: ID of the screening result to update
            enhanced_result: Enhanced analysis result
        """
        update_doc = {
            'borderline_decision': enhanced_result.decision,
            'borderline_confidence': enhanced_result.confidence,
            'borderline_rationale': enhanced_result.rationale,
            'borderline_model': enhanced_result.model_used,
            'borderline_reviewed': True,
            'borderline_timestamp': datetime.utcnow(),
            'final_decision': enhanced_result.decision,  # Override with borderline result
            'final_confidence': enhanced_result.confidence
        }

        # Add classification data if available
        if enhanced_result.contract_violations:
            update_doc['contract_violations'] = enhanced_result.contract_violations
        if enhanced_result.novel_patterns:
            update_doc['novel_patterns'] = enhanced_result.novel_patterns
        if enhanced_result.research_value:
            update_doc['research_value'] = enhanced_result.research_value
        if enhanced_result.verification_notes:
            update_doc['verification_notes'] = enhanced_result.verification_notes

        await self.db.update_one(
            'llm_screening_results',
            {'_id': result_id},
            {'$set': update_doc}
        )

        # Log provenance
        await self.provenance.log_transformation(
            source_id=result_id,
            source_collection='llm_screening_results',
            target_id=result_id,
            target_collection='llm_screening_results',
            transformation_type='borderline_screening',
            metadata={'enhanced_confidence': enhanced_result.confidence}
        )

    async def get_borderline_statistics(self) -> Dict[str, Any]:
        """Get statistics about borderline screening performance.

        Returns:
            Dictionary with borderline screening statistics
        """
        pipeline = [
            {'$match': {'borderline_reviewed': True}},
            {'$group': {
                '_id': None,
                'total_reviewed': {'$sum': 1},
                'confidence_improved': {
                    '$sum': {
                        '$cond': [
                            {'$gt': [
                                {'$abs': {'$subtract': [
                                    '$borderline_confidence', '$confidence']}},
                                0.2
                            ]},
                            1,
                            0
                        ]
                    }
                },
                'avg_original_confidence': {'$avg': '$confidence'},
                'avg_borderline_confidence': {'$avg': '$borderline_confidence'},
                'decision_changes': {
                    '$sum': {
                        '$cond': [
                            {'$ne': ['$decision', '$borderline_decision']},
                            1,
                            0
                        ]
                    }
                }
            }}
        ]

        result = await self.db.aggregate('llm_screening_results', pipeline)
        results = await result.to_list(1)

        if results:
            return results[0]
        else:
            return {
                'total_reviewed': 0,
                'confidence_improved': 0,
                'avg_original_confidence': 0.0,
                'avg_borderline_confidence': 0.0,
                'decision_changes': 0
            }

    async def validate_api_connection(self) -> bool:
        """Validate OpenAI API connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_prompt = "Respond with 'API connection successful'"
            response = await self._call_openai_api(test_prompt)
            return "successful" in response.lower()
        except Exception as e:
            logger.error(f"API validation failed: {str(e)}")
            return False
