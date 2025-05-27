"""
Borderline LLM Screener using GPT-4.1.

Re-evaluates posts with uncertain confidence scores from bulk screening
for higher accuracy on edge cases.
"""

import asyncio
import logging
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
        model: str = "gpt-4-1106-preview"
    ):
        """Initialize borderline screener.

        Args:
            api_key: OpenAI API key
            db_manager: MongoDB manager for storage
            base_url: OpenAI API base URL
            model: GPT-4 model to use for screening
        """
        self.api_key = api_key
        self.db = db_manager
        self.base_url = base_url
        self.model = model
        self.provenance = ProvenanceTracker(db_manager)

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
            f"Starting direct screening of {len(unscreened_posts)} unscreened posts")

        # Process posts individually
        for filtered_post in unscreened_posts:
            try:
                # Screen post directly without previous analysis
                screening_result = await self._screen_unscreened_post(filtered_post)

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

                # Rate limiting
                await asyncio.sleep(2.0)  # Conservative for GPT-4

            except Exception as e:
                logger.error(
                    f"Error screening post {filtered_post.get('_id')}: {str(e)}")
                stats['errors'] += 1

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
            f"Starting borderline screening of {len(borderline_posts)} posts")

        # Process posts individually (higher quality, slower processing)
        for screening_result in borderline_posts:
            try:
                # Get the original filtered post
                filtered_post = await self.db.find_one(
                    'filtered_posts',
                    {'_id': screening_result['filtered_post_id']}
                )

                if not filtered_post:
                    logger.warning(
                        f"Filtered post not found for screening result {screening_result['_id']}")
                    continue

                # Re-screen with detailed analysis
                enhanced_result = await self._screen_single_post(
                    filtered_post,
                    previous_analysis=screening_result
                )

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

                # Rate limiting
                await asyncio.sleep(2.0)  # More conservative for GPT-4

            except Exception as e:
                logger.error(f"Error in borderline screening: {str(e)}")
                stats['errors'] += 1

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
        """Make API call to OpenAI.

        Args:
            prompt: The screening prompt

        Returns:
            API response text
        """
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

            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

    def _parse_screening_response(self, response: str) -> LLMScreeningResult:
        """Parse the API response into structured result.

        Args:
            response: Raw API response text

        Returns:
            LLMScreeningResult object
        """
        lines = response.strip().split('\n')

        # Default values
        decision = "Borderline"
        confidence = 0.5
        rationale = "Failed to parse response"

        # Parse response format
        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence)
                                     )  # Clamp to valid range
                except (ValueError, IndexError):
                    confidence = 0.5
            elif line.startswith('RATIONALE:'):
                rationale = line.split(':', 1)[1].strip()

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
            model_used=f"borderline_{self.model}"
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
