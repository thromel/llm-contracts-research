"""
Bulk LLM Screener using DeepSeek-R1.

Efficiently processes large volumes of filtered posts to identify
explicit or implicit API usage contract violations.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import httpx
from tqdm import tqdm

from ..common.models import FilteredPost, LLMScreeningResult
from ..common.database import MongoDBManager, ProvenanceTracker

logger = logging.getLogger(__name__)


class BulkScreener:
    """
    Bulk LLM screening using DeepSeek-R1.

    Features:
    - High-throughput processing
    - Cost-effective screening
    - Confidence-based filtering
    - Batch processing support
    """

    def __init__(
        self,
        api_key: str,
        db_manager: MongoDBManager,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-reasoner"
    ):
        """Initialize bulk screener.

        Args:
            api_key: DeepSeek API key
            db_manager: MongoDB manager for storage
            base_url: DeepSeek API base URL
            model: Model to use for screening
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

        # Screening prompt template
        self.screening_prompt = """You are analyzing a post from GitHub or Stack Overflow to determine if it contains explicit or implicit API usage contract violations related to Large Language Models (LLMs).

API contracts include:
- Parameter constraints (max_tokens, temperature, top_p ranges)
- Rate limiting and quota requirements  
- Input format specifications (JSON schema, function calling)
- Context length limitations
- Authentication and authorization requirements
- Error handling expectations
- Response format contracts

Your task: Does this post contain an explicit or implicit API-usage contract violation or discussion?

Post Title: {title}

Post Content: {content}

Respond with exactly this format:
DECISION: [Y/N]
CONFIDENCE: [0.0-1.0]
RATIONALE: [One sentence explanation]

Examples of Y (contract-related):
- "max_tokens must be between 1 and 4096 but I'm getting an error"
- "Rate limit exceeded with 429 status code" 
- "JSON schema validation failed for function calling"
- "Context length exceeded the model's limit"

Examples of N (not contract-related):
- General programming questions unrelated to LLM APIs
- Installation or environment setup issues
- Conceptual discussions without specific API constraints
- Generic error messages without API context"""

    async def screen_batch(
        self,
        batch_size: int = 100,
        confidence_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """Screen a batch of filtered posts.

        Args:
            batch_size: Number of posts to process
            confidence_threshold: Minimum confidence for positive classification

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

        # Get posts that passed keyword filter but haven't been LLM screened
        posts_to_screen = []
        async for filtered_post in self.db.get_posts_for_labelling("bulk_screening", batch_size):
            posts_to_screen.append(filtered_post)

        logger.info(f"Starting bulk screening of {len(posts_to_screen)} posts")

        # Process in smaller concurrent batches to respect rate limits
        concurrent_batch_size = 10
        for i in range(0, len(posts_to_screen), concurrent_batch_size):
            batch = posts_to_screen[i:i + concurrent_batch_size]

            # Process batch concurrently
            tasks = [self._screen_single_post(post) for post in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for post, result in zip(batch, results):
                try:
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error screening post {post.get('_id')}: {str(result)}")
                        stats['errors'] += 1
                        continue

                    screening_result = result
                    stats['processed'] += 1
                    stats['api_calls'] += 1

                    # Classify result
                    if screening_result.decision == 'Y':
                        stats['positive_decisions'] += 1
                        if screening_result.confidence >= 0.7:
                            stats['high_confidence'] += 1
                    elif screening_result.decision == 'N':
                        stats['negative_decisions'] += 1
                    else:
                        stats['borderline_cases'] += 1

                    # Save screening result (you'll implement this in the orchestrator)
                    await self._save_screening_result(post, screening_result)

                except Exception as e:
                    logger.error(
                        f"Error processing screening result for post {post.get('_id')}: {str(e)}")
                    stats['errors'] += 1

            # Rate limiting - wait between batches
            await asyncio.sleep(1.0)

        stats['processing_time'] = (
            datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Bulk screening completed: {stats}")

        return stats

    async def _screen_single_post(self, filtered_post: Dict[str, Any]) -> LLMScreeningResult:
        """Screen a single filtered post.

        Args:
            filtered_post: Filtered post document from MongoDB

        Returns:
            LLMScreeningResult with decision and confidence
        """
        # Get the original raw post content
        raw_post = await self.db.find_one('raw_posts', {'_id': filtered_post['raw_post_id']})

        if not raw_post:
            raise ValueError(
                f"Raw post not found for filtered post {filtered_post['_id']}")

        # Prepare content for screening
        title = raw_post.get('title', '')
        content = raw_post.get('body_md', '')

        # Truncate content if too long (to stay within context limits)
        if len(content) > 3000:
            content = content[:3000] + "..."

        # Format prompt
        prompt = self.screening_prompt.format(
            title=title,
            content=content
        )

        # Make API call
        response = await self._call_deepseek_api(prompt)

        # Parse response
        return self._parse_screening_response(response)

    async def _call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API for screening.

        Args:
            prompt: Formatted prompt for screening

        Returns:
            Raw response text from the API
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Low temperature for consistent decisions
                "max_tokens": 150,   # Short response expected
                "stream": False
            }

            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                if response.status_code == 429:
                    # Rate limit hit, wait and retry
                    logger.warning("Rate limit hit, waiting 10 seconds")
                    await asyncio.sleep(10)
                    return await self._call_deepseek_api(prompt)

                response.raise_for_status()

                data = response.json()
                return data['choices'][0]['message']['content']

            except httpx.HTTPError as e:
                logger.error(f"HTTP error calling DeepSeek API: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error calling DeepSeek API: {str(e)}")
                raise

    def _parse_screening_response(self, response: str) -> LLMScreeningResult:
        """Parse the LLM response into structured result.

        Args:
            response: Raw response from DeepSeek API

        Returns:
            Parsed LLMScreeningResult
        """
        try:
            # Parse structured response
            lines = response.strip().split('\n')
            decision = None
            confidence = 0.0
            rationale = ""

            for line in lines:
                line = line.strip()
                if line.startswith('DECISION:'):
                    decision = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = line.split(':', 1)[1].strip()
                    confidence = float(confidence_str)
                elif line.startswith('RATIONALE:'):
                    rationale = line.split(':', 1)[1].strip()

            # Validate decision
            if decision not in ['Y', 'N']:
                # Fallback parsing for non-standard responses
                response_lower = response.lower()
                if 'decision: y' in response_lower or 'yes' in response_lower:
                    decision = 'Y'
                elif 'decision: n' in response_lower or 'no' in response_lower:
                    decision = 'N'
                else:
                    decision = 'Unsure'
                    confidence = 0.5

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return LLMScreeningResult(
                decision=decision,
                rationale=rationale,
                confidence=confidence,
                model_used=self.model
            )

        except Exception as e:
            logger.error(
                f"Error parsing screening response: {str(e)}\nResponse: {response}")

            # Return fallback result
            return LLMScreeningResult(
                decision='Unsure',
                rationale=f"Parse error: {str(e)}",
                confidence=0.5,
                model_used=self.model
            )

    async def _save_screening_result(
        self,
        filtered_post: Dict[str, Any],
        screening_result: LLMScreeningResult
    ) -> None:
        """Save screening result to database.

        This is a placeholder - the actual saving will be handled
        by the screening orchestrator to avoid circular dependencies.
        """
        # This method will be called by the orchestrator
        pass

    async def get_screening_statistics(self) -> Dict[str, Any]:
        """Get bulk screening statistics."""
        # This would query the labelled_posts collection for bulk screening results
        # Implementation depends on how results are stored by the orchestrator
        stats = {
            'total_screened': 0,
            'positive_rate': 0.0,
            'average_confidence': 0.0,
            'borderline_rate': 0.0
        }

        return stats

    async def validate_api_connection(self) -> bool:
        """Test API connection and authentication."""
        try:
            test_prompt = "Test message. Respond with 'OK'."
            response = await self._call_deepseek_api(test_prompt)
            return 'ok' in response.lower()
        except Exception as e:
            logger.error(f"API validation failed: {str(e)}")
            return False
