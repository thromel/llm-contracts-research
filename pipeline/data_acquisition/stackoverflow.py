"""
Stack Overflow Data Acquisition for LLM Contracts Research Pipeline.

Leverages Stack Overflow's peer-review signals and accepted-answer metadata
to identify LLM API contract violations and usage patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, AsyncGenerator, Dict, Any
import httpx
from tqdm import tqdm
import gzip
import json

from ..common.models import RawPost, Platform
from ..common.database import MongoDBManager

logger = logging.getLogger(__name__)


class StackOverflowAcquisition:
    """
    Stack Overflow acquisition using Stack Exchange API v2.3.

    Features:
    - Tag-based filtering for LLM-related content
    - Accepted answer metadata extraction
    - Peer-review signal capture (votes, views, etc.)
    - Back-pagination for historical data
    """

    def __init__(self, db_manager: MongoDBManager, api_key: Optional[str] = None):
        """Initialize Stack Overflow acquisition.

        Args:
            db_manager: MongoDB manager for storage
            api_key: Optional Stack Exchange API key for higher rate limits
        """
        self.db = db_manager
        self.api_key = api_key
        self.base_url = "https://api.stackexchange.com/2.3"
        self.site = "stackoverflow"

        # LLM-related tags to search for
        self.llm_tags = [
            # Core LLM APIs
            'openai-api', 'gpt-3', 'gpt-4', 'chatgpt', 'claude',
            'anthropic', 'palm-api', 'bard-api',

            # Libraries and frameworks
            'langchain', 'transformers', 'huggingface', 'openai-python',
            'semantic-kernel', 'llama-index',

            # Technical concepts
            'json-schema', 'function-calling', 'prompt-engineering',
            'token-limit', 'rate-limiting', 'context-window',

            # Error patterns
            'api-rate-limit', 'timeout', 'authentication-error',
            'json-parsing', 'schema-validation'
        ]

        # Request parameters
        self.default_params = {
            'site': self.site,
            'pagesize': 100,
            'order': 'desc',
            'sort': 'activity',
            'filter': '!nNPvSNdWme'  # Custom filter for full question/answer content
        }

        if self.api_key:
            self.default_params['key'] = self.api_key

    async def acquire_tagged_questions(
        self,
        tags: Optional[List[str]] = None,
        since_days: int = 30,
        max_questions: int = 10000,
        include_answers: bool = True
    ) -> AsyncGenerator[RawPost, None]:
        """
        Acquire questions from Stack Overflow based on tags.

        Args:
            tags: List of tags to search for (defaults to self.llm_tags)
            since_days: Only fetch questions from last N days
            max_questions: Maximum questions to fetch
            include_answers: Whether to also yield answer posts

        Yields:
            RawPost objects for each question/answer
        """
        if tags is None:
            tags = self.llm_tags

        since_timestamp = int(
            (datetime.utcnow() - timedelta(days=since_days)).timestamp())

        with tqdm(total=len(tags), desc="Tags") as tag_pbar:
            for tag in tags:
                try:
                    question_count = 0

                    async for post in self._fetch_questions_by_tag(
                        tag, since_timestamp, max_questions // len(tags)
                    ):
                        yield post
                        question_count += 1

                        # If this is a question and we want answers, fetch them
                        if include_answers and hasattr(post, 'source_id'):
                            async for answer in self._fetch_question_answers(post.source_id):
                                yield answer

                    logger.info(
                        f"Fetched {question_count} questions for tag: {tag}")
                    tag_pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing tag {tag}: {str(e)}")
                    tag_pbar.update(1)
                    continue

    async def _fetch_questions_by_tag(
        self,
        tag: str,
        since_timestamp: int,
        max_questions: int
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch questions for a specific tag."""

        async with httpx.AsyncClient(timeout=30) as client:
            page = 1
            questions_fetched = 0
            has_more = True

            while has_more and questions_fetched < max_questions:
                try:
                    params = {
                        **self.default_params,
                        'tagged': tag,
                        'fromdate': since_timestamp,
                        'page': page
                    }

                    url = f"{self.base_url}/questions"
                    response = await client.get(url, params=params)

                    if response.status_code == 400:
                        # Quota exceeded or other error
                        data = response.json()
                        if 'quota_remaining' in data and data['quota_remaining'] == 0:
                            logger.warning(
                                "Stack Exchange API quota exhausted")
                            break
                        else:
                            logger.error(
                                f"API error: {data.get('error_message', 'Unknown error')}")
                            break

                    response.raise_for_status()

                    # Handle gzipped response
                    if response.headers.get('content-encoding') == 'gzip':
                        content = gzip.decompress(response.content)
                        data = json.loads(content.decode('utf-8'))
                    else:
                        data = response.json()

                    questions = data.get('items', [])
                    has_more = data.get('has_more', False)

                    if not questions:
                        break

                    for question in questions:
                        # Filter for LLM relevance
                        if not self._is_llm_relevant_question(question):
                            continue

                        # Convert to RawPost
                        raw_post = self._convert_question_to_rawpost(
                            question, tag)
                        yield raw_post

                        questions_fetched += 1
                        if questions_fetched >= max_questions:
                            break

                    page += 1

                    # Respect rate limits
                    await asyncio.sleep(0.1)

                except httpx.HTTPError as e:
                    logger.error(
                        f"HTTP error fetching questions for tag {tag}: {str(e)}")
                    break
                except Exception as e:
                    logger.error(
                        f"Error fetching questions for tag {tag}: {str(e)}")
                    break

    async def _fetch_question_answers(
        self,
        question_id: str
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch answers for a specific question."""

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                params = {
                    **self.default_params,
                    'order': 'desc',
                    'sort': 'votes'
                }

                url = f"{self.base_url}/questions/{question_id}/answers"
                response = await client.get(url, params=params)
                response.raise_for_status()

                # Handle gzipped response
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(response.content)
                    data = json.loads(content.decode('utf-8'))
                else:
                    data = response.json()

                answers = data.get('items', [])

                for answer in answers:
                    # Filter for LLM relevance
                    if not self._is_llm_relevant_answer(answer):
                        continue

                    # Convert to RawPost
                    raw_post = self._convert_answer_to_rawpost(
                        answer, question_id)
                    yield raw_post

            except Exception as e:
                logger.error(
                    f"Error fetching answers for question {question_id}: {str(e)}")

    def _is_llm_relevant_question(self, question: Dict[str, Any]) -> bool:
        """Check if question is relevant to LLM contracts research."""

        # Check title and body for LLM keywords
        text_content = f"{question.get('title', '')} {question.get('body', '')}"
        text_lower = text_content.lower()

        # LLM-specific keywords
        llm_keywords = [
            'max_tokens', 'temperature', 'top_p', 'openai', 'gpt',
            'claude', 'anthropic', 'langchain', 'api_key', 'rate_limit',
            'context_length', 'token_limit', 'json_schema', 'function_calling',
            'prompt', 'completion', 'embeddings', 'fine_tuning'
        ]

        for keyword in llm_keywords:
            if keyword in text_lower:
                return True

        # Check tags
        tags = [tag.lower() for tag in question.get('tags', [])]
        for tag in tags:
            if any(llm_tag in tag for llm_tag in ['openai', 'gpt', 'llm', 'langchain', 'claude']):
                return True

        # Check for error-related patterns
        error_patterns = [
            'error', 'exception', 'failed', 'not working', 'issue',
            'problem', 'trouble', 'help', 'fix'
        ]

        for pattern in error_patterns:
            if pattern in text_lower and any(keyword in text_lower for keyword in llm_keywords[:5]):
                return True

        return False

    def _is_llm_relevant_answer(self, answer: Dict[str, Any]) -> bool:
        """Check if answer is relevant to LLM contracts research."""

        # Check answer body for LLM keywords
        body = answer.get('body', '').lower()

        llm_keywords = [
            'max_tokens', 'temperature', 'top_p', 'openai', 'gpt',
            'claude', 'anthropic', 'api_key', 'rate_limit', 'context_length',
            'json_schema', 'function_calling', 'completion'
        ]

        for keyword in llm_keywords:
            if keyword in body:
                return True

        # Prioritize accepted answers
        if answer.get('is_accepted', False):
            return True

        # High-scored answers are more likely to be useful
        if answer.get('score', 0) >= 3:
            return True

        return False

    def _convert_question_to_rawpost(
        self,
        question: Dict[str, Any],
        original_tag: str
    ) -> RawPost:
        """Convert Stack Overflow question to RawPost format."""

        return RawPost(
            platform=Platform.STACKOVERFLOW,
            source_id=str(question['question_id']),
            url=question.get('link', ''),
            title=question.get('title', ''),
            body_md=question.get('body', ''),
            created_at=datetime.fromtimestamp(
                question.get('creation_date', 0)),
            updated_at=datetime.fromtimestamp(
                question.get('last_activity_date', 0)),
            score=question.get('score', 0),
            tags=question.get('tags', []) + [original_tag],
            author=question.get('owner', {}).get('display_name', 'unknown'),
            state='closed' if question.get('closed_date') else 'open',
            comments_count=question.get('comment_count', 0),
            view_count=question.get('view_count', 0),
            accepted_answer_id=str(question.get('accepted_answer_id')) if question.get(
                'accepted_answer_id') else None,
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

    def _convert_answer_to_rawpost(
        self,
        answer: Dict[str, Any],
        question_id: str
    ) -> RawPost:
        """Convert Stack Overflow answer to RawPost format."""

        return RawPost(
            platform=Platform.STACKOVERFLOW,
            source_id=str(answer['answer_id']),
            url=answer.get('link', ''),
            title=f"Answer to question {question_id}",
            body_md=answer.get('body', ''),
            created_at=datetime.fromtimestamp(answer.get('creation_date', 0)),
            updated_at=datetime.fromtimestamp(
                answer.get('last_activity_date', 0)),
            score=answer.get('score', 0),
            answer_score=answer.get('score', 0),
            tags=[f"answer-to-{question_id}"],
            author=answer.get('owner', {}).get('display_name', 'unknown'),
            state='accepted' if answer.get('is_accepted', False) else 'open',
            comments_count=answer.get('comment_count', 0),
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

    async def save_to_database(self, raw_post: RawPost) -> str:
        """Save RawPost to MongoDB with deduplication."""

        # Check for existing post
        existing = await self.db.find_one(
            'raw_posts',
            {
                'platform': raw_post.platform.value,
                'source_id': raw_post.source_id
            }
        )

        if existing:
            # Update if newer
            if raw_post.updated_at > existing.get('updated_at', datetime.min):
                await self.db.update_one(
                    'raw_posts',
                    {'_id': existing['_id']},
                    {'$set': raw_post.to_dict()}
                )
                return str(existing['_id'])
            else:
                return str(existing['_id'])
        else:
            # Insert new
            result = await self.db.insert_one('raw_posts', raw_post.to_dict())
            return str(result.inserted_id)

    async def get_tag_statistics(self, tags: List[str] = None) -> Dict[str, Any]:
        """Get statistics about tag usage and question counts."""

        if tags is None:
            tags = self.llm_tags

        tag_stats = {}

        async with httpx.AsyncClient(timeout=30) as client:
            for tag in tags:
                try:
                    params = {
                        **self.default_params,
                        'inname': tag,
                        'pagesize': 1
                    }

                    url = f"{self.base_url}/tags"
                    response = await client.get(url, params=params)
                    response.raise_for_status()

                    data = response.json()
                    tag_info = data.get('items', [])

                    if tag_info:
                        tag_data = tag_info[0]
                        tag_stats[tag] = {
                            'name': tag_data.get('name', tag),
                            'count': tag_data.get('count', 0),
                            'has_synonyms': tag_data.get('has_synonyms', False),
                            'is_moderator_only': tag_data.get('is_moderator_only', False),
                            'is_required': tag_data.get('is_required', False)
                        }
                    else:
                        tag_stats[tag] = {
                            'name': tag,
                            'count': 0,
                            'exists': False
                        }

                    # Respect rate limits
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(
                        f"Error getting stats for tag {tag}: {str(e)}")
                    tag_stats[tag] = {'name': tag, 'error': str(e)}

        return tag_stats

    async def search_questions_by_text(
        self,
        search_terms: List[str],
        since_days: int = 30,
        max_results: int = 1000
    ) -> AsyncGenerator[RawPost, None]:
        """Search questions using text search rather than tags."""

        since_timestamp = int(
            (datetime.utcnow() - timedelta(days=since_days)).timestamp())

        async with httpx.AsyncClient(timeout=30) as client:
            for search_term in search_terms:
                try:
                    page = 1
                    results_fetched = 0
                    has_more = True

                    while has_more and results_fetched < max_results:
                        params = {
                            **self.default_params,
                            'intitle': search_term,
                            'fromdate': since_timestamp,
                            'page': page
                        }

                        url = f"{self.base_url}/search"
                        response = await client.get(url, params=params)
                        response.raise_for_status()

                        data = response.json()
                        questions = data.get('items', [])
                        has_more = data.get('has_more', False)

                        if not questions:
                            break

                        for question in questions:
                            if self._is_llm_relevant_question(question):
                                raw_post = self._convert_question_to_rawpost(
                                    question, f"search:{search_term}"
                                )
                                yield raw_post
                                results_fetched += 1

                        page += 1
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(
                        f"Error searching for '{search_term}': {str(e)}")
                    continue
