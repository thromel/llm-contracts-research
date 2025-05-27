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

    def __init__(self, db_manager: MongoDBManager, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Stack Overflow acquisition.

        Args:
            db_manager: MongoDB manager for storage
            api_key: Optional Stack Exchange API key for higher rate limits
            config: Optional configuration dict with filtering criteria
        """
        self.db = db_manager
        self.api_key = api_key
        self.base_url = "https://api.stackexchange.com/2.3"
        self.site = "stackoverflow"

        # Load filtering configuration
        self.config = config or {}
        so_filtering = self.config.get('sources', {}).get(
            'stackoverflow', {}).get('filtering', {})

        # Configurable filtering criteria
        self.min_score = so_filtering.get('min_score', 1)
        self.require_answered = so_filtering.get('require_answered', True)
        self.require_accepted_answer = so_filtering.get(
            'require_accepted_answer', False)
        self.min_accepted_answer_score = so_filtering.get(
            'min_accepted_answer_score', 0)
        self.check_duplicates = so_filtering.get('check_duplicates', True)

        # Focus on highest-signal tags only
        self.llm_tags = [
            'openai-api',
            'langchain',
            'gpt-4',
            'chatgpt-api'
        ]

        # No exclusion patterns - minimal filtering only

        # Request parameters
        self.default_params = {
            'site': self.site,
            'pagesize': 100,
            'order': 'desc',
            'sort': 'activity',
            'filter': '!9_bDDxJY5'  # Include body content for questions/answers
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
                        'page': page,
                        'min': self.min_score
                    }

                    # Log filtering configuration for debugging
                    if page == 1:  # Only log on first page to avoid spam
                        logger.info(
                            f"Stack Overflow filtering for tag '{tag}': min_score={self.min_score}, require_answered={self.require_answered}, require_accepted_answer={self.require_accepted_answer}, check_duplicates={self.check_duplicates}")

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

                    # Handle response - Stack Exchange API can return gzipped or regular JSON
                    try:
                        if response.headers.get('content-encoding') == 'gzip':
                            content = gzip.decompress(response.content)
                            data = json.loads(content.decode('utf-8'))
                        else:
                            data = response.json()
                    except (gzip.BadGzipFile, json.JSONDecodeError) as e:
                        # Fallback: try to parse as regular JSON
                        try:
                            data = response.json()
                        except json.JSONDecodeError:
                            logger.error(
                                f"Could not parse response for tag {tag}: {str(e)}")
                            break

                    questions = data.get('items', [])
                    has_more = data.get('has_more', False)

                    if not questions:
                        break

                    for question in questions:
                        # Check if we already have this post (if enabled)
                        if self.check_duplicates:
                            question_id = str(question['question_id'])
                            existing = await self.db.find_one(
                                'raw_posts',
                                {
                                    'platform': 'stackoverflow',
                                    'source_id': question_id
                                }
                            )
                            if existing:
                                continue

                        # Stage 1: Check if question must be answered
                        if self.require_answered and not question.get('is_answered', False):
                            continue

                        # Stage 2: Check if accepted answer is required
                        if self.require_accepted_answer and not question.get('accepted_answer_id'):
                            continue

                        # Stage 3: Validate accepted answer score if required
                        if (self.require_accepted_answer and
                            question.get('accepted_answer_id') and
                                self.min_accepted_answer_score > 0):
                            accepted_answer_score = await self._get_accepted_answer_score(
                                str(question['question_id']),
                                str(question['accepted_answer_id']),
                                client
                            )
                            if accepted_answer_score < self.min_accepted_answer_score:
                                continue

                        # Convert to RawPost (with comments)
                        raw_post = await self._convert_question_to_rawpost_with_comments(
                            question, tag, client)
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

                # Handle response - Stack Exchange API can return gzipped or regular JSON
                try:
                    if response.headers.get('content-encoding') == 'gzip':
                        content = gzip.decompress(response.content)
                        data = json.loads(content.decode('utf-8'))
                    else:
                        data = response.json()
                except (gzip.BadGzipFile, json.JSONDecodeError) as e:
                    # Fallback: try to parse as regular JSON
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        logger.error(
                            f"Could not parse response for question {question_id}: {str(e)}")
                        return

                answers = data.get('items', [])

                for answer in answers:
                    # Include all answers without filtering
                    raw_post = self._convert_answer_to_rawpost(
                        answer, question_id)
                    yield raw_post

            except Exception as e:
                logger.error(
                    f"Error fetching answers for question {question_id}: {str(e)}")

    async def _fetch_question_comments(
        self,
        question_id: str,
        client: httpx.AsyncClient
    ) -> str:
        """Fetch comments for a question."""

        try:
            params = {
                **self.default_params,
                'order': 'desc',
                'sort': 'creation'
            }

            url = f"{self.base_url}/questions/{question_id}/comments"
            response = await client.get(url, params=params)
            response.raise_for_status()

            # Handle response - Stack Exchange API can return gzipped or regular JSON
            try:
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(response.content)
                    data = json.loads(content.decode('utf-8'))
                else:
                    data = response.json()
            except (gzip.BadGzipFile, json.JSONDecodeError) as e:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse comments response for question {question_id}: {str(e)}")
                    return ""

            comments = data.get('items', [])
            comments_text = []

            for comment in comments:
                author = comment.get('owner', {}).get(
                    'display_name', 'unknown')
                body = comment.get('body', '').strip()
                score = comment.get('score', 0)
                creation_date = comment.get('creation_date', 0)

                if body:
                    comments_text.append(
                        f"Comment by {author} (score: {score}):\n{body}")

            return '\n\n'.join(comments_text)

        except Exception as e:
            logger.error(
                f"Error fetching comments for question {question_id}: {str(e)}")
            return ""

    async def _fetch_question_answers_text(
        self,
        question_id: str,
        client: httpx.AsyncClient,
        accepted_answer_id: Optional[str] = None
    ) -> str:
        """Fetch answers for a question as formatted text with accepted answer first."""

        try:
            # Use specific filter for answers to ensure body content is included
            params = {
                'site': self.site,
                'pagesize': 100,
                'order': 'desc',
                'sort': 'votes',
                'filter': '!nNPvSNdWme'  # Filter specifically for answers with body content
            }

            if self.api_key:
                params['key'] = self.api_key

            url = f"{self.base_url}/questions/{question_id}/answers"
            response = await client.get(url, params=params)
            response.raise_for_status()

            # Handle response - Stack Exchange API can return gzipped or regular JSON
            try:
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(response.content)
                    data = json.loads(content.decode('utf-8'))
                else:
                    data = response.json()
            except (gzip.BadGzipFile, json.JSONDecodeError) as e:
                # Fallback: try to parse as regular JSON
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse answers response for question {question_id}: {str(e)}")
                    return ""

            answers = data.get('items', [])
            answers_text = []
            accepted_answers = []
            other_answers = []

            # Separate accepted and other answers
            for answer in answers:
                author = answer.get('owner', {}).get('display_name', 'unknown')
                body = answer.get('body', '').strip()
                score = answer.get('score', 0)
                is_accepted = answer.get('is_accepted', False)
                answer_id = str(answer.get('answer_id', ''))

                if body:
                    if is_accepted or (accepted_answer_id and answer_id == str(accepted_answer_id)):
                        accepted_mark = " ✓ ACCEPTED"
                        formatted_answer = f"Answer by {author} (score: {score}){accepted_mark}:\n{body}"
                        accepted_answers.append(formatted_answer)
                    else:
                        formatted_answer = f"Answer by {author} (score: {score}):\n{body}"
                        other_answers.append(formatted_answer)

            # Put accepted answers first, then other answers sorted by score
            all_answers = accepted_answers + other_answers
            return '\n\n'.join(all_answers)

        except Exception as e:
            logger.error(
                f"Error fetching answers for question {question_id}: {str(e)}")
            return ""

    def _contains_code(self, text: str) -> bool:
        """Stage 1: Check if text contains executable code."""
        if not text:
            return False

        # Code indicators
        code_indicators = [
            '```',  # Markdown code blocks
            '<code>',  # HTML code tags
            'import ',  # Python imports
            'from ',  # Python imports
            'def ',  # Python functions
            'class ',  # Python classes
            'function ',  # JavaScript functions
            'const ',  # JavaScript constants
            'let ',  # JavaScript variables
            'var ',  # JavaScript variables
            '#!/',  # Shebang
            'SELECT ',  # SQL
            'INSERT ',  # SQL
            'UPDATE ',  # SQL
            'curl ',  # API calls
            'POST ',  # HTTP methods
            'GET ',  # HTTP methods
            'openai.',  # OpenAI API calls
            'client.',  # API client calls
            'await ',  # Async code
            'async ',  # Async code
            '.api.',  # API calls
            'response =',  # API responses
            'request =',  # API requests
        ]

        text_lower = text.lower()
        code_count = sum(
            1 for indicator in code_indicators if indicator.lower() in text_lower)

        # Require at least 2 code indicators for high confidence
        return code_count >= 2

    async def _get_accepted_answer_score(
        self,
        question_id: str,
        accepted_answer_id: str,
        client: httpx.AsyncClient
    ) -> int:
        """Stage 3: Get the score of the accepted answer."""
        try:
            params = {
                'site': self.site,
                'filter': '!nNPvSNdWme'
            }
            if self.api_key:
                params['key'] = self.api_key

            url = f"{self.base_url}/answers/{accepted_answer_id}"
            response = await client.get(url, params=params)
            response.raise_for_status()

            # Handle response
            try:
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(response.content)
                    data = json.loads(content.decode('utf-8'))
                else:
                    data = response.json()
            except (gzip.BadGzipFile, json.JSONDecodeError):
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse answer response for {accepted_answer_id}")
                    return 0

            answers = data.get('items', [])
            if answers:
                return answers[0].get('score', 0)
            return 0

        except Exception as e:
            logger.error(
                f"Error fetching accepted answer score for {accepted_answer_id}: {str(e)}")
            return 0

    async def _convert_question_to_rawpost_with_comments(
        self,
        question: Dict[str, Any],
        original_tag: str,
        client: httpx.AsyncClient
    ) -> RawPost:
        """Convert Stack Overflow question to RawPost format with comments."""

        # Fetch comments for the question
        comments_text = await self._fetch_question_comments(
            str(question['question_id']), client
        )

        # Fetch answers for the question (accepted answer prioritized)
        answers_text = await self._fetch_question_answers_text(
            str(question['question_id']), client, question.get(
                'accepted_answer_id')
        )

        # Combine body with comments and answers
        body_with_comments = question.get('body', '') or ''
        if comments_text:
            body_with_comments += f"\n\n--- COMMENTS ---\n{comments_text}"
        if answers_text:
            body_with_comments += f"\n\n--- ANSWERS ---\n{answers_text}"

        return RawPost(
            platform=Platform.STACKOVERFLOW,
            source_id=str(question['question_id']),
            url=question['link'],
            title=question['title'],
            body_md=body_with_comments,
            created_at=datetime.fromtimestamp(question['creation_date']),
            updated_at=datetime.fromtimestamp(
                question.get('last_activity_date', question['creation_date'])
            ),
            score=question.get('score', 0),
            tags=question.get('tags', []) + [original_tag],
            author=question.get('owner', {}).get('display_name', 'unknown'),
            view_count=question.get('view_count', 0),
            comments_count=question.get('comment_count', 0),
            accepted_answer_id=question.get('accepted_answer_id'),
            acquisition_timestamp=datetime.utcnow(),
            acquisition_version="1.0.0"
        )

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
                            'page': page,
                            'min': self.min_score
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
                            # Check if we already have this post (if enabled)
                            if self.check_duplicates:
                                question_id = str(question['question_id'])
                                existing = await self.db.find_one(
                                    'raw_posts',
                                    {
                                        'platform': 'stackoverflow',
                                        'source_id': question_id
                                    }
                                )
                                if existing:
                                    continue

                            # Configurable filtering
                            if self.require_answered and not question.get('is_answered', False):
                                continue

                            if self.require_accepted_answer and not question.get('accepted_answer_id'):
                                continue

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
