"""OpenAI client implementation."""

import logging
import backoff
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client implementation."""

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use
            **kwargs: Additional settings for completions
        """
        # Extract OpenAI client kwargs
        client_kwargs = {
            'api_key': api_key,
            'max_retries': kwargs.pop('max_retries', 3),
            'timeout': kwargs.pop('timeout', 30.0)
        }
        if 'base_url' in kwargs:
            client_kwargs['base_url'] = kwargs.pop('base_url')

        # Initialize client
        self.client = OpenAI(**client_kwargs)

        # Store model and completion settings
        self.model = model
        self.completion_settings = {
            'temperature': kwargs.get('temperature', 0.1),
            'max_tokens': kwargs.get('max_tokens', 1000),
            'top_p': kwargs.get('top_p', 1.0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0)
        }

        # Log configuration
        logger.info(f"Initialized OpenAI client with model: {model}")
        if 'base_url' in client_kwargs:
            logger.info(f"Using custom base URL: {client_kwargs['base_url']}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=settings.MAX_RETRIES,
        giveup=lambda e: not self._should_retry(e)
    )
    def get_analysis(self, system_prompt: str, user_prompt: str) -> str:
        """Get analysis from OpenAI.

        Args:
            system_prompt: System context prompt
            user_prompt: User query prompt

        Returns:
            Analysis response content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.completion_settings
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise

    def _should_retry(self, e: Exception) -> bool:
        """Determine if error should trigger retry."""
        if hasattr(e, 'status_code'):
            if e.status_code == 429:  # Rate limit
                logger.warning("Rate limit hit, backing off...")
                return True
            elif e.status_code >= 500:  # Server error
                logger.warning(
                    f"OpenAI server error {e.status_code}, retrying...")
                return True
        return False
