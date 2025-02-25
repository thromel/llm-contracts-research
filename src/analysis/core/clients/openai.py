"""OpenAI client for analysis."""
import logging
import json
import openai
import backoff
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use
            base_url: Optional base URL for API (for Azure OpenAI)
            max_retries: Maximum number of retries
            timeout: Timeout in seconds
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # Configure OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)

        logger.info(f"Initialized OpenAI client with model: {model}")

    @backoff.on_exception(
        backoff.expo,
        (openai.OpenAIError, openai.APITimeoutError),
        max_tries=3
    )
    def analyze(self, prompt: str) -> str:
        """Analyze text using OpenAI.

        Args:
            prompt: Text to analyze

        Returns:
            Analysis result
        """
        try:
            logger.debug(f"Sending prompt to OpenAI: {prompt[:100]}...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                        "content": "You are an expert API developer and analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )

            # Extract the content from the response
            content = response.choices[0].message.content if response.choices else ""

            logger.debug(f"Got response from OpenAI: {content[:100]}...")
            return content

        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {str(e)}")
            raise
