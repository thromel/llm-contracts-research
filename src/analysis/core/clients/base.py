"""Base protocol for LLM clients."""

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for LLM API clients."""

    def get_analysis(self, system_prompt: str, user_prompt: str) -> str:
        """Get analysis from LLM.

        Args:
            system_prompt: System context prompt
            user_prompt: User query prompt

        Returns:
            Analysis response content
        """
        pass
