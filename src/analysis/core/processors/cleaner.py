"""Response cleaning implementations."""

from typing import Protocol


class ResponseCleaner(Protocol):
    """Protocol for response cleaning strategies."""

    def clean(self, content: str) -> str:
        """Clean the response content.

        Args:
            content: Raw response content

        Returns:
            Cleaned content
        """
        pass


class MarkdownResponseCleaner:
    """Cleans markdown-formatted responses."""

    def clean(self, content: str) -> str:
        """Remove markdown formatting from response.

        Args:
            content: Raw response content

        Returns:
            Cleaned content
        """
        if content.startswith('```json\n'):
            content = content.replace('```json\n', '', 1)
            if content.endswith('\n```'):
                content = content[:-4]
        elif content.startswith('```\n'):
            content = content.replace('```\n', '', 1)
            if content.endswith('\n```'):
                content = content[:-4]
        return content.strip()
