"""
Utility functions for the LLM Contracts Research Pipeline.

Provides text normalization, datetime helpers, and other common utilities.
"""

import re
import string
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import unicodedata


class TextNormalizer:
    """Text normalization utilities for consistent processing."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        if not text:
            return ""

        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        return text.strip()

    @staticmethod
    def remove_markdown_formatting(text: str) -> str:
        """Remove basic markdown formatting for text analysis."""
        if not text:
            return ""

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', ' [CODE BLOCK] ', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'https?://[^\s]+', ' [URL] ', text)

        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Remove emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        return TextNormalizer.normalize_whitespace(text)

    @staticmethod
    def extract_code_snippets(text: str) -> List[str]:
        """Extract code snippets from markdown text."""
        if not text:
            return []

        snippets = []

        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        for block in code_blocks:
            # Remove the ``` markers and language identifier
            code = re.sub(r'^```[^\n]*\n?', '', block)
            code = re.sub(r'\n?```$', '', code)
            if code.strip():
                snippets.append(code.strip())

        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', text)
        snippets.extend(inline_code)

        return snippets

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        if not text:
            return ""

        # Normalize to NFKD form and remove diacritics
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))

        return text

    @staticmethod
    def clean_for_keyword_matching(text: str) -> str:
        """Clean text for keyword matching."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Normalize unicode
        text = TextNormalizer.normalize_unicode(text)

        # Remove punctuation except hyphens and underscores
        text = re.sub(r'[^\w\s\-_]', ' ', text)

        # Normalize whitespace
        text = TextNormalizer.normalize_whitespace(text)

        return text

    @staticmethod
    def extract_error_messages(text: str) -> List[str]:
        """Extract error messages from text."""
        if not text:
            return []

        error_patterns = [
            r'Error:\s*`([^`]+)`',
            r'Error:\s*([^\n]+)',
            r'Exception:\s*([^\n]+)',
            r'Traceback[\s\S]*?(\w+Error:.*?)(?:\n|$)',
            r'HTTP\s+(\d{3}[^\n]*)',
            r'(\d{3}\s+[A-Za-z][^\n]*)',
        ]

        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            errors.extend(matches)

        return [error.strip() for error in errors if error.strip()]


class DateTimeHelpers:
    """Date and time utility functions."""

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)

    @staticmethod
    def parse_iso_string(iso_string: str) -> Optional[datetime]:
        """Parse ISO format datetime string."""
        if not iso_string:
            return None

        try:
            # Handle various ISO formats
            formats = [
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(iso_string, fmt)
                except ValueError:
                    continue

            # If no format matches, try fromisoformat (Python 3.7+)
            return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))

        except (ValueError, AttributeError):
            return None

    @staticmethod
    def to_iso_string(dt: datetime) -> str:
        """Convert datetime to ISO string."""
        if not dt:
            return ""

        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.isoformat()

    @staticmethod
    def days_ago(days: int) -> datetime:
        """Get datetime that is N days ago from now."""
        from datetime import timedelta
        return DateTimeHelpers.utc_now() - timedelta(days=days)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    @staticmethod
    def is_recent(dt: datetime, hours: int = 24) -> bool:
        """Check if datetime is within the last N hours."""
        if not dt:
            return False

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        from datetime import timedelta
        cutoff = DateTimeHelpers.utc_now() - timedelta(hours=hours)
        return dt >= cutoff


class ValidationHelpers:
    """Validation utility functions."""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if string is a valid URL."""
        if not url:
            return False

        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            # domain...
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return bool(url_pattern.match(url))

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if string is a valid email."""
        if not email:
            return False

        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return bool(email_pattern.match(email))

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem use."""
        if not filename:
            return "untitled"

        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')

        # Limit length
        if len(filename) > 255:
            filename = filename[:255]

        return filename or "untitled"

    @staticmethod
    def validate_confidence_score(score: float) -> float:
        """Validate and clamp confidence score to [0.0, 1.0]."""
        if score is None:
            return 0.0

        return max(0.0, min(1.0, float(score)))


class DataStructureHelpers:
    """Helpers for working with data structures."""

    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DataStructureHelpers.flatten_dict(
                    v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def safe_get(d: Dict[str, Any], keys: str, default: Any = None) -> Any:
        """Safely get nested dictionary value using dot notation."""
        try:
            value = d
            for key in keys.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError, AttributeError):
            return default

    @staticmethod
    def remove_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from dictionary."""
        return {k: v for k, v in d.items() if v is not None}

    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple dictionaries, with later ones taking precedence."""
        result = {}
        for d in dicts:
            if d:
                result.update(d)
        return result
