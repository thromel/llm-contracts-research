import os
import json
import yaml
from typing import Dict, Any, List


class ContextManager:
    def __init__(self, context_dir: str = "context"):
        """Initialize context manager with directory containing context files."""
        self.context_dir = context_dir
        self.contexts = {
            'paper': None,
            'violation_types': None,
            'severity_criteria': None,
            'categorization': None,
            'examples': None
        }

    def load_contexts(self) -> None:
        """Load all context files from the context directory."""
        os.makedirs(self.context_dir, exist_ok=True)

        # Load each context file if it exists
        for context_type in self.contexts.keys():
            # Try different file extensions
            for ext in ['.yaml', '.json', '.md']:
                filepath = os.path.join(
                    self.context_dir, f"{context_type}{ext}")
                if os.path.exists(filepath):
                    self.contexts[context_type] = self._load_file(filepath)
                    break

    def _load_file(self, filepath: str) -> Any:
        """Load a single context file based on its extension."""
        ext = os.path.splitext(filepath)[1].lower()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if ext == '.json':
                    return json.load(f)
                elif ext == '.yaml':
                    return yaml.safe_load(f)
                else:  # .md or other text files
                    return f.read()
        except Exception as e:
            print(f"Error loading context file {filepath}: {str(e)}")
            return None

    def generate_system_prompt(self) -> str:
        """Generate the system prompt from loaded contexts."""
        if not any(self.contexts.values()):
            raise ValueError(
                "No context files loaded. Please add context files to the 'context' directory.")

        prompt_parts = []

        # Add paper context
        if self.contexts['paper']:
            prompt_parts.append("Based on the research paper:")
            prompt_parts.append(self.contexts['paper'])

        # Add violation types
        if self.contexts['violation_types']:
            prompt_parts.append("\nContract violation types include:")
            if isinstance(self.contexts['violation_types'], list):
                for vtype in self.contexts['violation_types']:
                    prompt_parts.append(f"- {vtype}")
            else:
                prompt_parts.append(str(self.contexts['violation_types']))

        # Add severity criteria
        if self.contexts['severity_criteria']:
            prompt_parts.append("\nSeverity levels are determined by:")
            prompt_parts.append(str(self.contexts['severity_criteria']))

        # Add categorization
        if self.contexts['categorization']:
            prompt_parts.append("\nIssue categorization guidelines:")
            prompt_parts.append(str(self.contexts['categorization']))

        # Add examples if available
        if self.contexts['examples']:
            prompt_parts.append("\nReference examples:")
            prompt_parts.append(str(self.contexts['examples']))

        # Add analysis instructions
        prompt_parts.append("""
For each issue, provide a detailed analysis in the following JSON format:
{
    "is_violation": boolean,
    "violation_type": string | null,
    "severity": "high" | "medium" | "low" | null,
    "impact": {
        "description": string,
        "affected_users": string,
        "scope": string
    },
    "resolution": {
        "was_addressed": boolean,
        "resolution_type": string,
        "effectiveness": string
    },
    "categorization": {
        "primary_category": string,
        "subcategory": string,
        "tags": [string]
    },
    "reasoning": {
        "key_points": [string],
        "evidence": [string],
        "confidence": number
    }
}
""")

        return "\n".join(prompt_parts)

    def get_missing_contexts(self) -> List[str]:
        """Return a list of missing context types."""
        return [k for k, v in self.contexts.items() if v is None]

    def validate_contexts(self) -> Dict[str, bool]:
        """Validate that all required contexts are properly formatted."""
        validation = {}

        for context_type, content in self.contexts.items():
            if content is None:
                validation[context_type] = False
                continue

            try:
                if context_type == 'violation_types':
                    validation[context_type] = isinstance(content, (list, str))
                elif context_type == 'severity_criteria':
                    validation[context_type] = isinstance(content, (dict, str))
                elif context_type == 'examples':
                    validation[context_type] = isinstance(content, (list, str))
                else:
                    validation[context_type] = True
            except Exception:
                validation[context_type] = False

        return validation
