"""Utility to extract context from research papers and generate context files."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Define paper directory relative to project root
PAPERS_DIR = Path(settings.BASE_DIR) / 'papers'


class ContextExtractor:
    def __init__(self, model: str = None):
        """Initialize the context extractor."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in settings")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.OPENAI_MODEL
        self.context_dir = Path(settings.CONTEXT_DIR)
        self.papers_dir = PAPERS_DIR

        # Create papers directory if it doesn't exist
        self.papers_dir.mkdir(exist_ok=True)

    def list_papers(self) -> List[Path]:
        """List all available papers in the papers directory."""
        return list(self.papers_dir.glob('*.{txt,md,pdf}'))

    def extract_paper_context(self, paper_text: str) -> Dict[str, Any]:
        """Extract key findings and context from the research paper."""
        prompt = """Analyze this research paper on LLM API contract violations and extract the following information:

1. Key Findings: What are the main discoveries about LLM contract violations?
2. Methodology: How were contract violations identified and classified?
3. Contract Definition: How are contracts defined in the context of LLM APIs?
4. Violation Criteria: What criteria determine contract violations?
5. Impact Assessment: How is the impact of violations measured?

Format your response in markdown with clear sections."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant expert in LLM contracts and API design."},
                {"role": "user", "content": "Here's the paper text:\n\n" +
                    paper_text + "\n\n" + prompt}
            ],
            temperature=0.1
        )

        return response.choices[0].message.content

    def generate_violation_types(self, paper_text: str) -> Dict[str, Any]:
        """Generate structured violation types from the paper."""
        prompt = """Extract and categorize the types of LLM API contract violations discussed in the paper.
        Format the response as a YAML-compatible dictionary with:
        - violation_type: description"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant expert in LLM contracts and API design."},
                {"role": "user", "content": "Here's the paper text:\n\n" +
                    paper_text + "\n\n" + prompt}
            ],
            temperature=0.1
        )

        return yaml.safe_load(response.choices[0].message.content)

    def generate_severity_criteria(self, paper_text: str) -> Dict[str, Any]:
        """Generate severity criteria for violations from the paper."""
        prompt = """Extract the criteria used to assess the severity of LLM API contract violations.
        Format the response as a YAML-compatible dictionary with:
        - criterion: description and impact level"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant expert in LLM contracts and API design."},
                {"role": "user", "content": "Here's the paper text:\n\n" +
                    paper_text + "\n\n" + prompt}
            ],
            temperature=0.1
        )

        return yaml.safe_load(response.choices[0].message.content)

    def generate_categorization(self, paper_text: str) -> Dict[str, Any]:
        """Generate violation categorization schema from the paper."""
        prompt = """Extract the categorization schema used for LLM API contract violations.
        Format the response as a YAML-compatible dictionary with categories and subcategories."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant expert in LLM contracts and API design."},
                {"role": "user", "content": "Here's the paper text:\n\n" +
                    paper_text + "\n\n" + prompt}
            ],
            temperature=0.1
        )

        return yaml.safe_load(response.choices[0].message.content)

    def generate_examples(self, paper_text: str) -> Dict[str, Any]:
        """Generate example violations from the paper."""
        prompt = """Extract concrete examples of LLM API contract violations from the paper.
        Format the response as a YAML-compatible dictionary with:
        - example_name:
            description: detailed description
            type: violation type
            severity: severity level
            impact: impact description"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant expert in LLM contracts and API design."},
                {"role": "user", "content": "Here's the paper text:\n\n" +
                    paper_text + "\n\n" + prompt}
            ],
            temperature=0.1
        )

        return yaml.safe_load(response.choices[0].message.content)

    def save_context_files(self, paper_path: str | Path) -> None:
        """Generate and save all context files from a paper."""
        paper_path = Path(paper_path)
        if not paper_path.exists():
            # If not found, check in papers directory
            paper_path = self.papers_dir / paper_path
            if not paper_path.exists():
                raise FileNotFoundError(f"Paper not found: {paper_path}")

        logger.info(f"Reading paper from: {paper_path}")

        # Read paper text
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_text = f.read()

        logger.info("Extracting context from paper...")

        # Create context directory if it doesn't exist
        self.context_dir.mkdir(exist_ok=True)

        # Extract and save paper context
        paper_context = self.extract_paper_context(paper_text)
        with open(self.context_dir / 'paper.md', 'w') as f:
            f.write(paper_context)
        logger.info("Saved paper context")

        # Generate and save violation types
        violation_types = self.generate_violation_types(paper_text)
        with open(self.context_dir / 'violation_types.yaml', 'w') as f:
            yaml.dump(violation_types, f, sort_keys=False, allow_unicode=True)
        logger.info("Saved violation types")

        # Generate and save severity criteria
        severity_criteria = self.generate_severity_criteria(paper_text)
        with open(self.context_dir / 'severity_criteria.yaml', 'w') as f:
            yaml.dump(severity_criteria, f,
                      sort_keys=False, allow_unicode=True)
        logger.info("Saved severity criteria")

        # Generate and save categorization
        categorization = self.generate_categorization(paper_text)
        with open(self.context_dir / 'categorization.yaml', 'w') as f:
            yaml.dump(categorization, f, sort_keys=False, allow_unicode=True)
        logger.info("Saved categorization schema")

        # Generate and save examples
        examples = self.generate_examples(paper_text)
        with open(self.context_dir / 'examples.yaml', 'w') as f:
            yaml.dump(examples, f, sort_keys=False, allow_unicode=True)
        logger.info("Saved example violations")


def main():
    """Main function to extract context from paper."""
    extractor = ContextExtractor()

    # List available papers
    papers = extractor.list_papers()
    if not papers:
        logger.error(f"No papers found in {PAPERS_DIR}")
        logger.info(
            "Please add your research papers (txt, md, pdf) to the 'papers' directory")
        return

    # If multiple papers, let user choose
    if len(papers) > 1:
        print("\nAvailable papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.name}")
        choice = input("\nEnter the number of the paper to process: ")
        try:
            paper_path = papers[int(choice) - 1]
        except (ValueError, IndexError):
            logger.error("Invalid choice")
            return
    else:
        paper_path = papers[0]

    # Extract and save context
    try:
        extractor.save_context_files(paper_path)
        logger.info("Context extraction complete")
    except Exception as e:
        logger.error(f"Error during context extraction: {str(e)}")


if __name__ == '__main__':
    main()
