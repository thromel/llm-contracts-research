"""Simple test script for GitHub analyzer."""
from analysis.core.analyzers.github_analyzer import GitHubIssuesAnalyzer
from analysis.core.clients.openai import OpenAIClient
import asyncio
import os
import sys
sys.path.append('.')


async def test_analyzer():
    """Test the GitHub analyzer with a simple issue."""
    try:
        # Initialize the LLM client
        llm_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

        # Initialize the analyzer
        analyzer = GitHubIssuesAnalyzer(
            llm_client=llm_client,
            github_token=os.getenv('GITHUB_TOKEN')
        )

        # Test with a simple issue
        title = "API breaking change in v2.0"
        body = """
        The new version 2.0 removed the `get_data()` method without any deprecation warning.
        This breaks all existing code that relies on this method.
        
        Expected behavior: The method should still work or there should have been a deprecation warning.
        Actual behavior: Method is completely removed and throws AttributeError.
        """
        comments = "This is a major breaking change that affects our production system."

        print("Testing GitHub Issues Analyzer...")
        print(f"Title: {title}")
        print(f"Body: {body[:100]}...")

        # Analyze the issue
        result = analyzer.analyze_issue(title, body, comments)

        print("\nAnalysis Result:")
        print(f"Has violation: {result.has_violation}")
        if result.has_violation:
            print(f"Violation type: {result.violation_type}")
            print(f"Severity: {result.severity}")
            print(f"Confidence: {result.confidence}")
            print(f"Description: {result.description}")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_analyzer())
