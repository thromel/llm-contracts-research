#!/usr/bin/env python3
"""
Test script for the Advanced Keyword Filtering System

Demonstrates the new multi-stage filtering capabilities with
semantic categories, pattern matching, and confidence scoring.
"""

import yaml
from datetime import datetime
from pipeline.common.models import RawPost
from pipeline.common.database import MongoDBManager
from pipeline.preprocessing.keyword_filter import AdvancedKeywordFilter
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FilteringDemonstrator:
    """Demonstrates advanced keyword filtering capabilities."""

    def __init__(self):
        self.db = MongoDBManager()
        self.filter = AdvancedKeywordFilter(self.db)

    def create_test_posts(self) -> list[RawPost]:
        """Create test posts representing different quality levels."""

        test_posts = [
            # HIGH QUALITY - Clear contract violation
            RawPost(
                _id="test_1",
                platform="github",
                source_id="issue_1",
                title="OpenAI API rate_limit exceeded error - max_tokens parameter issue",
                body_md="""
I'm getting this error when calling the OpenAI API:

```
HTTP 429: Rate limit exceeded
{
  "error": {
    "message": "You exceeded your current quota, please check your plan and billing details.",
    "type": "invalid_request_error",
    "param": "max_tokens"
  }
}
```

My code:
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=10000,  # This seems to be the problem
    temperature=0.7
)
```

Expected: API should work with max_tokens=10000
Actual: Getting rate limit error even though I have credits

The API documentation says max_tokens can be up to 4096 for gpt-4, but I'm setting 10000.
""",
                tags=["openai-api", "rate-limiting", "api-error"],
                labels=["bug"]
            ),

            # MEDIUM QUALITY - API issue but less clear
            RawPost(
                _id="test_2",
                platform="stackoverflow",
                source_id="question_2",
                title="Claude API returning invalid JSON response",
                body_md="""
I'm using the Anthropic Claude API and sometimes get malformed JSON responses.
The response_format is set to json_mode but it's not working consistently.

```python
client = anthropic.Anthropic(api_key=api_key)
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Return JSON"}],
    response_format={"type": "json_object"}
)
```

Sometimes works, sometimes doesn't. Any ideas?
""",
                tags=["anthropic", "claude-api", "json"],
                labels=[]
            ),

            # MEDIUM QUALITY - LLM framework issue
            RawPost(
                _id="test_3",
                platform="github",
                source_id="issue_3",
                title="LangChain token count error with large documents",
                body_md="""
Getting token limit exceeded error when processing large documents with LangChain:

```
langchain.schema.exceptions.InvalidRequestError: This model's maximum context length is 8192 tokens. 
However, your messages resulted in 12000 tokens.
```

Using:
- langchain==0.1.0
- OpenAI gpt-3.5-turbo
- Document chunking with RecursiveCharacterTextSplitter

The document is being split but somehow still exceeding context_length.
""",
                tags=["langchain", "token-limit", "openai"],
                labels=["enhancement"]
            ),

            # LOW QUALITY - Tutorial/conceptual question
            RawPost(
                _id="test_4",
                platform="stackoverflow",
                source_id="question_4",
                title="What is the difference between GPT-3.5 and GPT-4?",
                body_md="""
I'm new to AI and want to understand the differences between GPT-3.5 and GPT-4.

What are the main advantages of GPT-4? 
Is it worth the extra cost?
Which one should I use for my chatbot project?

I'm a beginner and would appreciate a simple explanation.
""",
                tags=["gpt-4", "gpt-3.5", "comparison"],
                labels=[]
            ),

            # LOW QUALITY - Installation help
            RawPost(
                _id="test_5",
                platform="github",
                source_id="issue_5",
                title="How to install OpenAI Python package?",
                body_md="""
I'm trying to install the OpenAI Python package but getting errors.

```
pip install openai
```

I'm on Python 3.8. Is this compatible?
Also, where do I get an API key?

This is my first time using the OpenAI API.
""",
                tags=["installation", "python"],
                labels=["question"]
            ),

            # HIGH QUALITY - Schema validation error
            RawPost(
                _id="test_6",
                platform="stackoverflow",
                source_id="question_6",
                title="OpenAI function_calling schema validation failed",
                body_md="""
I'm getting schema validation errors when using OpenAI function calling:

```
openai.BadRequestError: Error code: 400
{
  "error": {
    "message": "Invalid schema for function 'get_weather': Expected object, got string at path 'parameters.properties.location.type'",
    "type": "invalid_request_error"
  }
}
```

My function schema:
```json
{
  "name": "get_weather",
  "description": "Get weather information",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",  # This line seems wrong
        "description": "City name"
      }
    },
    "required": ["location"]
  }
}
```

According to the JSON schema spec, this should be valid. What's wrong?
""",
                tags=["openai-api", "function-calling", "json-schema"],
                labels=["api-error"]
            ),

            # VERY LOW QUALITY - Off-topic
            RawPost(
                _id="test_7",
                platform="github",
                source_id="issue_7",
                title="Python syntax help needed",
                body_md="""
I'm learning Python and need help with basic syntax.

How do I declare a variable?
What's the difference between lists and tuples?

Please help, I'm completely lost and this is urgent for my homework.
""",
                tags=["python", "beginner"],
                labels=["help-wanted"]
            ),
        ]

        return test_posts

    async def demonstrate_filtering(self):
        """Run the filtering demonstration."""
        print("🔍 Advanced Keyword Filtering Demonstration")
        print("=" * 60)

        test_posts = self.create_test_posts()

        results = []

        for i, post in enumerate(test_posts, 1):
            print(f"\n📄 Test Post {i}: {post.title[:50]}...")
            print("-" * 50)

            # Apply the advanced filter
            filter_result = self.filter.apply_filter(post)

            # Store result for summary
            results.append({
                'post_id': post._id,
                'title': post.title,
                'passed': filter_result.passed,
                'confidence': filter_result.confidence,
                'categories': filter_result.filter_metadata['categories_matched'],
                'patterns': filter_result.filter_metadata['patterns_matched'],
                'quality': filter_result.filter_metadata['quality_score'],
                'decision_factors': filter_result.filter_metadata['decision_factors']
            })

            # Display results
            print(f"✅ PASSED: {filter_result.passed}")
            print(f"🎯 CONFIDENCE: {filter_result.confidence:.3f}")
            print(
                f"⭐ QUALITY: {filter_result.filter_metadata['quality_score']:.3f}")

            print(f"\n📊 Categories Matched:")
            categories = filter_result.filter_metadata['categories_matched']
            for category, count in categories.items():
                if count > 0:
                    print(f"  • {category}: {count}")

            if filter_result.filter_metadata['patterns_matched'] > 0:
                print(
                    f"🔍 Pattern Matches: {filter_result.filter_metadata['patterns_matched']}")

            if filter_result.matched_keywords:
                print(
                    f"🔑 Top Keywords: {', '.join(filter_result.matched_keywords[:5])}")

            if filter_result.relevant_snippets:
                print(
                    f"📝 Relevant Snippet: {filter_result.relevant_snippets[0][:100]}...")

            print(f"\n🧠 Decision Factors:")
            factors = filter_result.filter_metadata['decision_factors']
            for factor, value in factors.items():
                print(f"  • {factor}: {value}")

        # Summary
        print("\n" + "=" * 60)
        print("📈 FILTERING SUMMARY")
        print("=" * 60)

        passed_posts = [r for r in results if r['passed']]
        failed_posts = [r for r in results if not r['passed']]

        print(f"Total Posts: {len(results)}")
        print(
            f"✅ Passed: {len(passed_posts)} ({len(passed_posts)/len(results)*100:.1f}%)")
        print(
            f"❌ Failed: {len(failed_posts)} ({len(failed_posts)/len(results)*100:.1f}%)")

        if passed_posts:
            avg_confidence = sum(r['confidence']
                                 for r in passed_posts) / len(passed_posts)
            print(f"📊 Average Confidence (Passed): {avg_confidence:.3f}")

            high_conf = len(
                [r for r in passed_posts if r['confidence'] >= 0.7])
            med_conf = len(
                [r for r in passed_posts if 0.4 <= r['confidence'] < 0.7])
            low_conf = len([r for r in passed_posts if r['confidence'] < 0.4])

            print(f"🎯 Confidence Distribution:")
            print(f"   High (≥0.7): {high_conf}")
            print(f"   Med (0.4-0.7): {med_conf}")
            print(f"   Low (<0.4): {low_conf}")

        print(f"\n🏆 Quality Rankings:")
        sorted_results = sorted(
            results, key=lambda x: x['confidence'], reverse=True)
        for i, result in enumerate(sorted_results, 1):
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(
                f"{i}. {status} | {result['confidence']:.3f} | {result['title'][:40]}...")

        print(f"\n📋 Detailed Analysis:")
        print(
            f"Contract Violation Posts: {len([r for r in results if r['categories']['contract_violations'] > 0])}")
        print(
            f"API Error Posts: {len([r for r in results if r['categories']['error_indicators'] > 0])}")
        print(
            f"Pattern Match Posts: {len([r for r in results if r['patterns'] > 0])}")
        print(
            f"High Quality Posts: {len([r for r in results if r['quality'] > 0.6])}")

        return results

    async def test_configuration_loading(self):
        """Test loading configuration from pipeline_config.yaml."""
        print("\n🔧 Configuration Test")
        print("-" * 30)

        try:
            with open(project_root / 'pipeline_config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            if 'keyword_filtering' in config:
                filter_config = config['keyword_filtering']
                print("✅ Advanced filtering configuration found:")
                print(
                    f"  • Confidence threshold: {filter_config.get('confidence_threshold', 'N/A')}")
                print(
                    f"  • Quality threshold: {filter_config.get('quality_threshold', 'N/A')}")
                print(
                    f"  • Batch size: {filter_config.get('batch_size', 'N/A')}")
                print(
                    f"  • Negative filtering: {filter_config.get('negative_filtering', {}).get('enabled', 'N/A')}")
            else:
                print(
                    "⚠️  Advanced filtering configuration not found in pipeline_config.yaml")

        except Exception as e:
            print(f"❌ Error loading configuration: {e}")

    async def run_demonstration(self):
        """Run the complete demonstration."""
        try:
            print("🚀 Starting Advanced Keyword Filtering Demonstration\n")

            # Test configuration
            await self.test_configuration_loading()

            # Run filtering demo
            results = await self.demonstrate_filtering()

            print(
                f"\n✨ Demonstration complete! The new advanced filtering system provides:")
            print(f"  • Multi-stage semantic analysis")
            print(f"  • Context-aware pattern matching")
            print(f"  • Quality-based confidence scoring")
            print(f"  • Negative signal filtering")
            print(f"  • Detailed decision transparency")

        except Exception as e:
            print(f"❌ Error during demonstration: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main function."""
    demonstrator = FilteringDemonstrator()
    await demonstrator.run_demonstration()


if __name__ == "__main__":
    asyncio.run(main())
