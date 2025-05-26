#!/usr/bin/env python3
"""
End-to-End Pipeline Test with Mock Data.

This script demonstrates the complete pipeline flow:
1. Mock data generation
2. Keyword filtering
3. LLM screening (mock API calls)
4. Results processing

No external dependencies required (MongoDB, APIs, etc.)
"""

from pipeline.common.config import get_development_config
from pipeline.common.models import RawPost, FilteredPost, LLMScreeningResult, Platform
import asyncio
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
import json

# Add the pipeline to the path
sys.path.insert(0, '.')


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDatabaseManager:
    """Mock database manager for testing without MongoDB."""

    def __init__(self):
        self.collections = {
            'raw_posts': [],
            'filtered_posts': [],
            'llm_screening_results': []
        }
        self.id_counter = 1

    async def connect(self):
        logger.info("📊 Connected to mock database")

    async def disconnect(self):
        logger.info("📊 Disconnected from mock database")

    async def insert_one(self, collection: str, document: Dict[str, Any]):
        """Mock insert operation."""
        document['_id'] = self.id_counter
        self.id_counter += 1
        self.collections[collection].append(document)

        class MockResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id

        return MockResult(document['_id'])

    async def find_one(self, collection: str, query: Dict[str, Any]):
        """Mock find one operation."""
        for doc in self.collections[collection]:
            if '_id' in query and doc['_id'] == query['_id']:
                return doc
        return None

    async def get_posts_for_screening(self, batch_size: int = 50, screening_type: str = "mock"):
        """Mock method to get posts for screening."""
        for post in self.collections['filtered_posts'][:batch_size]:
            if post.get('passed_keyword_filter', False) and not post.get('llm_screened', False):
                yield post

    async def save_screening_result(self, result_dict: Dict[str, Any]) -> str:
        """Mock save screening result."""
        result = await self.insert_one('llm_screening_results', result_dict)

        # Mark the filtered post as screened
        for post in self.collections['filtered_posts']:
            if post['_id'] == result_dict['filtered_post_id']:
                post['llm_screened'] = True
                break

        return str(result.inserted_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'raw_posts': len(self.collections['raw_posts']),
            'filtered_posts': len(self.collections['filtered_posts']),
            'llm_screening_results': len(self.collections['llm_screening_results']),
            'passed_filter': len([p for p in self.collections['filtered_posts'] if p.get('passed_keyword_filter', False)]),
            'screened': len([p for p in self.collections['filtered_posts'] if p.get('llm_screened', False)])
        }


class MockLLMScreener:
    """Mock LLM screener for testing without API calls."""

    def __init__(self, db_manager: MockDatabaseManager):
        self.db = db_manager
        self.model = "mock-model-v1"

    async def screen_single_post(self, filtered_post: Dict[str, Any]) -> LLMScreeningResult:
        """Mock screening of a single post."""
        # Get the "raw post"
        raw_post = await self.db.find_one('raw_posts', {'_id': filtered_post['raw_post_id']})

        if not raw_post:
            raise ValueError(
                f"Raw post not found for filtered post {filtered_post['_id']}")

        # Simple mock logic based on content
        title = raw_post.get('title', '').lower()
        content = raw_post.get('body_md', '').lower()

        # Mock decision logic
        contract_keywords = ['rate limit', 'api error',
                             'max_tokens', 'openai', 'gpt', 'authentication', 'quota']
        violation_count = sum(
            1 for keyword in contract_keywords if keyword in title + ' ' + content)

        if violation_count >= 2:
            decision = "Y"
            confidence = min(0.9, 0.6 + (violation_count * 0.1))
            rationale = f"Mock analysis found {violation_count} contract-related indicators"
        elif violation_count >= 1:
            decision = "Borderline"
            confidence = 0.5 + (violation_count * 0.1)
            rationale = f"Mock analysis found {violation_count} potential contract indicator"
        else:
            decision = "N"
            confidence = 0.8
            rationale = "Mock analysis found no clear contract violations"

        return LLMScreeningResult(
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            model_used=self.model
        )

    async def screen_batch(self, batch_size: int = 10) -> Dict[str, Any]:
        """Mock batch screening."""
        stats = {
            'processed': 0,
            'positive_decisions': 0,
            'negative_decisions': 0,
            'borderline_cases': 0,
            'high_confidence': 0,
            'errors': 0
        }

        async for filtered_post in self.db.get_posts_for_screening(batch_size):
            try:
                result = await self.screen_single_post(filtered_post)
                stats['processed'] += 1

                if result.decision == 'Y':
                    stats['positive_decisions'] += 1
                elif result.decision == 'N':
                    stats['negative_decisions'] += 1
                else:
                    stats['borderline_cases'] += 1

                if result.confidence >= 0.8:
                    stats['high_confidence'] += 1

                # Save result
                result_doc = {
                    'filtered_post_id': filtered_post['_id'],
                    'decision': result.decision,
                    'confidence': result.confidence,
                    'rationale': result.rationale,
                    'model_used': result.model_used,
                    'created_at': datetime.utcnow()
                }

                await self.db.save_screening_result(result_doc)

            except Exception as e:
                logger.error(
                    f"Error screening post {filtered_post['_id']}: {str(e)}")
                stats['errors'] += 1

        return stats


def create_mock_raw_posts() -> List[Dict[str, Any]]:
    """Create mock raw posts for testing."""

    mock_posts = [
        {
            'platform': 'github',
            'source_id': 'test-1',
            'url': 'https://github.com/test/repo/issues/1',
            'title': 'OpenAI API rate limit exceeded error',
            'body_md': '''I'm getting a rate limit error when calling the OpenAI API:

```python
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Hello world",
    max_tokens=150
)
```

The error message is: `Rate limit exceeded. Please try again later.`

How can I handle this properly?''',
            'created_at': datetime.utcnow(),
            'score': 5,
            'tags': ['openai', 'api', 'python'],
            'author': 'test_user_1'
        },
        {
            'platform': 'stackoverflow',
            'source_id': 'test-2',
            'url': 'https://stackoverflow.com/questions/12345',
            'title': 'Invalid max_tokens parameter in GPT-4 API call',
            'body_md': '''I'm trying to use GPT-4 API but getting this error:

```
{
  "error": {
    "message": "max_tokens must be between 1 and 4096",
    "type": "invalid_request_error"
  }
}
```

My code:
```javascript
const response = await openai.createCompletion({
  model: "gpt-4",
  prompt: "Explain quantum computing",
  max_tokens: 5000  // This seems to be the problem
});
```

What's the correct range for max_tokens?''',
            'created_at': datetime.utcnow(),
            'score': 12,
            'tags': ['gpt-4', 'openai-api', 'javascript', 'max-tokens'],
            'author': 'test_user_2'
        },
        {
            'platform': 'github',
            'source_id': 'test-3',
            'url': 'https://github.com/test/repo/issues/3',
            'title': 'How to install Python on Windows',
            'body_md': '''I'm having trouble installing Python on Windows 10. 

When I download from python.org and run the installer, it gets stuck.

Has anyone encountered this before? Any suggestions?''',
            'created_at': datetime.utcnow(),
            'score': 2,
            'tags': ['python', 'windows', 'installation'],
            'author': 'test_user_3'
        },
        {
            'platform': 'stackoverflow',
            'source_id': 'test-4',
            'url': 'https://stackoverflow.com/questions/54321',
            'title': 'Anthropic Claude API authentication error',
            'body_md': '''Getting an authentication error with Claude API:

```
HTTP 401: {"error": "Authentication failed"}
```

I'm using the API key like this:
```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

But the documentation shows a different format. What's the correct way?''',
            'created_at': datetime.utcnow(),
            'score': 8,
            'tags': ['anthropic', 'claude', 'api', 'authentication'],
            'author': 'test_user_4'
        },
        {
            'platform': 'github',
            'source_id': 'test-5',
            'url': 'https://github.com/test/repo/issues/5',
            'title': 'Best practices for machine learning',
            'body_md': '''What are some general best practices for machine learning projects?

I'm starting a new project and want to make sure I follow good practices from the beginning.

Areas I'm particularly interested in:
- Data preprocessing
- Model selection  
- Evaluation metrics
- Deployment strategies

Any recommendations for resources or frameworks?''',
            'created_at': datetime.utcnow(),
            'score': 15,
            'tags': ['machine-learning', 'best-practices', 'data-science'],
            'author': 'test_user_5'
        }
    ]

    return mock_posts


async def simulate_keyword_filtering(db: MockDatabaseManager, raw_posts: List[Dict[str, Any]]) -> None:
    """Simulate keyword filtering process."""
    logger.info("🔍 Starting keyword filtering simulation...")

    # Simple keyword lists for simulation
    contract_keywords = [
        'api', 'openai', 'gpt', 'claude', 'anthropic', 'rate limit', 'quota',
        'max_tokens', 'authentication', 'error', 'token', 'request', 'response'
    ]

    filtered_count = 0
    passed_count = 0

    for i, raw_post in enumerate(raw_posts):
        # Save raw post first
        raw_post['_id'] = i + 1
        raw_post['acquisition_timestamp'] = datetime.utcnow()
        raw_post['acquisition_version'] = '1.0.0'
        await db.insert_one('raw_posts', raw_post)

        # Apply keyword filtering
        text_to_check = (raw_post['title'] + ' ' + raw_post['body_md']).lower()
        matched_keywords = [
            kw for kw in contract_keywords if kw in text_to_check]

        # Filter logic - needs at least 1 keyword match
        passed_filter = len(matched_keywords) > 0
        confidence = min(0.9, 0.3 + (len(matched_keywords) * 0.1))

        filtered_post = {
            'raw_post_id': raw_post['_id'],
            'passed_keyword_filter': passed_filter,
            'filter_confidence': confidence,
            'matched_keywords': matched_keywords,
            'filter_timestamp': datetime.utcnow(),
            'filter_version': '1.0.0',
            'llm_screened': False
        }

        await db.insert_one('filtered_posts', filtered_post)
        filtered_count += 1

        if passed_filter:
            passed_count += 1
            logger.info(
                f"✅ Post {i+1}: PASSED - {len(matched_keywords)} keywords matched")
        else:
            logger.info(f"❌ Post {i+1}: FILTERED OUT - No keyword matches")

    logger.info(
        f"🔍 Keyword filtering complete: {passed_count}/{filtered_count} posts passed")


async def main():
    """Run the end-to-end pipeline test."""
    logger.info("🚀 Starting End-to-End Pipeline Test with Mock Data")

    # Initialize mock database
    db = MockDatabaseManager()
    await db.connect()

    try:
        # Step 1: Create mock raw data
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Creating Mock Raw Data")
        logger.info("="*60)

        raw_posts = create_mock_raw_posts()
        logger.info(f"📝 Created {len(raw_posts)} mock raw posts")

        # Step 2: Keyword filtering
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Keyword Filtering")
        logger.info("="*60)

        await simulate_keyword_filtering(db, raw_posts)

        # Step 3: LLM Screening
        logger.info("\n" + "="*60)
        logger.info("STEP 3: LLM Screening")
        logger.info("="*60)

        screener = MockLLMScreener(db)
        screening_stats = await screener.screen_batch(batch_size=10)

        logger.info(f"🤖 LLM Screening Results:")
        logger.info(f"   - Processed: {screening_stats['processed']}")
        logger.info(
            f"   - Positive (Y): {screening_stats['positive_decisions']}")
        logger.info(
            f"   - Negative (N): {screening_stats['negative_decisions']}")
        logger.info(f"   - Borderline: {screening_stats['borderline_cases']}")
        logger.info(
            f"   - High Confidence: {screening_stats['high_confidence']}")
        logger.info(f"   - Errors: {screening_stats['errors']}")

        # Step 4: Results Summary
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Pipeline Results Summary")
        logger.info("="*60)

        stats = db.get_stats()
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   - Raw Posts: {stats['raw_posts']}")
        logger.info(f"   - Filtered Posts: {stats['filtered_posts']}")
        logger.info(f"   - Passed Filter: {stats['passed_filter']}")
        logger.info(f"   - LLM Screened: {stats['screened']}")
        logger.info(
            f"   - Screening Results: {stats['llm_screening_results']}")

        # Show some example screening results
        logger.info("\n📋 Example Screening Results:")
        for i, result in enumerate(db.collections['llm_screening_results'][:3]):
            logger.info(
                f"   Result {i+1}: {result['decision']} (confidence: {result['confidence']:.2f})")
            logger.info(f"      Rationale: {result['rationale']}")

        # Success metrics
        success_rate = (stats['screened'] /
                        max(stats['passed_filter'], 1)) * 100
        logger.info(f"\n✅ Pipeline Success Rate: {success_rate:.1f}%")

        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Pipeline is working correctly!")
        elif success_rate >= 75:
            logger.info(
                "✅ GOOD: Pipeline is mostly working, minor issues possible")
        else:
            logger.info(
                "⚠️ NEEDS ATTENTION: Pipeline has issues that need fixing")

        logger.info("\n💡 Next Steps for Real Deployment:")
        logger.info("   1. Set up MongoDB Atlas connection")
        logger.info("   2. Configure real API keys (OpenAI, DeepSeek, etc.)")
        logger.info("   3. Run data acquisition from GitHub/Stack Overflow")
        logger.info("   4. Test with small batches first")
        logger.info("   5. Scale up gradually")

    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
