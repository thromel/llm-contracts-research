#!/usr/bin/env python3

"""Test script for parsing logic"""

from pipeline.common.database import MongoDBManager
from pipeline.llm_screening.borderline_screener import BorderlineScreener
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_parsing():
    """Test different response formats"""

    # Create a mock screener just for parsing
    screener = BorderlineScreener(
        api_key="test",
        db_manager=None,
        rate_limit_delay=0,
        max_concurrent_requests=1
    )

    # Test cases
    test_cases = [
        # Good text format
        """DECISION: Y
CONFIDENCE: 0.8
RATIONALE: Clear contract violation with max_tokens error""",

        # Partial response (the problematic case)
        '"contains_violation"',

        # JSON format
        '{"contains_violation": true, "confidence": "high", "verification_notes": "Test violation"}',

        # Mixed content
        """Some text before
DECISION: N
CONFIDENCE: 0.6
RATIONALE: No contract violation found
Some text after""",

        # Incomplete response
        "Error in processing request"
    ]

    print("Testing parsing logic:")
    print("=" * 50)

    for i, test_response in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_response[:50]}...")
        try:
            result = screener._parse_screening_response(test_response)
            print(f"  Decision: {result.decision}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Rationale: {result.rationale[:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 50)
    print("Parsing test completed!")


if __name__ == "__main__":
    test_parsing()
