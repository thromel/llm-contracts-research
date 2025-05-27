#!/usr/bin/env python3
"""Quick test to check Stack Overflow API response format."""

import asyncio
import httpx
import json
import time
from datetime import datetime, timedelta


async def test_stackoverflow_api():
    """Test what fields Stack Overflow API returns."""

    # API parameters
    base_url = "https://api.stackexchange.com/2.3"

    params = {
        'site': 'stackoverflow',
        'pagesize': 1,  # Just get one question
        'order': 'desc',
        'sort': 'activity',
        'filter': '!9_bDDxJY5'  # Include body content
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            print("Testing Stack Overflow API with body filter...")
            print("Waiting 60 seconds to avoid rate limits...")
            await asyncio.sleep(60)  # Wait to avoid rate limits

            url = f"{base_url}/questions"
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            questions = data.get('items', [])

            if questions:
                question = questions[0]
                print("Available fields in question:")
                print(json.dumps(list(question.keys()), indent=2))
                print("\nTitle:", question.get('title', 'NO TITLE'))
                print("Body present:", 'body' in question)
                if 'body' in question:
                    body = question.get('body', '')
                    print(f"Body length: {len(body)}")
                    print(f"Body preview: {body[:200]}...")
                else:
                    print("NO BODY FIELD FOUND!")

            else:
                print("No questions returned!")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stackoverflow_api())
