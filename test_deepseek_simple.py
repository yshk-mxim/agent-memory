#!/usr/bin/env python3
"""Simple DeepSeek generation test."""
import httpx
import json

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Count from 1 to 5. Just output the numbers."}],
        "max_tokens": 100,
        "temperature": 0.0
    },
    timeout=30.0
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"Response length: {len(content)}")
    print(f"Response: {repr(content)}")
else:
    print(f"Error: {response.text}")
