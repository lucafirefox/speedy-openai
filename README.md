# Speedy OpenAI

[![PyPI version](https://badge.fury.io/py/speedy-openai.svg)](https://badge.fury.io/py/speedy-openai)
[![Python](https://img.shields.io/pypi/pyversions/speedy-openai.svg)](https://pypi.org/project/speedy-openai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A high-performance, asynchronous Python client for the OpenAI API with built-in rate limiting and concurrency control.

## Features

- ⚡ Asynchronous request handling for optimal performance
- 🔄 Built-in rate limiting for both requests and tokens
- 🎛️ Configurable concurrency control
- 🔁 Automatic retry mechanism with backoff
- 📊 Progress tracking for batch requests
- 🎯 Token counting and management
- 📝 Comprehensive logging

## Installation

```bash
pip install speedy-openai
```

## Quick Start

```python
import asyncio
from speedy_openai import OpenAIClient

async def main():
    # Initialize the client
    client = OpenAIClient(
        api_key="your-api-key",
        max_requests_per_min=5000,  # Optional: default 5000
        max_tokens_per_min=15000000,  # Optional: default 15M
        max_concurrent_requests=250  # Optional: default 250
    )

    # Single request
    request = {
        "custom_id": "req1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
    }
    
    response = await client.process_request(request)

    # Batch requests
    requests = [request, request]  # List of requests
    responses = await client.process_batch(requests)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | Required | Your OpenAI API key |
| `max_requests_per_min` | 5000 | Maximum API requests per minute |
| `max_tokens_per_min` | 15000000 | Maximum tokens per minute |
| `max_concurrent_requests` | 250 | Maximum concurrent requests |
| `max_retries` | 5 | Maximum retry attempts |
| `max_sleep_time` | 60 | Maximum sleep time between retries (seconds) |

## Features in Detail

### Rate Limiting

The client includes a sophisticated rate limiter that manages both request frequency and token usage:
- Automatically tracks remaining requests and tokens
- Updates limits from API response headers
- Implements waiting periods when limits are reached
- Supports dynamic limit adjustments

### Concurrency Control

- Manages concurrent requests using asyncio semaphores
- Prevents overwhelming the API with too many simultaneous requests
- Configurable maximum concurrent requests

### Retry Mechanism

Built-in retry logic for handling common API errors:
- Automatic retries with fixed wait times
- Configurable maximum retry attempts
- Specific exception handling for API-related errors

### Progress Tracking

Batch requests include:
- Progress bar visualization using tqdm
- Processing time logging
- Detailed success/failure reporting

## Error Handling

The client includes comprehensive error handling:
- API response validation
- Rate limit handling
- Network error recovery
- Invalid request detection

## Requirements

- Python 3.7+
- aiohttp
- tiktoken
- tenacity
- tqdm
- loguru
- pydantic

## Common Use Cases

### 1. Chat Completion with GPT-4

```python
import asyncio
from speedy_openai import OpenAIClient

async def chat_with_gpt4():
    client = OpenAIClient(api_key="your-api-key")
    
    request = {
        "custom_id": "chat-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            "temperature": 0.7
        }
    }
    
    response = await client.process_request(request)
    print(response["response"]["choices"][0]["message"]["content"])

asyncio.run(chat_with_gpt4())
```

### 2. Batch Processing Multiple Conversations

```python
async def process_multiple_conversations():
    client = OpenAIClient(api_key="your-api-key")
    
    conversations = [
        {"role": "user", "content": "What is AI?"},
        {"role": "user", "content": "Explain machine learning."},
        {"role": "user", "content": "What is deep learning?"}
    ]
    
    requests = [
        {
            "custom_id": f"batch-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [conv],
                "temperature": 0.7
            }
        }
        for i, conv in enumerate(conversations)
    ]
    
    responses = await client.process_batch(requests)
    return responses
```

## Testing

The project uses pytest for testing. To run the tests:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speedy-openai.git
cd speedy-openai
```

2. Install development dependencies:
```bash
poetry install
```

3. Run tests:
```bash
poetry run pytest
```

### Test Structure

The test suite includes:

- Unit tests for core functionality
- Integration tests for API interactions
- Rate limiting tests
- Concurrency tests
- Error handling tests

### Running Specific Tests

Run specific test categories:
```bash
# Run only rate limiter tests
poetry run pytest tests/test_rate_limiter.py

# Run only client tests
poetry run pytest tests/test_client.py

# Run with verbose output
poetry run pytest -v

# Run with coverage report
poetry run pytest --cov=speedy_openai
```

### Writing Tests

When contributing new features, please ensure:
- All new features have corresponding tests
- Test coverage remains above 80%
- Tests are properly documented
- Both success and failure cases are covered

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please create an issue in the GitHub repository.

