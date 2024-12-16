from unittest.mock import patch

import pytest

from speedy_openai import OpenAIClient


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def async_client(api_key):
    return OpenAIClient(api_key)


@pytest.fixture
def sample_request():
    return {
        "custom_id": "test-1",
        "method": "POST",  # Added method field
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]},
    }


@pytest.fixture
def sample_response():
    return {
        "custom_id": "test-1",
        "response": {"choices": [{"message": {"content": "Hello! How can I help you today?", "role": "assistant"}}]},
    }


@pytest.mark.asyncio
async def test_init_async_client(api_key):
    client = OpenAIClient(api_key, max_requests_per_min=100, max_tokens_per_min=1000, max_concurrent_requests=10)

    assert client.config.api_key == api_key
    assert client.config.max_requests_per_min == 100
    assert client.config.max_tokens_per_min == 1000
    assert client.config.max_concurrent_requests == 10
    assert client.headers["Authorization"] == f"Bearer {api_key}"


def test_count_tokens():
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    model = "gpt-3.5-turbo"

    token_count = OpenAIClient.count_tokens(messages, model)
    assert isinstance(token_count, int)
    assert token_count > 0


@pytest.mark.asyncio
async def test_process_request_success(async_client, sample_request, sample_response):
    with patch.object(async_client, "_make_request") as mock_make_request:
        mock_make_request.return_value = sample_response

        result = await async_client.process_request(sample_request)

        assert result == sample_response
        mock_make_request.assert_called_once()


@pytest.mark.asyncio
async def test_process_request_invalid_response(async_client, sample_request):
    invalid_response = {"custom_id": "test-1", "response": {}}  # Missing 'choices' key

    with patch.object(async_client, "_make_request") as mock_make_request:
        mock_make_request.return_value = invalid_response

        with pytest.raises(ValueError):
            await async_client.process_request(sample_request)


@pytest.mark.asyncio
async def test_process_batch(async_client, sample_request, sample_response):
    requests = [sample_request] * 3

    with patch.object(async_client, "process_request") as mock_process_request:
        mock_process_request.return_value = sample_response

        results = await async_client.process_batch(requests)

        assert len(results) == 3
        assert all(result == sample_response for result in results)
        assert mock_process_request.call_count == 3
