import asyncio
import time
from unittest.mock import patch

import pytest

from speedy_openai.rate_limiter import RateLimiter


class Configs:
    max_sleep_time = 60


@pytest.fixture(autouse=True)
def mock_configs():
    with patch("speedy_openai.Configs", Configs):  # Fix the patch path
        yield


@pytest.fixture
def rate_limiter():
    return RateLimiter(max_requests=10, max_tokens=100, max_sleep_time=Configs.max_sleep_time)


@pytest.mark.asyncio
async def test_basic_initialization():
    """Test basic initialization of RateLimiter."""
    limiter = RateLimiter(max_requests=5, max_tokens=50, max_sleep_time=Configs.max_sleep_time)
    assert limiter.max_requests == 5
    assert limiter.max_tokens == 50
    assert limiter.remaining_requests == 5
    assert limiter.remaining_tokens == 50


@pytest.mark.asyncio
async def test_wait_for_availability_basic(rate_limiter):
    """Test basic wait_for_availability functionality."""
    await rate_limiter.wait_for_availability(required_tokens=1)
    assert rate_limiter.remaining_requests == 9
    assert rate_limiter.remaining_tokens == 99


@pytest.mark.asyncio
async def test_wait_for_availability_no_tokens():
    """Test wait_for_availability with no token limit."""
    limiter = RateLimiter(max_requests=5, max_sleep_time=Configs.max_sleep_time)
    await limiter.wait_for_availability()
    assert limiter.remaining_requests == 4
    assert limiter.remaining_tokens is None


@pytest.mark.asyncio
async def test_update_from_headers(rate_limiter):
    """Test updating limits from response headers."""
    headers = {
        "x-ratelimit-remaining-requests": "5",
        "x-ratelimit-remaining-tokens": "50",
        "x-ratelimit-reset-requests": "30s",
        "x-ratelimit-reset-tokens": "45s",
    }
    rate_limiter.update_from_headers(headers)
    assert rate_limiter.remaining_requests == 5
    assert rate_limiter.remaining_tokens == 50


@pytest.mark.asyncio
async def test_parse_reset_time():
    """Test parsing different time formats."""
    limiter = RateLimiter(max_sleep_time=Configs.max_sleep_time)
    assert limiter._parse_reset_time("30s") == 30
    assert limiter._parse_reset_time("2m") == 120
    assert limiter._parse_reset_time("1h") == 3600
    assert limiter._parse_reset_time("500ms") == 0.5
    assert limiter._parse_reset_time("1m30s") == 90


@pytest.mark.asyncio
async def test_update_limits(rate_limiter):
    """Test automatic limit updates."""
    # Set limits to 0 and reset time to past
    rate_limiter.remaining_requests = 0
    rate_limiter.remaining_tokens = 0
    rate_limiter.reset_time_requests = time.monotonic() - 1
    rate_limiter.reset_time_tokens = time.monotonic() - 1

    rate_limiter.update_limits()
    assert rate_limiter.remaining_requests == 10
    assert rate_limiter.remaining_tokens == 100


@pytest.mark.asyncio
async def test_wait_for_availability_with_sleep():
    """Test waiting behavior when limits are exceeded."""
    with patch("speedy_openai.Configs", Configs):
        limiter = RateLimiter(max_requests=1, max_tokens=1, max_sleep_time=Configs.max_sleep_time)

        # Use up all requests and tokens
        await limiter.wait_for_availability(required_tokens=1)

        # Set remaining resources to 0
        limiter.remaining_requests = 0
        limiter.remaining_tokens = 0

        # Set reset times to a specific future time
        future_time = time.monotonic() + 0.1
        limiter.reset_time_requests = future_time
        limiter.reset_time_tokens = future_time

        # Mock sleep to avoid actual waiting
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await limiter.wait_for_availability(required_tokens=1)
            # Check if sleep was called at least once
            assert mock_sleep.call_count > 0


@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases and invalid inputs."""
    with patch("speedy_openai.Configs", Configs):
        # Test with zero limits
        limiter = RateLimiter(max_requests=1, max_tokens=1, max_sleep_time=Configs.max_sleep_time)
        limiter.remaining_requests = 0
        limiter.remaining_tokens = 0
        limiter.reset_time_requests = time.monotonic() + 0.1
        limiter.reset_time_tokens = time.monotonic() + 0.1

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await limiter.wait_for_availability(required_tokens=1)
            assert mock_sleep.called

        # Test with invalid header values
        limiter = RateLimiter(max_requests=5, max_tokens=50, max_sleep_time=Configs.max_sleep_time)
        previous_requests = limiter.remaining_requests
        previous_tokens = limiter.remaining_tokens

        # Test with missing headers (shouldn't raise exception)
        limiter.update_from_headers({})
        assert limiter.remaining_requests == previous_requests
        assert limiter.remaining_tokens == previous_tokens

        # Test with partial valid headers
        headers = {
            "x-ratelimit-remaining-requests": "3",
            "x-ratelimit-reset-requests": "30s",
        }
        limiter.update_from_headers(headers)
        assert limiter.remaining_requests == 3


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test behavior with concurrent requests."""
    with patch("speedy_openai.Configs", Configs):
        # Initialize with enough capacity for our test
        limiter = RateLimiter(max_requests=5, max_tokens=50, max_sleep_time=Configs.max_sleep_time)
        request_count = 3
        tokens_per_request = 10

        async def make_request():
            try:
                await limiter.wait_for_availability(required_tokens=tokens_per_request)
                # Simulate some work
                await asyncio.sleep(0.01)
                return True
            except Exception as e:
                print(f"Request failed with error: {e}")
                return False

        # Execute requests concurrently
        with patch("asyncio.sleep", side_effect=lambda x: None):  # Mock sleep but keep brief delays
            # Create and gather tasks
            tasks = [make_request() for _ in range(request_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify results
            successful_requests = [r for r in results if r is True]
            assert (
                len(successful_requests) == request_count
            ), f"Expected {request_count} successful requests, got {len(successful_requests)}"

            # Verify rate limiter state
            expected_remaining_requests = limiter.max_requests - request_count
            expected_remaining_tokens = limiter.max_tokens - (request_count * tokens_per_request)

            assert (
                limiter.remaining_requests >= 0
            ), f"Remaining requests should not be negative: {limiter.remaining_requests}"
            assert limiter.remaining_tokens >= 0, f"Remaining tokens should not be negative: {limiter.remaining_tokens}"

            assert limiter.remaining_requests <= expected_remaining_requests, "Too many remaining requests"
            assert limiter.remaining_tokens <= expected_remaining_tokens, "Too many remaining tokens"
