"""
A flexible rate limiter implementation for managing API request rates and token consumption.

This class provides mechanisms to control both request frequency and token usage in API calls,
particularly useful for services with dual-limit systems. It supports dynamic limit updates
from API response headers and handles waiting periods when limits are reached.

Process Flow:
1. Initialize with maximum requests and/or token limits
2. Before making an API call:
   - Check current limits via wait_for_availability()
   - If limits are exceeded, sleep until reset
3. After API calls:
   - Update limits from response headers if available
   - Track remaining requests and tokens
4. Auto-reset limits when reset time is reached
"""

from __future__ import annotations

import asyncio
import re
import time


class RateLimiter:
    """A rate limiter implementation for managing API request and token limits.

    Attributes:
        REGEX_TIME (Pattern): Regular expression pattern for parsing time strings.
        max_requests (int | None): Maximum number of requests allowed in the time window.
        max_tokens (int | None): Maximum number of tokens allowed in the time window.
        remaining_requests (int): Number of requests remaining in the current window.
        remaining_tokens (int): Number of tokens remaining in the current window.
        reset_time_requests (float): Timestamp when request limit resets.
        reset_time_tokens (float): Timestamp when token limit resets.

    Args:
        max_requests (int | None, optional): Maximum number of requests. Defaults to None.
        max_tokens (int | None, optional): Maximum number of tokens. Defaults to None.
    """

    REGEX_TIME = re.compile(r"(?P<value>\d+)(?P<unit>[smhms]+)")

    def __init__(self, max_sleep_time: int, max_requests: int | None = None, max_tokens: int | None = None):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.remaining_requests = max_requests
        self.remaining_tokens = max_tokens
        self.max_sleep_time = max_sleep_time
        self.reset_time_requests = time.monotonic() + 60
        self.reset_time_tokens = time.monotonic() + 60

    async def wait_for_availability(self, required_tokens: int = 0):
        """Wait until rate limits allow for the requested operation."""
        while self.remaining_requests <= 0 or (self.max_tokens is not None and self.remaining_tokens < required_tokens):
            self.update_limits()
            sleep_time = self._get_seconds_to_sleep()
            await asyncio.sleep(sleep_time)

        self.remaining_requests -= 1
        if self.max_tokens is not None:
            self.remaining_tokens -= required_tokens

    def update_limits(self):
        """Update rate limits based on current time."""
        current_time = time.monotonic()
        if current_time >= self.reset_time_requests:
            self.remaining_requests = self.max_requests
            self.reset_time_requests = current_time + 60

        if self.max_tokens is not None and current_time >= self.reset_time_tokens:
            self.remaining_tokens = self.max_tokens
            self.reset_time_tokens = current_time + 60

    def update_from_headers(self, headers: dict[str, str]):
        """Update rate limit information from response headers."""
        self.remaining_requests = int(headers.get("x-ratelimit-remaining-requests", self.remaining_requests))
        if self.max_tokens is not None:
            self.remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens", self.remaining_tokens))
            reset_tokens_seconds = self._parse_reset_time(headers.get("x-ratelimit-reset-tokens", "60s"))
            self.reset_time_tokens = time.monotonic() + reset_tokens_seconds

        reset_requests_seconds = self._parse_reset_time(headers.get("x-ratelimit-reset-requests", "60s"))
        self.reset_time_requests = time.monotonic() + reset_requests_seconds

    def _get_seconds_to_sleep(self) -> float:
        """Calculate the number of seconds to sleep before next request."""
        if self.max_tokens is None:
            sleep_time = self.reset_time_requests - time.monotonic()
        else:
            sleep_time = min(
                self.reset_time_requests - time.monotonic(),
                self.reset_time_tokens - time.monotonic(),
            )
        return max(min(sleep_time, self.max_sleep_time), 1)

    def _parse_reset_time(self, reset_time_str: str) -> float:
        """Parse a reset time string into seconds."""
        total_seconds = 0
        for match in self.REGEX_TIME.finditer(reset_time_str):
            value = int(match.group("value"))
            unit = match.group("unit")
            if unit == "s":
                total_seconds += value
            elif unit == "m":
                total_seconds += value * 60
            elif unit == "h":
                total_seconds += value * 3600
            elif unit == "ms":
                total_seconds += value / 1000.0
        return total_seconds if total_seconds > 0 else 60
