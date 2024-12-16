from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import tiktoken
from loguru import logger as log
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from .configs import Configs, Request
from .rate_limiter import RateLimiter


class OpenAIClient:
    """
    Asynchronous client for making requests to the OpenAI API with built-in rate limiting and concurrency control.

    Attributes:
        config (Configs): Configuration settings for the client
        headers (dict): HTTP headers for API requests
        rate_limiter (RateLimiter): Rate limiting controller
        semaphore (asyncio.Semaphore): Concurrency control mechanism
        base_url (str): Base URL for the OpenAI API
    """

    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key (str): OpenAI API key
            **kwargs: Additional configuration parameters including:
                - max_requests_per_min (int): Maximum requests per minute
                - max_tokens_per_min (int): Maximum tokens per minute
                - max_concurrent_requests (int): Maximum concurrent requests
                - max_retries (int): Maximum number of retry attempts
                - max_sleep_time (int): Maximum sleep time between retries
        """
        self.config = Configs(**{"api_key": api_key, **self._get_config_params(kwargs)})
        self.headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        self.rate_limiter = RateLimiter(
            self.config.max_sleep_time, self.config.max_requests_per_min, self.config.max_tokens_per_min
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.base_url = "https://api.openai.com"

    @staticmethod
    def _get_config_params(kwargs):
        """Extracts and returns a dictionary of valid configuration parameters from the provided keyword arguments."""
        valid_params = {
            "max_requests_per_min",
            "max_tokens_per_min",
            "max_concurrent_requests",
            "max_retries",
            "max_sleep_time",
        }

        return {key: value for key, value in kwargs.items() if key in valid_params and value is not None}

    @staticmethod
    def count_tokens(messages: list[dict[str, str]], model: str) -> int:
        """Counts the number of tokens required for a given list of messages and model."""
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for value in message.values():
                num_tokens += len(encoding.encode(value))

        num_tokens += 2
        return num_tokens

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(10),
        retry=(retry_if_exception_type((aiohttp.ClientResponseError, aiohttp.ClientError))),
    )
    async def _make_request(self, request: Request, required_tokens: int) -> dict:
        """Make an HTTP POST request to the OpenAI API with retry mechanism."""
        await self.rate_limiter.wait_for_availability(required_tokens)

        url = f"{self.base_url}{request.url}"

        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=request.body) as response:
                    response.raise_for_status()
                    self.rate_limiter.update_from_headers(response.headers)
                    return {"custom_id": request.custom_id, "response": await response.json()}

    async def process_request(self, request_data: dict) -> dict:
        """Process a single API request."""
        request = Request(**request_data)
        messages = request.body.get("messages")
        model = request.body.get("model")
        required_tokens = self.count_tokens(messages, model)

        response = await self._make_request(request, required_tokens)

        if "response" not in response or "choices" not in response["response"]:
            error_msg = f"Invalid response format: {response}"
            raise ValueError(error_msg)

        return response

    async def process_batch(self, requests: list[dict]) -> list[dict]:
        """
        Process multiple API requests concurrently.

        Note:
            Uses tqdm for progress tracking and logs processing time"""
        t0 = time.monotonic()
        log.info(f"Processing {len(requests)} requests")

        with tqdm(total=len(requests), desc="Processing requests") as pbar:
            tasks = []
            for req in requests:
                task = asyncio.create_task(self.process_request(req))
                task.add_done_callback(lambda _: pbar.update(1))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        processing_time = (time.monotonic() - t0) / 60
        log.info(f"Batch processed in {processing_time:.2f} minutes")
        return results
