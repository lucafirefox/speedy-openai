from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Configs(BaseModel):
    """Configuration settings for API requests.

    Attributes:
        api_key (str): Authentication key for API access.
        max_requests_per_min (int): Maximum number of API requests allowed per minute. Defaults to 5,000.
        max_tokens_per_min (int): Maximum number of tokens allowed per minute. Defaults to 15,000,000.
        max_concurrent_requests (int): Maximum number of simultaneous API requests allowed. Defaults to 250.
        max_retries (int): Maximum number of retry attempts for failed requests. Defaults to 5.
        max_sleep_time (int): Maximum sleep duration between retries in seconds. Defaults to 60.
    """

    model_config = {"arbitrary_types_allowed": True}
    api_key: str
    max_requests_per_min: int = Field(default=5_000)
    max_tokens_per_min: int = Field(default=15_000_000)
    max_concurrent_requests: int = Field(default=250)
    max_retries: int = Field(default=5)
    max_sleep_time: int = Field(default=60)


class Request(BaseModel):
    """Represents an API request structure.

    Attributes:
        custom_id (str): Unique identifier for the request.
        method (str): HTTP method for the request (e.g., GET, POST, PUT, DELETE).
        url (str): Target URL for the API request.
        body (dict[str, Any]): Request body containing key-value pairs of request parameters.
    """

    model_config = {"arbitrary_types_allowed": True}
    custom_id: str
    method: str
    url: str
    body: dict[str, Any]
