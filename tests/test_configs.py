import pytest
from pydantic import ValidationError

from speedy_openai.configs import Request
from speedy_openai import Configs


def test_configs_with_valid_data():
    """Test Configs initialization with valid data."""
    config = Configs(
        api_key="test_key_123",
        max_requests_per_min=6000,
        max_tokens_per_min=16000000,
        max_concurrent_requests=300,
        max_retries=3,
        max_sleep_time=30,
    )

    assert config.api_key == "test_key_123"
    assert config.max_requests_per_min == 6000
    assert config.max_tokens_per_min == 16000000
    assert config.max_concurrent_requests == 300
    assert config.max_retries == 3
    assert config.max_sleep_time == 30


def test_configs_with_default_values():
    """Test Configs initialization with only required fields."""
    config = Configs(api_key="test_key_123")

    assert config.api_key == "test_key_123"
    assert config.max_requests_per_min == 5000
    assert config.max_tokens_per_min == 15000000
    assert config.max_concurrent_requests == 250
    assert config.max_retries == 5
    assert config.max_sleep_time == 60


def test_configs_missing_required_field():
    """Test Configs initialization without required api_key."""
    with pytest.raises(ValidationError):
        Configs()


def test_configs_invalid_types():
    """Test Configs initialization with invalid data types."""
    with pytest.raises(ValidationError):
        Configs(
            api_key=123,  # should be string
            max_requests_per_min="invalid",  # should be int
            max_tokens_per_min="invalid",  # should be int
            max_concurrent_requests="invalid",  # should be int
            max_retries="invalid",  # should be int
            max_sleep_time="invalid",  # should be int
        )


def test_request_with_valid_data():
    """Test Request initialization with valid data."""
    request = Request(custom_id="req_123", method="POST", url="https://api.example.com/endpoint", body={"key": "value"})

    assert request.custom_id == "req_123"
    assert request.method == "POST"
    assert request.url == "https://api.example.com/endpoint"
    assert request.body == {"key": "value"}


def test_request_with_empty_body():
    """Test Request initialization with empty body."""
    request = Request(custom_id="req_123", method="GET", url="https://api.example.com/endpoint", body={})

    assert request.custom_id == "req_123"
    assert request.method == "GET"
    assert request.url == "https://api.example.com/endpoint"
    assert request.body == {}


def test_request_missing_required_fields():
    """Test Request initialization with missing required fields."""
    with pytest.raises(ValidationError):
        Request()


def test_request_invalid_types():
    """Test Request initialization with invalid data types."""
    with pytest.raises(ValidationError):
        Request(
            custom_id=123,  # should be string
            method=123,  # should be string
            url=123,  # should be string
            body="invalid",  # should be dict
        )


def test_request_complex_body():
    """Test Request initialization with complex nested body structure."""
    complex_body = {"string": "value", "number": 123, "bool": True, "list": [1, 2, 3], "nested": {"key": "value"}}

    request = Request(custom_id="req_123", method="POST", url="https://api.example.com/endpoint", body=complex_body)

    assert request.body == complex_body


def test_request_invalid_body_structure():
    """Test Request initialization with invalid body structure."""
    with pytest.raises(ValidationError):
        Request(
            custom_id="req_123",
            method="POST",
            url="https://api.example.com/endpoint",
            body=["invalid", "body", "structure"],  # should be dict
        )
