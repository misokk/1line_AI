"""Actual Google SDK serialization over MockTransport. No external API calls."""

import json
from unittest.mock import patch

import httpx
import pytest
from google import genai
from google.genai import types

from api.config import Settings
from api.gemini import SDKClient, compatible_schema
from src.errors import (
    InvalidResponseError,
    ModelAccessError,
    ProviderBadGatewayError,
    ProviderError,
    ProviderRequestError,
    ProviderServiceUnavailableError,
    RateLimitError,
    RequestTimeoutError,
)
from src.models import Keywords


def envelope(data, status="STOP"):
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": json.dumps(data)}]},
                "finishReason": status,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 10,
            "totalTokenCount": 20,
        },
    }


def adapter(handler):
    client = genai.Client(
        api_key="fake-private-key",
        vertexai=False,
        http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )
    with patch("api.gemini.sdk_client", return_value=client):
        return SDKClient("gemini", "fake-private-key", Settings(live_enabled=True))


def test_official_sdk_payload_no_unsupported_parameters():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path.endswith("/models/gemini-3.8-flash:generateContent")
        config = body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["additionalProperties"] is False
        assert not set(config).intersection({"temperature", "topP", "topK", "candidateCount"})
        assert config["maxOutputTokens"] == 8192
        return httpx.Response(200, json=envelope({"keywords": ["grammar"]}))

    client = adapter(handler)
    try:
        assert client.generate_json(
            model="gemini-3.8-flash", prompt="source", schema=Keywords
        ).keywords == ["grammar"]
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    "code,kind",
    [
        (400, ProviderRequestError),
        (401, ModelAccessError),
        (403, ModelAccessError),
        (404, ModelAccessError),
        (429, RateLimitError),
        (502, ProviderBadGatewayError),
        (503, ProviderServiceUnavailableError),
    ],
)
def test_errors_safe_and_never_retried(code, kind):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            code,
            headers={"Retry-After": "12"},
            json={
                "error": {
                    "code": code,
                    "message": "PRIVATE_SOURCE fake-private-key",
                    "status": "UNKNOWN",
                }
            },
        )

    client = adapter(handler)
    try:
        with pytest.raises(kind) as error:
            client.generate_json(model="gemini-3.8-flash", prompt="source", schema=Keywords)
        assert len(calls) == 1 and "PRIVATE_SOURCE" not in str(error.value)
        assert "fake-private-key" not in str(error.value)
        if code == 429:
            assert error.value.retry_after == 12
    finally:
        client.close()


@pytest.mark.parametrize(
    "data,status",
    [
        ({"keywords": ["x"]}, "MAX_TOKENS"),
        ({"keywords": [123]}, "STOP"),
        ({"keywords": ["x"]}, "SAFETY"),
    ],
)
def test_incomplete_or_invalid_output_not_repaired(data, status):
    client = adapter(lambda req: httpx.Response(200, json=envelope(data, status)))
    try:
        with pytest.raises(InvalidResponseError):
            client.generate_json(model="gemini-3.8-flash", prompt="source", schema=Keywords)
    finally:
        client.close()


def test_timeout_without_retries_or_private_message():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("PRIVATE_SOURCE", request=request)

    client = adapter(handler)
    try:
        with pytest.raises(RequestTimeoutError) as error:
            client.generate_json(model="gemini-3.8-flash", prompt="source", schema=Keywords)
        assert len(calls) == 1 and "PRIVATE_SOURCE" not in str(error.value)
    finally:
        client.close()


def test_http_500_distinguished_without_private_provider_message():
    client = adapter(
        lambda request: httpx.Response(
            500, json={"error": {"code": 500, "message": "PRIVATE_SOURCE", "status": "INTERNAL"}}
        )
    )
    try:
        with pytest.raises(ProviderError, match="HTTP 500") as error:
            client.generate_json(model="gemini-3.8-flash", prompt="source", schema=Keywords)
        assert "PRIVATE_SOURCE" not in str(error.value)
    finally:
        client.close()


@pytest.mark.parametrize(
    "exception,label",
    [
        (httpx.ConnectError("PRIVATE_SOURCE secret-url"), "network connection"),
        (httpx.RemoteProtocolError("PRIVATE_SOURCE secret-url"), "network protocol"),
        (ValueError("PRIVATE_SOURCE secret-url"), "ValueError"),
    ],
)
def test_non_http_error_category_without_raw_exception(exception, label):
    from api.gemini import provider_error

    with pytest.raises(ProviderError, match=label) as error:
        provider_error(exception)
    assert "PRIVATE_SOURCE" not in str(error.value) and "secret-url" not in str(error.value)


def test_schema_compatibility_does_not_relax_final_rules():
    from typing import Literal

    from pydantic import Field

    from src.models import Model

    class Constrained(Model):
        kind: Literal["fixed"]
        label: str = Field(min_length=2, max_length=5, pattern="^[a-z]+$")
        items: list[int] = Field(min_length=1, max_length=2)

    original = Constrained.model_json_schema()
    wire = compatible_schema(original)
    assert "const" in original["properties"]["kind"]
    assert wire["properties"]["kind"]["enum"] == ["fixed"]
    assert "pattern" not in wire["properties"]["label"]
    assert wire["properties"]["items"]["maxItems"] == 2
    assert wire["additionalProperties"] is False
    for wrong in ("TOOLONG", "1", "x"):
        client = adapter(
            lambda req, wrong=wrong: httpx.Response(
                200, json=envelope({"kind": "fixed", "label": wrong, "items": [1]})
            )
        )
        try:
            with pytest.raises(InvalidResponseError):
                client.generate_json(model="gemini-3.8-flash", prompt="source", schema=Constrained)
        finally:
            client.close()
