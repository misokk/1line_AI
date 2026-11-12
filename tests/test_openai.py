"""Official SDK over MockTransport: no paid API requests."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from api.config import Settings
from api.openai import SDKClient
from src.errors import (
    InvalidResponseError,
    ModelAccessError,
    ProviderBadGatewayError,
    ProviderRequestError,
    ProviderServiceUnavailableError,
    RateLimitError,
    RequestTimeoutError,
)
from src.models import Keywords


def envelope(data, status="completed"):
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-5.4-mini",
        "output": [
            {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(data), "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    }


def adapter(handler):
    def factory(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    with patch("api.openai.importlib.import_module", return_value=SimpleNamespace(OpenAI=factory)):
        return SDKClient("openai", "fake-private-key", Settings(live_enabled=True))


def test_official_sdk_structured_outputs_and_store_false():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/v1/responses"
        assert body["model"] == "gpt-5.4-mini" and body["store"] is False
        fmt = body["text"]["format"]
        assert fmt["type"] == "json_schema" and fmt["strict"] is True
        assert fmt["schema"]["additionalProperties"] is False
        return httpx.Response(200, json=envelope({"keywords": ["grammar"]}))

    client = adapter(handler)
    try:
        assert client.generate_json(
            model="gpt-5.4-mini", prompt="source", schema=Keywords
        ).keywords == ["grammar"]
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    "code,kind",
    [
        (400, ProviderRequestError),
        (429, RateLimitError),
        (401, ModelAccessError),
        (403, ModelAccessError),
        (404, ModelAccessError),
        (502, ProviderBadGatewayError),
        (503, ProviderServiceUnavailableError),
    ],
)
def test_official_sdk_errors_no_retry_no_provider_content(code, kind):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            code, json={"error": {"message": "PRIVATE_SOURCE", "type": "invalid_request_error"}}
        )

    client = adapter(handler)
    try:
        with pytest.raises(kind) as error:
            client.generate_json(model="gpt-5.4-mini", prompt="source", schema=Keywords)
        assert len(calls) == 1 and "PRIVATE_SOURCE" not in str(error.value)
    finally:
        client.close()


@pytest.mark.parametrize(
    "data,status", [({"keywords": ["x"]}, "incomplete"), ({"keywords": [123]}, "completed")]
)
def test_invalid_or_incomplete_sdk_output_not_finalized(data, status):
    client = adapter(lambda request: httpx.Response(200, json=envelope(data, status)))
    try:
        with pytest.raises(InvalidResponseError):
            client.generate_json(model="gpt-5.4-mini", prompt="source", schema=Keywords)
    finally:
        client.close()


def test_retry_after_returned_without_retry():
    client = adapter(
        lambda req: httpx.Response(
            429,
            headers={"Retry-After": "12"},
            json={"error": {"message": "private", "type": "rate_limit_error"}},
        )
    )
    try:
        with pytest.raises(RateLimitError) as error:
            client.generate_json(model="gpt-5.4-mini", prompt="source", schema=Keywords)
        assert error.value.retry_after == 12
    finally:
        client.close()


def test_sdk_timeout_does_not_retry_or_expose_exception_text():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("PRIVATE_SOURCE", request=request)

    client = adapter(handler)
    try:
        with pytest.raises(RequestTimeoutError) as error:
            client.generate_json(model="gpt-5.4-mini", prompt="source", schema=Keywords)
        assert len(calls) == 1 and "PRIVATE_SOURCE" not in str(error.value)
    finally:
        client.close()


def test_sdk_refusal_never_treated_as_success():
    data = envelope({"keywords": ["word"]})
    data["output"][0]["content"] = [{"type": "refusal", "refusal": "PRIVATE_SOURCE"}]
    client = adapter(lambda req: httpx.Response(200, json=data))
    try:
        with pytest.raises(InvalidResponseError) as error:
            client.generate_json(model="gpt-5.4-mini", prompt="source", schema=Keywords)
        assert "PRIVATE_SOURCE" not in str(error.value)
    finally:
        client.close()
