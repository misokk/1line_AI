"""Common transport/deadline diagnostics. All SDK HTTP traffic uses MockTransport."""

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from api.app import create_app
from api.gemini import SDKClient, diagnostic_event, diagnostic_scope, error_diagnostics
from src.errors import InvalidResponseError, RequestTimeoutError
from src.models import ExplanationRequest, Keywords
from tests.smoke_gemini import run
from tests.test_gemini import envelope

SERVER_KEY = "diagnostic-server-secret"
PROVIDER_KEY = "diagnostic-provider-secret"


def test_400_diagnostics_save_only_fixed_hints_not_provider_text():
    exc = errors.ClientError(400, {"error": {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": 'Invalid JSON payload. Unknown name "additionalProperties" in '
        'generationConfig.responseJsonSchema: PRIVATE_SOURCE ' + PROVIDER_KEY,
    }})
    result = error_diagnostics(exc)
    assert result["request_error_hints"] == ["unknown_field", "invalid_json_payload"]
    assert set(result["mentioned_request_fields"]) == {
        "additionalProperties", "generationConfig", "responseJsonSchema",
    }
    assert "PRIVATE_SOURCE" not in json.dumps(result)
    assert PROVIDER_KEY not in json.dumps(result)


def test_generic_400_does_not_invent_a_schema_cause():
    exc = errors.ClientError(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument. PRIVATE_SOURCE",
    }})
    result = error_diagnostics(exc)
    assert result["request_error_hints"] == []
    assert result["mentioned_request_fields"] == []


def config(settings, tmp_path, **changes):
    return settings.model_copy(
        update={
            "live_enabled": True,
            "environment": "development",
            "recording_root": tmp_path,
            "server_api_key": SecretStr(SERVER_KEY),
            "gemini_api_key": SecretStr(PROVIDER_KEY),
            "timeout_seconds": 120,
            **changes,
        }
    )


def rows(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / ".cache/diagnostics/gemini.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def factory_for(handler):
    original = genai.Client

    def factory(*, api_key, vertexai, http_options):
        assert vertexai is False
        assert http_options.timeout == 120000
        assert http_options.retry_options.attempts == 1
        assert http_options.base_url == "https://generativelanguage.googleapis.com"
        return original(
            api_key=api_key,
            vertexai=vertexai,
            http_options=http_options.model_copy(
                update={"httpx_client": httpx.Client(transport=httpx.MockTransport(handler))}
            ),
        )

    return factory


def test_initialization_key_model_and_deadline_on_actual_sdk_wire(settings, tmp_path):
    settings = config(settings, tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["x-goog-api-key"] == PROVIDER_KEY
        assert request.headers["X-Server-Timeout"] == "120"
        assert request.extensions["timeout"]["read"] == 120
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path.endswith("/models/fixture-model:generateContent")
        return httpx.Response(200, json=envelope({"keywords": ["grammar"]}))

    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        with diagnostic_scope(settings, "test_wire", settings.default_model_id):
            client = SDKClient("gemini", PROVIDER_KEY, settings)
            try:
                assert client.generate_json(
                    model="fixture-model", prompt="PRIVATE_LEARNING_TEXT", schema=Keywords
                ).keywords == ["grammar"]
            finally:
                client.close()
    assert len(calls) == 1
    events = rows(tmp_path)
    assert len({e["trace_id"] for e in events}) == 1
    assert all(e["model_id"] == settings.default_model_id for e in events)
    assert [e["stage"] for e in events] == [
        "client_initialization_started",
        "client_initialized",
        "sdk_preparation_started",
        "sdk_request_started",
        "sdk_response_received",
        "output_schema_validated",
    ]
    saved = json.dumps(events)
    for private in (PROVIDER_KEY, SERVER_KEY, "PRIVATE_LEARNING_TEXT"):
        assert private not in saved
    assert all(e["elapsed_ms"] >= 0 and e["sdk_version"] for e in events)


@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", "SAFETY", "no_candidates"])
def test_response_diagnostics_distinguish_finish_and_usage_without_raw_content(
    settings, tmp_path, finish,
):
    settings = config(settings, tmp_path)
    payload = envelope({"keywords": ["grammar"]}, status=finish)
    payload["usageMetadata"].update(thoughtsTokenCount=8000, totalTokenCount=8020)
    if finish == "no_candidates":
        payload["candidates"] = []
        payload["promptFeedback"] = {"blockReason": "SAFETY"}
    else:
        payload["candidates"][0]["finishMessage"] = "PRIVATE_SOURCE " + PROVIDER_KEY
        payload["candidates"][0]["content"]["parts"].insert(
            0, {"thought": True, "text": "PRIVATE_SOURCE " + PROVIDER_KEY},
        )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        client = SDKClient("gemini", PROVIDER_KEY, settings)
        try:
            if finish == "STOP":
                assert client.generate_json(
                    model="fixture-model", prompt="PRIVATE_SOURCE", schema=Keywords,
                ).keywords == ["grammar"]
            else:
                with pytest.raises(InvalidResponseError):
                    client.generate_json(
                        model="fixture-model", prompt="PRIVATE_SOURCE", schema=Keywords,
                    )
        finally:
            client.close()
    assert len(calls) == 1
    saved = rows(tmp_path)
    event = next(e for e in saved if e["stage"] == "sdk_response_received")
    assert event["finish_reasons"] == ([] if finish == "no_candidates" else [finish])
    assert event["candidate_count"] == (0 if finish == "no_candidates" else 1)
    assert event["prompt_block_reason"] == ("SAFETY" if finish == "no_candidates" else None)
    assert event["token_usage"]["thoughts_token_count"] == 8000
    assert event["token_usage"]["total_token_count"] == 8020
    assert "PRIVATE_SOURCE" not in json.dumps(saved) and PROVIDER_KEY not in json.dumps(saved)


def test_unrecognized_response_reason_cannot_leak_into_logs(settings, tmp_path):
    settings = config(settings, tmp_path)
    response = types.GenerateContentResponse.model_construct(
        candidates=[types.Candidate.model_construct(finish_reason=PROVIDER_KEY)],
        prompt_feedback=None, usage_metadata=None,
    )
    diagnostic_event(settings, "sdk_response_received", response=response)
    event = rows(tmp_path)[0]
    assert event["finish_reasons"] == ["UNKNOWN"]
    assert PROVIDER_KEY not in json.dumps(event)


@pytest.mark.parametrize("mode", ["google_rpc", "html_gateway", "client_timeout"])
def test_distinguish_upstream_504_and_local_timeout(settings, tmp_path, mode):
    settings = config(settings, tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        if mode == "client_timeout":
            raise httpx.ReadTimeout("PRIVATE_LEARNING_TEXT secret-url", request=request)
        if mode == "html_gateway":
            return httpx.Response(
                504,
                text="<html>PRIVATE_LEARNING_TEXT secret-url</html>",
                headers={"Content-Type": "text/html"},
            )
        return httpx.Response(
            504,
            json={
                "error": {
                    "code": 504,
                    "status": "DEADLINE_EXCEEDED",
                    "message": "PRIVATE_LEARNING_TEXT " + PROVIDER_KEY,
                }
            },
        )

    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        with diagnostic_scope(settings, "test_deadline", settings.default_model_id):
            client = SDKClient("gemini", PROVIDER_KEY, settings)
            try:
                with pytest.raises(RequestTimeoutError):
                    client.generate_json(
                        model="fixture-model", prompt="PRIVATE_LEARNING_TEXT", schema=Keywords
                    )
            finally:
                client.close()
    assert len(calls) == 1
    error = next(e for e in rows(tmp_path) if e["stage"] == "sdk_request_failed")
    expected = {
        "google_rpc": "google_rpc_http_response",
        "html_gateway": "upstream_http_unverified",
        "client_timeout": "client_timeout",
    }
    assert error["error_origin"] == expected[mode]
    assert error["http_status"] == (None if mode == "client_timeout" else 504)
    assert error["original_code"] == (None if mode == "client_timeout" else 504)
    assert error["rpc_status"] == ("DEADLINE_EXCEEDED" if mode == "google_rpc" else None)
    assert all(
        private not in json.dumps(rows(tmp_path))
        for private in (PROVIDER_KEY, "PRIVATE_LEARNING_TEXT", "secret-url")
    )


def test_explain_preflight_reaches_sdk_and_maps_upstream_timeout(settings, tmp_path, context):
    settings = config(settings, tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            504,
            json={
                "error": {"code": 504, "status": "DEADLINE_EXCEEDED", "message": "PRIVATE_SOURCE"}
            },
        )

    body = ExplanationRequest(
        model_id=settings.default_model_id,
        context=context,
        concept_version_id=context.concepts[0].version.id,
    ).model_dump(mode="json")
    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/internal/explanations/generate",
                json=body,
                headers={"X-AI-API-Key": SERVER_KEY, "X-AI-Key": PROVIDER_KEY},
            )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "PROVIDER_TIMEOUT"
    assert "HTTP 504" in response.json()["error"]["message"] and len(calls) == 1
    events = rows(tmp_path)
    stages = [e["stage"] for e in events]
    assert stages.index("feature_preflight_passed") < stages.index("sdk_request_started")
    assert stages[-1] == "function_failed"
    assert all(e["feature"] == "src.explanation.generate" for e in events)
    assert all(e["credential_source"] == "request_header" for e in events)
    assert len({e["trace_id"] for e in events}) == 1


def test_explain_bad_evidence_stops_before_any_http(settings, tmp_path, context):
    settings = config(settings, tmp_path)
    body = ExplanationRequest(
        model_id=settings.default_model_id,
        context=context,
        concept_version_id=context.concepts[0].version.id,
    ).model_dump(mode="json")
    body["context"]["concepts"][0]["version"]["evidence"][0]["excerpt"] = "PRIVATE_FAKE_SOURCE"
    with patch(
        "api.gemini.genai.Client",
        side_effect=factory_for(
            lambda request: pytest.fail("No HTTP allowed before source validation")
        ),
    ):
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/internal/explanations/generate",
                json=body,
                headers={"X-AI-API-Key": SERVER_KEY, "X-AI-Key": PROVIDER_KEY},
            )
    assert response.status_code == 422
    assert "sdk_request_started" not in [e["stage"] for e in rows(tmp_path)]
    assert "PRIVATE_FAKE_SOURCE" not in json.dumps(rows(tmp_path))


def test_explain_actual_sdk_wire_and_final_grounding_with_mock_http(settings, tmp_path, context):
    from jsonschema import Draft202012Validator

    settings = config(settings, tmp_path)
    payload = {
        "title": "Simple Present",
        "definition": {"text": "Habits and general truths.", "citations": [0]},
        "key_points": [], "rules": [], "cautions": [], "examples": [],
    }
    calls = []

    def handler(request):
        wire = json.loads(request.content)
        calls.append(wire)
        generation = wire["generationConfig"]
        assert generation["responseMimeType"] == "application/json"
        assert "responseSchema" not in generation
        schema = generation["responseJsonSchema"]
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
        return httpx.Response(200, json=envelope(payload))

    body = ExplanationRequest(
        model_id=settings.default_model_id,
        context=context,
        concept_version_id=context.concepts[0].version.id,
    ).model_dump(mode="json")
    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/internal/explanations/generate", json=body,
                headers={"X-AI-API-Key": SERVER_KEY, "X-AI-Key": PROVIDER_KEY},
            )
    assert response.status_code == 200
    assert len(calls) == 1
    result = response.json()
    assert set(result) == {"title", "explanation_markdown", "evidence"}
    assert result["evidence"][0]["chunk_id"] == str(context.chunks[0].id)


def test_no_logs_in_production_or_when_disk_fails(settings, tmp_path):
    production = config(settings, tmp_path, environment="production")
    diagnostic_event(production, "test")
    assert not (tmp_path / ".cache").exists()
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked", encoding="utf-8")
    diagnostic_event(config(settings, blocked), "test", exc=RuntimeError("PRIVATE_SOURCE"))


def test_probe_default_never_constructs_sdk(settings):
    with patch("tests.smoke_gemini.SDKClient", side_effect=AssertionError("No network")):
        result = run(settings=settings)
    assert result["sdk_calls"] == 0 and result["status"] == "not_run"


@pytest.mark.parametrize("mode", ["text", "json"])
def test_each_probe_mode_sends_one_synthetic_call_and_no_source(settings, tmp_path, mode):
    settings = config(settings, tmp_path)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=envelope({"status": "ok"}))

    with patch("api.gemini.genai.Client", side_effect=factory_for(handler)):
        result = run(mode=mode, live=True, settings=settings)
    assert result["success"] and result["sdk_calls"] == 1 and len(calls) == 1
    assert calls[0]["generationConfig"]["maxOutputTokens"] == 512
    assert ("responseJsonSchema" in calls[0]["generationConfig"]) == (mode == "json")


def test_hostile_model_metadata_and_sdk_status_never_log_credentials(settings, tmp_path):
    from google.genai.errors import ServerError

    settings = config(settings, tmp_path)
    with diagnostic_scope(
        settings, "diagnostic", "gemini:" + PROVIDER_KEY, secrets=(PROVIDER_KEY,)
    ):
        diagnostic_event(
            settings,
            "failed",
            exc=ServerError(504, {"error": {"status": "PRIVATE_STATUS", "message": PROVIDER_KEY}}),
        )
    saved = json.dumps(rows(tmp_path))
    assert PROVIDER_KEY not in saved and "PRIVATE_STATUS" not in saved
    assert rows(tmp_path)[0]["rpc_status"] is None
    with diagnostic_scope(settings, "diagnostic", "unrecognized-private-token"):
        diagnostic_event(settings, "invalid_model")
    assert rows(tmp_path)[-1]["model_id"] is None
