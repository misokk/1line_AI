from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.app import create_app
from api.config import Settings
from src.errors import (
    ConfigurationError,
    InvalidResponseError,
    ModelAccessError,
    ProviderError,
    RateLimitError,
    RequestTimeoutError,
)
from src.models import EvaluationRequest, Message, MultipleChoiceAnswer

HEADERS = {"X-AI-API-Key": "local-test-server-key"}
MODEL_ID = "gemini:fixture-model"


@pytest.fixture
def client(settings, parse_payload):
    settings = settings.model_copy(update={"server_api_key": SecretStr(HEADERS["X-AI-API-Key"])})
    import pymupdf

    def recognize(data, mime):
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return parse_payload(len(pdf))

    with patch("api.upstage.UpstageDocumentParser._request", side_effect=recognize):
        with TestClient(create_app(settings)) as client:
            yield client


def evaluation_payload(context):
    return EvaluationRequest(
        context=context,
        concept_version_id=context.concepts[0].version.id,
        model_id=MODEL_ID,
        explanation_language="ko",
        history=[Message(role="user", content="The simple present describes habits.")],
    ).model_dump(mode="json")


def test_health_docs_internal_only_and_no_backend_duplication(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert all(path == "/health" or path.startswith("/internal/") for path in schema["paths"])
    assert "/study-plan" not in schema["paths"] and "/questions/recommend" not in schema["paths"]
    assert schema["components"]["securitySchemes"]["ProviderBYOK"]["name"] == "X-AI-Key"
    assert "/internal/explanations/generate" in schema["paths"]
    assert "/internal/teachback/sessions" in schema["paths"]
    assert "/internal/teachback/respond" not in schema["paths"]
    assert (
        client.post("/internal/explanations/generate", headers=HEADERS, json={}).status_code == 422
    )


def test_internal_openapi_matches_before_refactor(client):
    import json
    from pathlib import Path

    baseline = json.loads(
        (Path(__file__).parent / "fixtures/internal_openapi.json").read_text(encoding="utf-8")
    )
    assert client.get("/openapi.json").json() == baseline


@pytest.mark.parametrize("headers", [{}, {"X-AI-API-Key": "wrong"}])
def test_all_function_routes_require_server_auth(client, headers):
    schema = client.get("/openapi.json").json()
    for path in schema["paths"]:
        if path != "/health":
            method = (
                client.get
                if path == "/internal/models" or path.endswith("{session_id}")
                else client.post
            )
            assert (
                method(
                    path.replace("{session_id}", "00000001-1111-4111-8111-111111111111"),
                    headers=headers,
                ).status_code
                == 401
            )


def test_no_server_key_fails_closed():
    with pytest.raises(ConfigurationError):
        with TestClient(create_app(Settings())):
            raise AssertionError("Startup must fail")


def test_mocked_parse_and_real_txt_http(client, pdf_data):
    result = client.post(
        "/internal/materials/extract",
        content=pdf_data,
        headers={**HEADERS, "Content-Type": "application/pdf"},
    )
    assert result.status_code == 200 and len(result.json()["document"]["pages"]) == 2
    assert "id" not in result.json()["chunk_drafts"][0]
    result = client.post(
        "/internal/materials/extract",
        content=b"Simple present describes habits.",
        headers={**HEADERS, "Content-Type": "text/plain"},
    )
    assert result.status_code == 200 and result.json()["document"]["pages"][0]["number"] == 1


def test_pdf_errors_and_size_bound(client):
    headers = {**HEADERS, "Content-Type": "application/pdf"}
    assert (
        client.post("/internal/materials/extract", content=b"bad", headers=headers).status_code
        == 422
    )
    assert (
        client.post(
            "/internal/materials/extract", content=b"x" * (20 * 1024 * 1024 + 1), headers=headers
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/internal/evaluations/generate", content=b"x" * (2 * 1024 * 1024 + 1), headers=HEADERS
        ).status_code
        == 413
    )


def test_validation_does_not_echo_private_inputs(client):
    result = client.post(
        "/internal/evaluations/generate",
        headers=HEADERS,
        json={"concept_version_id": "private-secret-invalid-uuid"},
    )
    assert result.status_code == 422 and "private-secret" not in result.text


def test_unknown_model_rejected_before_provider(client, context):
    body = evaluation_payload(context)
    body["model_id"] = "https://attacker.invalid/arbitrary"
    with patch("api.router.SDKClient", side_effect=AssertionError("No provider call")):
        result = client.post("/internal/evaluations/generate", headers=HEADERS, json=body)
    assert result.status_code == 422 and result.json()["error"]["code"] == "MODEL_NOT_ALLOWED"


def test_http_provider_key_never_falls_back_to_server_key(client, context):
    result = client.post(
        "/internal/evaluations/generate", headers=HEADERS, json=evaluation_payload(context)
    )
    assert result.status_code == 422
    with patch("api.router.SDKClient") as factory:
        key = "request-only-private-key"
        factory.return_value.generate_json.side_effect = InvalidResponseError(
            "Rejected fixture output"
        )
        result = client.post(
            "/internal/evaluations/generate",
            headers={**HEADERS, "X-AI-Key": key},
            json=evaluation_payload(context),
        )
        assert factory.call_args.args[1] == key
        factory.return_value.close.assert_called_once()
        assert key not in result.text


def test_server_text_key_requires_explicit_opt_in(settings):
    from api.router import request_router

    settings = settings.model_copy(update={"gemini_api_key": SecretStr("server-only-key")})
    with patch("api.router.SDKClient") as factory:
        with pytest.raises(ModelAccessError):
            with request_router(settings, MODEL_ID, None):
                raise AssertionError("BYOK default must reject missing request key")
        factory.assert_not_called()
        with request_router(
            settings.model_copy(update={"use_server_text_key": True}), MODEL_ID, None
        ):
            assert factory.call_args.args[1] == "server-only-key"
        factory.return_value.close.assert_called_once()


def test_unlisted_gemini_model_uses_request_key_and_preserves_contract(settings, context):
    config = settings.model_copy(
        update={"server_api_key": SecretStr(HEADERS["X-AI-API-Key"]), "allow_unlisted_models": True}
    )
    with patch("api.router.SDKClient") as factory:
        factory.return_value.generate_json.side_effect = InvalidResponseError("Fixture response")
        with TestClient(create_app(config)) as client:
            body = evaluation_payload(context)
            body["model_id"] = "gemini:test-unlisted-model"
            assert client.post("/internal/evaluations/generate", json=body).status_code == 401
            factory.assert_not_called()
            result = client.post(
                "/internal/evaluations/generate",
                json=body,
                headers={**HEADERS, "X-AI-Key": "request-model-key"},
            )
            assert result.status_code == 502 and result.json()["error"]["code"] == "INVALID_OUTPUT"
            assert factory.call_args.args[1] == "request-model-key"
            assert (
                factory.return_value.generate_json.call_args.kwargs["model"]
                == "test-unlisted-model"
            )
        factory.return_value.close.assert_called_once()


def test_deterministic_grading_needs_no_provider_key(client, context, stored):
    with patch("api.router.SDKClient", side_effect=AssertionError("No provider call")):
        result = client.post(
            "/internal/questions/grade",
            headers=HEADERS,
            json={
                "context": context.model_dump(mode="json"),
                "stored_question": stored.model_dump(mode="json"),
                "answer": MultipleChoiceAnswer(
                    type="multiple_choice", selected_option_ids=["A"]
                ).model_dump(),
            },
        )
    assert result.status_code == 200 and result.json()["is_correct"]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RateLimitError("quota"), 429, "RATE_LIMITED"),
        (RequestTimeoutError("timeout"), 504, "PROVIDER_TIMEOUT"),
        (InvalidResponseError("schema"), 502, "INVALID_OUTPUT"),
        (ProviderError("upstream"), 502, "PROVIDER_UNAVAILABLE"),
        (ModelAccessError("auth"), 422, "PROVIDER_AUTH"),
    ],
)
def test_provider_errors_map_to_http(client, context, error, status, code):
    with patch("api.router.SDKClient") as factory:
        factory.return_value.generate_json.side_effect = error
        result = client.post(
            "/internal/evaluations/generate",
            headers={**HEADERS, "X-AI-Key": "test-key"},
            json=evaluation_payload(context),
        )
    assert result.status_code == status and result.json()["error"]["code"] == code


def test_validate_key_never_generates_and_reports_unavailable(client):
    with patch("api.router.SDKClient") as factory:
        factory.return_value.check_model_access.side_effect = ProviderError("Unavailable")
        result = client.post(
            "/internal/models/validate-key",
            headers={**HEADERS, "X-AI-Key": "test-key"},
            json={"model_id": MODEL_ID},
        )
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "KEY_VALIDATION_UNAVAILABLE"
        factory.return_value.generate_json.assert_not_called()


def test_parse_requires_no_llm_provider_key(client, pdf_data):
    with patch("api.router.SDKClient") as provider:
        result = client.post(
            "/internal/materials/extract",
            content=pdf_data,
            headers={**HEADERS, "Content-Type": "application/pdf"},
        )
    assert result.status_code == 200
    assert result.json()["method"] == "upstage_document_parse"
    provider.assert_not_called()


def test_legacy_selected_page_option_is_not_silently_ignored(client, pdf_data):
    result = client.post(
        "/internal/materials/extract",
        params={"ocr_pages": [1]},
        content=pdf_data,
        headers={**HEADERS, "Content-Type": "application/pdf"},
    )
    assert result.status_code == 422


def test_webp_not_in_backend_contract(client):
    result = client.post(
        "/internal/materials/extract",
        content=b"RIFFinvalidWEBP",
        headers={**HEADERS, "Content-Type": "image/webp"},
    )
    assert result.status_code == 422


def test_extract_docs_do_not_request_llm_model_or_byok(client):
    operation = client.get("/openapi.json").json()["paths"]["/internal/materials/extract"]["post"]
    names = [p["name"] for p in operation.get("parameters", [])]
    assert "model_id" not in names and "max_ocr_pages" not in names
    assert all("ProviderBYOK" not in security for security in operation.get("security", []))
