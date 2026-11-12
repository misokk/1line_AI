import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.app import create_app
from api.config import Settings
from src.models import GradeRequest


def configured(settings, tmp_path, **updates):
    return settings.model_copy(
        update={
            "server_api_key": SecretStr("recording-server-secret"),
            "environment": "development",
            "record_io": True,
            "record_content": True,
            "recording_root": tmp_path,
            **updates,
        }
    )


def records(tmp_path):
    inputs = sorted((tmp_path / "input").glob("*_input.json"))
    outputs = sorted((tmp_path / "output").glob("*_output.json"))
    assert [p.name.removesuffix("_input.json") for p in inputs] == [
        p.name.removesuffix("_output.json") for p in outputs
    ]
    return [
        (
            json.loads(left.read_text(encoding="utf-8")),
            json.loads(right.read_text(encoding="utf-8")),
        )
        for left, right in zip(inputs, outputs, strict=True)
    ]


def test_real_grade_http_records_original_contract(settings, tmp_path, context, stored):
    config = configured(settings, tmp_path, server_api_key=SecretStr("recording-server-secret"))
    request = GradeRequest(context=context, stored_question=stored, answer=stored.correct_answer)
    payload = request.model_dump(mode="json")
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/internal/questions/grade",
            json=payload,
            headers={"X-AI-API-Key": "recording-server-secret"},
        )
    assert response.status_code == 200
    assert records(tmp_path) == [(payload, response.json())]
    assert "verification" not in response.json()
    assert not any("request_id" in p for p in response.json())


def test_private_content_minimized_by_default_actual_http_only(settings, tmp_path, context, stored):
    config = configured(settings, tmp_path, record_content=False)
    request = GradeRequest(context=context, stored_question=stored, answer=stored.correct_answer)
    app = create_app(config)
    assert not (tmp_path / "input").exists()
    with TestClient(app) as client:
        response = client.post(
            "/internal/questions/grade",
            json=request.model_dump(mode="json"),
            headers={"X-AI-API-Key": "recording-server-secret"},
        )
    assert response.status_code == 200
    incoming, outgoing = records(tmp_path)[0]
    assert incoming["context"]["chunks"][0]["text"]["redacted"]
    assert outgoing["explanation"]["redacted"]
    assert outgoing["is_correct"] is True
    assert context.chunks[0].text not in json.dumps(incoming, ensure_ascii=False)
    saved = json.loads(next((tmp_path / "output").glob("*.json")).read_text(encoding="utf-8"))
    assert saved == outgoing
    assert (tmp_path / "output/questions_grade_output.json").is_file()


@pytest.mark.parametrize(
    "environment,enabled", [("development", False), ("production", True), ("test", True)]
)
def test_recording_off_by_default_and_forbidden_in_production(
    settings, tmp_path, environment, enabled
):
    assert Settings().record_io is False
    config = configured(
        settings,
        tmp_path,
        environment=environment,
        record_io=enabled,
        server_api_key=SecretStr("server"),
    )
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
    assert not (tmp_path / "input").exists()
    assert not (tmp_path / "output").exists()


def test_secrets_redacted_without_modifying_response(settings, tmp_path):
    config = configured(settings, tmp_path, server_api_key=SecretStr("server-secret"))
    app = create_app(config)

    @app.post("/internal/echo")
    async def echo(request: Request):
        return await request.json()

    key = "gsk_" + "a" * 40
    payload = {
        "text": f"학습 발화 server-secret {key} Bearer unrelated-token",
        "nested": {"access_token": "unknown-secret", "AI_MAX_OUTPUT_TOKENS": 4096},
        "password": "password-value",
        "audio": "PRIVATE_AUDIO_BYTES",
        "private_key": "unknown-private-key",
    }
    with TestClient(app) as client:
        response = client.post("/internal/echo", json=payload, headers={"X-AI-Key": key})
    assert response.json() == payload
    incoming, outgoing = records(tmp_path)[0]
    assert incoming == outgoing
    assert incoming["nested"]["AI_MAX_OUTPUT_TOKENS"] == 4096
    assert "학습 발화" in incoming["text"]
    saved = json.dumps(incoming, ensure_ascii=False)
    for secret in [
        key,
        "server-secret",
        "unknown-secret",
        "password-value",
        "unrelated-token",
        "PRIVATE_AUDIO_BYTES",
        "unknown-private-key",
    ]:
        assert secret not in saved


def test_gemini_server_key_redacted_even_when_not_in_standard_key_format(settings, tmp_path):
    config = configured(
        settings, tmp_path, gemini_api_key=SecretStr("private-gemini-key-without-standard-prefix")
    )
    app = create_app(config)

    @app.post("/internal/echo")
    async def echo(request: Request):
        return await request.json()

    payload = {"text": "private-gemini-key-without-standard-prefix"}
    with TestClient(app) as client:
        response = client.post("/internal/echo", json=payload)
    assert response.json() == payload
    incoming, outgoing = records(tmp_path)[0]
    assert incoming["text"] == outgoing["text"] == "[REDACTED]"


def test_non_json_voice_input_never_saved(settings, tmp_path):
    config = configured(settings, tmp_path)
    data = b"PRIVATE_RAW_VOICE_STREAM"
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/internal/materials/extract",
            content=data,
            headers={
                "X-AI-API-Key": "recording-server-secret",
                "Content-Type": "audio/wav",
            },
        )
    assert response.status_code == 422
    incoming, _ = records(tmp_path)[0]
    assert incoming["size_bytes"] == len(data)
    assert incoming["mime_type"] == "audio/wav"
    assert data.decode() not in json.dumps(incoming)


def test_settings_recording_flags_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_ENV", raising=False)
    monkeypatch.delenv("AI_RECORD_IO", raising=False)
    path = tmp_path / "config.env"
    path.write_text("AI_ENV=development\nAI_RECORD_IO=true\n", encoding="utf-8")
    settings = Settings.from_env(path)
    assert settings.environment == "development" and settings.record_io


def test_binary_input_only_metadata_and_failed_output(settings, tmp_path, pdf_data):
    config = configured(
        settings, tmp_path, server_api_key=SecretStr("server"), upstage_api_key=SecretStr("")
    )
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/internal/materials/extract?filename=sample.pdf",
            content=pdf_data,
            headers={"X-AI-API-Key": "server", "Content-Type": "application/pdf"},
        )
    assert response.status_code == 503
    incoming, outgoing = records(tmp_path)[0]
    assert incoming["filename"] == "sample.pdf"
    assert incoming["size_bytes"] == len(pdf_data)
    assert incoming["sha256"] == hashlib.sha256(pdf_data).hexdigest()
    assert incoming["mime_type"] == "application/pdf" and incoming["body_complete"]
    assert outgoing == response.json()
    assert "%PDF" not in json.dumps(incoming)


def test_failure_records_invalid_json_auth_and_body_limit(settings, tmp_path):
    config = configured(settings, tmp_path, server_api_key=SecretStr("recording-auth-key"))
    with TestClient(create_app(config)) as client:
        first = client.post("/internal/materials/analyze", json={})
        assert first.status_code == 401
        assert records(tmp_path) == [({}, first.json())]
        second = client.post(
            "/internal/materials/analyze",
            content=b'{"bad":',
            headers={"Content-Type": "application/json", "X-AI-API-Key": "recording-auth-key"},
        )
        assert second.status_code == 422
        assert records(tmp_path)[0][0]["recording"] == "body_not_available_as_json"
        assert records(tmp_path)[0][1] == second.json()
        third = client.post(
            "/internal/materials/analyze",
            content=b"x" * (2 * 1024 * 1024 + 1),
            headers={"Content-Type": "application/json"},
        )
        assert third.status_code == 413
        assert len(records(tmp_path)) == 1
        assert records(tmp_path)[0][1] == third.json()
    assert (tmp_path / "input/analysis_input.json").is_file()
    assert (tmp_path / "output/analysis_output.json").is_file()


def test_disk_failure_never_breaks_api(settings, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    config = configured(settings, blocked, server_api_key=SecretStr("server"))
    with TestClient(create_app(config)) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_concurrent_requests_keep_one_matching_latest_pair(settings, tmp_path):
    app = create_app(configured(settings, tmp_path))

    @app.post("/internal/echo")
    async def echo(request: Request):
        return await request.json()

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(lambda i: client.post("/internal/echo", json={"value": i}), range(12))
            )
        assert all(r.status_code == 200 for r in responses)
    pairs = records(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0] == pairs[0][1] and pairs[0][0]["value"] in range(12)
    assert not list(tmp_path.rglob("*.tmp"))


def test_same_function_success_then_error_overwrites_latest(settings, tmp_path, context, stored):
    app = create_app(configured(settings, tmp_path))
    body = GradeRequest(
        context=context, stored_question=stored, answer=stored.correct_answer
    ).model_dump(mode="json")
    with TestClient(app) as client:
        first = client.post(
            "/internal/questions/grade",
            json=body,
            headers={"X-AI-API-Key": "recording-server-secret"},
        )
        assert first.status_code == 200
        assert records(tmp_path) == [(body, first.json())]
        body["answer"]["selected_option_ids"] = ["unknown"]
        second = client.post(
            "/internal/questions/grade",
            json=body,
            headers={"X-AI-API-Key": "recording-server-secret"},
        )
        assert second.status_code == 422
    assert records(tmp_path) == [(body, second.json())]
    assert set(
        json.loads((tmp_path / "output/questions_grade_output.json").read_text(encoding="utf-8"))
    ) == {"error"}


def test_separate_functions_have_separate_fixed_files(settings, tmp_path):
    with TestClient(create_app(configured(settings, tmp_path))) as client:
        health = client.get("/health")
        models = client.get("/internal/models", headers={"X-AI-API-Key": "recording-server-secret"})
    assert len(records(tmp_path)) == 2
    assert (
        json.loads((tmp_path / "output/health_output.json").read_text(encoding="utf-8"))
        == health.json()
    )
    assert (
        json.loads((tmp_path / "output/models_output.json").read_text(encoding="utf-8"))
        == models.json()
    )


def test_unhandled_exception_preserves_500_without_leaking_message(settings, tmp_path):
    app = create_app(configured(settings, tmp_path, server_api_key=SecretStr("server")))

    @app.get("/internal/failure")
    def fail():
        raise RuntimeError("private credential and source text")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/internal/failure")
    assert response.status_code == 500 and response.text == "Internal Server Error"
    assert records(tmp_path)[0][1] == {
        "error": {"code": "UNHANDLED_EXCEPTION", "type": "RuntimeError"}
    }
