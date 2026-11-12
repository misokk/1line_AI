"""Offline state, transport and contract regressions. No paid provider calls."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.gemini import LiveClient
from src.errors import InputError
from src.teachback import VoiceSessionRequest, VoiceSessions

HEADERS = {"X-AI-API-Key": "server-test-key"}


@pytest.fixture
def voice_request(context):
    return VoiceSessionRequest(
        context=context,
        concept_version_id=context.concepts[0].version.id,
    )


@pytest.fixture
def sessions():
    return VoiceSessions()


@pytest.fixture
def voice_settings(settings):
    return settings.model_copy(
        update={
            "server_api_key": SecretStr("server-test-key"),
            "gemini_api_key": SecretStr("private-gemini-key"),
            "gemini_live_enabled": True,
            "environment": "development",
        }
    )


def make_session(sessions, voice_request):
    ticket = sessions.create(voice_request, 30)
    return ticket, sessions.claim(ticket.session_id, ticket.connection_token)


def test_voice_prompt_one_korean_student_unlimited(voice_request, sessions, voice_settings):
    _, session = make_session(sessions, voice_request)
    config = LiveClient(voice_settings).connect_config(session)
    event = config.model_dump(mode="json", exclude_none=True)
    assert voice_settings.gemini_live_model == "gemini-3.8-live"
    assert event["response_modalities"] == ["AUDIO"]
    assert "thinking_config" not in event
    prompt = config.system_instruction
    assert "한국어" in prompt and "질문 횟수 제한은 없다" in prompt
    assert str(voice_request.concept_version_id) in prompt
    assert "private-gemini-key" not in json.dumps(event)
    for i in range(5):
        session.add_transcript(
            {
                "type": "session.output_transcript.delta",
                "event_id": str(i),
                "delta": "왜 그런가요?",
                "start_ms": i * 100,
                "end_ms": i * 100 + 99,
            }
        )
    assert session.question_count == 5


def test_voice_rejects_english_and_old_persona(context):
    from pydantic import ValidationError

    for extra in ({"explanation_language": "en"}, {"student_persona": "curious_beginner"}):
        with pytest.raises(ValidationError):
            VoiceSessionRequest(
                context=context, concept_version_id=context.concepts[0].version.id, **extra
            )


def test_one_time_and_expired_auth(sessions, voice_request):
    ticket, session = make_session(sessions, voice_request)
    with pytest.raises(InputError):
        sessions.claim(ticket.session_id, ticket.connection_token)
    ticket2 = sessions.create(voice_request, 30)
    sessions.get(ticket2.session_id).created -= 61
    with pytest.raises(InputError):
        sessions.claim(ticket2.session_id, ticket2.connection_token)
    assert ticket.connection_token not in session.token_hash


def test_eval_server_transcript_only_and_ids(sessions, voice_request):
    _, session = make_session(sessions, voice_request)
    for i, (role, text, start) in enumerate(
        [
            ("input", "It describes", 0),
            ("output", "PRIVATE ASSISTANT ANSWER", 100),
            ("input", " habits.", 300),
        ]
    ):
        event = {
            "type": f"session.{role}_transcript.delta",
            "event_id": str(i),
            "delta": text,
            "start_ms": start,
            "end_ms": start + 100,
        }
        assert session.add_transcript(event)
        assert session.add_transcript(event) is None
    with pytest.raises(InputError):
        session.evaluation_request("gemini:fixture-model")
    session.status = "closed"
    session.provider_finalized = True
    request = session.evaluation_request("gemini:fixture-model")
    assert request.history[0].content == "It describes habits."
    assert "PRIVATE ASSISTANT ANSWER" not in request.model_dump_json()
    assert request.concept_version_id == voice_request.concept_version_id
    assert request.context == voice_request.context


def test_http_session_auth_no_provider_call_and_safe_recording(
    voice_settings, voice_request, tmp_path
):
    settings = voice_settings.model_copy(update={"record_io": True, "recording_root": tmp_path})
    with patch("google.genai.live.ws_connect", side_effect=AssertionError("No network")):
        with TestClient(create_app(settings)) as client:
            assert client.post("/internal/teachback/sessions").status_code == 401
            result = client.post(
                "/internal/teachback/sessions",
                headers=HEADERS,
                json=voice_request.model_dump(mode="json"),
            )
            assert result.status_code == 200
            body = result.json()
            assert (
                client.get(
                    "/internal/teachback/sessions/" + body["session_id"], headers=HEADERS
                ).status_code
                == 200
            )
            assert client.get("/voice").status_code == 200
            assert client.post("/internal/teachback/respond", headers=HEADERS).status_code == 404
            with client.websocket_connect(body["websocket_path"]) as ws:
                ws.send_json({"type": "authenticate", "connection_token": "bad"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
    saved = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert body["connection_token"] not in saved
    assert "private-gemini-key" not in saved
    assert "server-test-key" not in saved


def test_voice_disabled_fails_without_network(settings, voice_request):
    settings = settings.model_copy(update={"server_api_key": SecretStr("server-test-key")})
    with patch("google.genai.live.ws_connect", side_effect=AssertionError("No network")):
        with TestClient(create_app(settings)) as client:
            result = client.post(
                "/internal/teachback/sessions",
                headers=HEADERS,
                json=voice_request.model_dump(mode="json"),
            )
            assert result.status_code == 503


class GeminiSocket:
    """WebSocket-only fake; actual google-genai Live setup/send/receive code still runs."""

    def __init__(self, *, partial=False):
        self.events = asyncio.Queue()
        self.commands = []
        self.turn = 0
        self.partial = partial

    async def send(self, raw):
        event = json.loads(raw)
        self.commands.append(event)
        if "setup" in event:
            setup = event["setup"]
            assert setup["model"] == "models/gemini-3.8-live"
            assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
            assert "thinkingConfig" not in setup["generationConfig"]
            assert "inputAudioTranscription" in setup and "outputAudioTranscription" in setup
            await self.events.put({"setupComplete": {"sessionId": "mock-google-session"}})
        else:
            realtime = event["realtime_input"]
            if realtime.get("audioStreamEnd"):
                if self.partial:
                    await self.events.put(
                        {
                            "serverContent": {
                                "inputTranscription": {"text": "habits.", "finished": True},
                                "turnComplete": True,
                            }
                        }
                    )
                else:
                    await self.events.put({"serverContent": {"turnComplete": True}})
            else:
                assert realtime["audio"]["mime_type"] == "audio/pcm;rate=24000"
                self.turn += 1
                await self.events.put(
                    {
                        "serverContent": {
                            "inputTranscription": {
                                "text": "It describes " if self.partial else "habits",
                                "finished": not self.partial,
                            },
                            "outputTranscription": {"text": "왜 그런가요?", "finished": True},
                            "modelTurn": {
                                "role": "model",
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "audio/pcm;rate=24000",
                                            "data": base64.b64encode(b"\0\0" * 960).decode(),
                                        }
                                    }
                                ],
                            },
                            "turnComplete": not self.partial,
                        },
                        "usageMetadata": {"totalTokenCount": 10},
                    }
                )

    async def recv(self, **kwargs):
        return json.dumps(await self.events.get())


def test_duplex_transport_real_handlers_with_fake_provider(voice_settings, voice_request):
    socket = GeminiSocket()

    @asynccontextmanager
    async def connection(*args, **kwargs):
        assert "generativelanguage.googleapis.com/ws/" in args[0]
        assert "v1alpha" in args[0]
        yield socket

    with patch("google.genai.live.ws_connect", connection):
        with TestClient(create_app(voice_settings)) as client:
            body = client.post(
                "/internal/teachback/sessions",
                headers=HEADERS,
                json=voice_request.model_dump(mode="json"),
            ).json()
            with client.websocket_connect(body["websocket_path"]) as ws:
                ws.send_json({"type": "authenticate", "connection_token": body["connection_token"]})
                assert ws.receive_json()["type"] == "ready"
                # SDK receive() ends at a turn boundary; multiple turns must keep working.
                for _ in range(5):
                    ws.send_bytes(b"\0\0" * 960)
                    assert ws.receive_json()["role"] == "user"
                    assert ws.receive_json()["role"] == "assistant"
                    assert ws.receive_bytes() == b"\0\0" * 960
                ws.send_json({"type": "end"})
                assert ws.receive_json()["type"] == "playback.clear"
                closed = ws.receive_json()
                assert closed["type"] == "closed" and closed["provider_finalized"]
            result = client.get(
                "/internal/teachback/sessions/" + body["session_id"], headers=HEADERS
            ).json()
            assert result["status"] == "closed" and result["question_count"] == 5
            assert result["usage"]["total_token_count"] == 10
            assert socket.commands[-1]["realtime_input"]["audioStreamEnd"] is True


@pytest.mark.parametrize("completed,new_speech", [(True, False), (False, False), (True, True)])
def test_end_without_new_provider_turn_preserves_only_confirmed_completion(
    voice_settings, voice_request, completed, new_speech,
):
    class NoEndAckSocket(GeminiSocket):
        async def send(self, raw):
            event = json.loads(raw)
            realtime = event.get("realtime_input", {})
            # Stream end has no independent acknowledgement when the provider is idle.
            # Subsequent microphone frames are silence and produce no further turn.
            if realtime.get("audioStreamEnd") or realtime.get("audio") and self.turn:
                self.commands.append(event)
                return
            await super().send(raw)

    socket = NoEndAckSocket(partial=not completed)
    original_wait_for = asyncio.wait_for

    async def fast_finalization_wait(awaitable, timeout):
        return await original_wait_for(awaitable, 0.05 if timeout == 15 else timeout)

    @asynccontextmanager
    async def connection(*args, **kwargs):
        yield socket

    with (
        patch("google.genai.live.ws_connect", connection),
        patch("api.gemini.asyncio.wait_for", fast_finalization_wait),
    ):
        with TestClient(create_app(voice_settings)) as client:
            body = client.post(
                "/internal/teachback/sessions", headers=HEADERS,
                json=voice_request.model_dump(mode="json"),
            ).json()
            with client.websocket_connect(body["websocket_path"]) as ws:
                ws.send_json({"type": "authenticate", "connection_token": body["connection_token"]})
                assert ws.receive_json()["type"] == "ready"
                ws.send_bytes(b"\0\0" * 960)
                if completed:
                    assert ws.receive_json()["role"] == "user"
                assert ws.receive_json()["role"] == "assistant"
                ws.receive_bytes()
                ws.send_bytes(b"\xd0\x07" * 960 if new_speech else b"\0\0" * 960)
                if new_speech:
                    assert ws.receive_json()["type"] == "playback.clear"
                ws.send_json({"type": "end"})
                assert ws.receive_json()["type"] == "playback.clear"
                closed = ws.receive_json()
                finalized = completed and not new_speech
                assert closed["provider_finalized"] is finalized
                assert closed["status"] == ("closed" if finalized else "failed")
                assert closed["error_code"] == (None if finalized else "PROVIDER_TIMEOUT")
            result = client.get(
                f"/internal/teachback/sessions/{body['session_id']}", headers=HEADERS,
            ).json()
            assert result["provider_finalized"] is finalized
            if finalized:
                assert any(f["role"] == "user" for f in result["transcript"])
            else:
                response = client.post(
                    f"/internal/teachback/sessions/{body['session_id']}/evaluate-detailed",
                    headers=HEADERS, json={"model_id": "gemini:fixture-model"},
                )
                assert response.status_code == 422


def test_immediate_duplex_unlimited_questions_and_gemini_evaluation_http(
    voice_settings, voice_request
):
    socket = GeminiSocket(partial=True)

    @asynccontextmanager
    async def connection(*args, **kwargs):
        yield socket

    seen = []

    def evaluate(request, router):
        from src import evaluation
        from src.models import CriterionJudgment, DetailedEvaluationResult, EvaluationJudgments

        seen.append(request)
        concept = request.context.concepts[0]
        report = evaluation.calculate_report(
            EvaluationJudgments(
                items=[
                    CriterionJudgment(
                        checklist_item_id=c.id,
                        status="explained",
                        description="Fixture",
                        confidence="high",
                        evidence=concept.version.evidence,
                        learner_quote="habits",
                    )
                    for c in concept.version.checklist_items
                ],
                next_action="Review",
            ),
            concept,
        )
        return DetailedEvaluationResult(report=report, feedback=[])

    from contextlib import contextmanager

    @contextmanager
    def router(*args):
        yield object()

    with patch("google.genai.live.ws_connect", connection):
        with TestClient(create_app(voice_settings)) as client:
            body = client.post(
                "/internal/teachback/sessions",
                headers=HEADERS,
                json=voice_request.model_dump(mode="json"),
            ).json()
            with client.websocket_connect(body["websocket_path"]) as ws:
                ws.send_json({"type": "authenticate", "connection_token": body["connection_token"]})
                assert ws.receive_json()["type"] == "ready"
                ws.send_bytes(b"\0\0" * 960)
                event = ws.receive_json()
                assert event.get("role") == "assistant", event
                # Audio is forwarded before the learner's final transcript is available.
                assert ws.receive_bytes() == b"\0\0" * 960
                await_event = {"serverContent": {"interrupted": True}}
                # Enqueue from the test client's event loop for thread-safe delivery.
                client.portal.call(socket.events.put, await_event)
                assert ws.receive_json()["type"] == "playback.clear"
                client.portal.call(socket.events.put, {"serverContent": {"turnComplete": True}})
                ws.send_json({"type": "end"})
                assert ws.receive_json()["type"] == "playback.clear"
                final = ws.receive_json()
                assert final["role"] == "user" and final["delta"] == "It describes habits."
                assert ws.receive_json()["provider_finalized"]
            path = f"/internal/teachback/sessions/{body['session_id']}/evaluate-detailed"
            with (
                patch("api.app.request_router", router),
                patch("api.app.evaluation.evaluate_detailed", evaluate),
            ):
                first = client.post(
                    path, headers=HEADERS, json={"model_id": "gemini:fixture-model"}
                )
                second = client.post(
                    path, headers=HEADERS, json={"model_id": "gemini:fixture-model"}
                )
                legacy = client.post(
                    path.replace("evaluate-detailed", "evaluate"),
                    headers=HEADERS,
                    json={"model_id": "gemini:fixture-model"},
                )
            assert first.status_code == second.status_code == 200
            assert first.json() == second.json() and len(seen) == 1
            assert legacy.json() == first.json()["report"]
            assert seen[0].history[0].content == "It describes habits."
            assert all(m.role == "user" for m in seen[0].history)


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "PROVIDER_REQUEST_INVALID"),
        (401, "PROVIDER_AUTH"),
        (429, "RATE_LIMITED"),
        (502, "PROVIDER_BAD_GATEWAY"),
        (503, "PROVIDER_SERVICE_UNAVAILABLE"),
    ],
)
def test_provider_handshake_errors_are_safe_no_retry(
    voice_settings, voice_request, sessions, status, code
):
    from unittest.mock import MagicMock

    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    _, session = make_session(sessions, voice_request)
    with patch(
        "google.genai.live.ws_connect",
        side_effect=InvalidStatus(
            Response(status, "private-provider-message", [], b"PRIVATE_SOURCE")
        ),
    ) as connect_mock:
        asyncio.run(LiveClient(voice_settings).bridge(MagicMock(), session))
    connect_mock.assert_called_once()
    assert session.status == "failed" and session.error_code == code
    assert "PRIVATE_SOURCE" not in session.result().model_dump_json()


def test_provider_timeout_is_safe_no_retry(voice_settings, voice_request, sessions):
    from unittest.mock import MagicMock

    _, session = make_session(sessions, voice_request)
    with patch(
        "google.genai.live.ws_connect", side_effect=TimeoutError("private-provider-detail")
    ) as factory:
        asyncio.run(LiveClient(voice_settings).bridge(MagicMock(), session))
    factory.assert_called_once()
    assert session.error_code == "PROVIDER_TIMEOUT" and not session.provider_finalized
    with pytest.raises(InputError):
        session.evaluation_request("gemini:fixture-model")
