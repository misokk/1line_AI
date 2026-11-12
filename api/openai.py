"""Official OpenAI Structured Outputs and GPT-Live voice transport."""

import asyncio
import base64
import importlib
import json
import math
import struct
import time

from fastapi import WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, WebSocketException

from api.base import ImageInput, T, validate_json
from api.config import Settings
from src.errors import (
    ConfigurationError,
    InputError,
    InvalidResponseError,
    ModelAccessError,
    ProviderBadGatewayError,
    ProviderError,
    ProviderRequestError,
    ProviderServiceUnavailableError,
    RateLimitError,
    RequestTimeoutError,
)
from src.teachback import VoiceSession


def pcm_level(data: bytes) -> float:
    if not data or len(data) % 2:
        return 0
    samples = struct.unpack(f"<{len(data) // 2}h", data)
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768


class LiveClient:
    """Official GPT-Live primary WS. One connection, no fallback or automatic retry."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def check_configuration(self):
        if not self.settings.openai_live_enabled:
            raise ConfigurationError("Enable OPENAI_LIVE_ENABLED explicitly for voice calls.")
        if not self.settings.openai_api_key.get_secret_value().strip():
            raise ConfigurationError("Set OPENAI_API_KEY on the server for voice sessions.")

    def start_event(self, session: VoiceSession):
        return {
            "type": "session.start",
            "session": {
                "model": self.settings.openai_live_model,
                "instructions": session.instructions(),
                "store": False,
                "audio": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "output": {"voice": self.settings.openai_live_voice},
                },
                "delegation": {"type": "client"},
            },
        }

    @staticmethod
    def event_error(event):
        error = event.get("error", {})
        code = error.get("code") or error.get("type")
        if code in ("rate_limit_exceeded", "too_many_requests"):
            return "RATE_LIMITED"
        if code in ("invalid_api_key", "authentication_error", "permission_denied"):
            return "PROVIDER_AUTH"
        if code in (
            "invalid_request_error",
            "unknown_parameter",
            "model_not_found",
            "unsupported_model",
        ):
            return "PROVIDER_REQUEST_INVALID"
        if code in ("server_error", "overloaded"):
            return "PROVIDER_SERVICE_UNAVAILABLE"
        return "PROVIDER_EVENT_ERROR"

    async def bridge(self, browser, session: VoiceSession):
        self.check_configuration()
        audit_started = time.perf_counter()
        # Validate the immutable context before opening a potentially billable connection.
        self.start_event(session)
        try:
            async with connect(
                "wss://api.openai.com/v1/live/sessions",
                additional_headers={
                    "Authorization": "Bearer " + self.settings.openai_api_key.get_secret_value(),
                },
                open_timeout=20,
                close_timeout=5,
                max_size=2 * 1024 * 1024,
                max_queue=16,
            ) as provider:
                await self._stream(browser, provider, session)
        except InvalidStatus as exc:
            status = exc.response.status_code
            code = {
                401: "PROVIDER_AUTH",
                403: "PROVIDER_AUTH",
                400: "PROVIDER_REQUEST_INVALID",
                429: "RATE_LIMITED",
                502: "PROVIDER_BAD_GATEWAY",
                503: "PROVIDER_SERVICE_UNAVAILABLE",
            }.get(status, "PROVIDER_UNAVAILABLE")
            session.error_code = code
        except TimeoutError:
            session.error_code = "PROVIDER_TIMEOUT"
        except (WebSocketException, OSError):
            session.error_code = "PROVIDER_UNAVAILABLE"
        except (ValueError, KeyError, TypeError, InvalidResponseError, InputError):
            session.error_code = "INVALID_OUTPUT"
        finally:
            if not session.provider_finalized:
                session.status = "failed"
                session.error_code = session.error_code or "INCOMPLETE_FINALIZATION"
            from api.recording import record_exchange

            await record_exchange(
                self.settings,
                "teachback_voice",
                str(session.id),
                session.request.model_dump(mode="json"),
                session.result().model_dump(mode="json"),
                status=200 if session.status == "closed" else 502,
                duration_ms=(time.perf_counter() - audit_started) * 1000,
                secrets=[
                    v.get_secret_value()
                    for v in (
                        self.settings.server_api_key,
                        self.settings.openai_api_key,
                        self.settings.upstage_api_key,
                    )
                    if v.get_secret_value()
                ],
            )

    async def _stream(self, browser, provider, session):
        async def send(event):
            await provider.send(json.dumps(event))

        await send(self.start_event(session))
        async with asyncio.timeout(20):
            while True:
                event = json.loads(await provider.recv())
                if event.get("type") == "error":
                    session.error_code = self.event_error(event)
                    return
                if event.get("type") == "session.started":
                    break
                if event.get("type") != "info":
                    raise InvalidResponseError("GPT-Live did not start; details withheld.")
        await browser.send_json({"type": "ready", "sample_rate": 24000})
        closing = False
        input_voiced_until = 0.0
        frame_bytes = 0
        began = time.monotonic()

        async def flush():
            await browser.send_json({"type": "playback.clear"})

        async def receive_browser():
            nonlocal closing, input_voiced_until, frame_bytes
            while not closing:
                message = await browser.receive()
                if message["type"] == "websocket.disconnect":
                    closing = True
                    await send({"type": "session.close"})
                    return
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    if not data or len(data) > 9600 or len(data) % 2:
                        raise InvalidResponseError("Expected mono PCM16 audio frames up to 200 ms.")
                    frame_bytes += len(data)
                    if frame_bytes > 48000 * (time.monotonic() - began + 2):
                        raise InvalidResponseError("Microphone stream exceeds realtime rate.")
                    if pcm_level(data) > 0.025:
                        now = time.monotonic()
                        if now >= input_voiced_until:
                            # Only a NEW speech onset interrupts. During continued user speech,
                            # permitted backchannels can still play (full duplex).
                            await flush()
                        input_voiced_until = now + 0.35
                    await send(
                        {
                            "type": "session.input_audio.append",
                            "audio": base64.b64encode(data).decode("ascii"),
                        }
                    )
                else:
                    command = json.loads(message.get("text", "{}"))
                    if command == {"type": "end"}:
                        closing = True
                        await flush()
                        await send({"type": "session.close"})
                        return
                    raise InvalidResponseError("Only PCM audio and the end command are accepted.")

        async def receive_provider():
            async for raw in provider:
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "error":
                    session.error_code = self.event_error(event)
                    return
                if kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
                    fragment = session.add_transcript(event)
                    if fragment:
                        await browser.send_json({"type": "transcript", **fragment.model_dump()})
                elif kind == "session.output_audio.delta" and not closing:
                    audio = base64.b64decode(event["delta"], validate=True)
                    if len(audio) % 2:
                        raise InvalidResponseError("Provider returned an incomplete PCM16 sample.")
                    # Native full duplex: forward audio immediately, independently of captions.
                    await browser.send_bytes(audio)
                elif kind == "session.usage.updated":
                    session.usage = event.get("usage", {})
                elif kind == "session.closed":
                    session.provider_finalized = True
                    reason = event.get("reason", "close_requested")
                    session.status = (
                        "closed"
                        if reason in ("close_requested", "expired", "remote_hangup")
                        else "failed"
                    )
                    if session.status == "failed":
                        session.error_code = "PROVIDER_SESSION_ENDED"
                    session.usage = event.get("usage", session.usage)
                    return
                elif kind == "session.delegation.created":
                    await send(
                        {
                            "type": "session.instructions.append",
                            "delegation_id": None,
                            "content": "Do not delegate. Stay in the student role and listen.",
                        }
                    )

        async def deadline():
            nonlocal closing
            await asyncio.sleep(session.duration)
            closing = True
            await flush()
            await send({"type": "session.close"})

        tasks = [asyncio.create_task(f()) for f in (receive_browser, receive_provider, deadline)]
        try:
            # A client end/disconnect starts a bounded provider drain for final transcripts/usage.
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=session.duration + 15
            )
            for task in done:
                task.result()
            if not tasks[1].done():
                await asyncio.wait_for(asyncio.shield(tasks[1]), 15)
        except (WebSocketDisconnect, RuntimeError):
            session.error_code = "CLIENT_DISCONNECTED"
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class SDKClient:
    def __init__(self, provider: str, api_key: str, settings: Settings):
        if not settings.live_enabled:
            raise ConfigurationError("Provider calls are disabled by AI_LIVE_ENABLED.")
        if provider != "openai":
            raise ConfigurationError("Only the OpenAI text SDK is supported.")
        if not api_key.strip():
            raise ConfigurationError("OPENAI_API_KEY or request-scoped X-AI-Key is required.")
        self.provider = provider
        self.settings = settings
        try:
            sdk = importlib.import_module(provider)
        except ImportError:
            raise ConfigurationError(
                f"Install the optional {provider} SDK before enabling it."
            ) from None
        constructor = sdk.OpenAI
        endpoint = "https://api.openai.com/v1"
        self.client = constructor(
            api_key=api_key, base_url=endpoint, timeout=settings.timeout_seconds, max_retries=0
        )

    def check_model_access(self, model: str) -> None:
        try:
            self.client.models.retrieve(model)
        except Exception as exc:
            self._error(exc)

    @staticmethod
    def _error(exc):
        code = getattr(exc, "status_code", None)
        if code == 429:
            retry = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            try:
                retry = min(86400, max(0, math.ceil(float(retry))))
            except (ValueError, TypeError, OverflowError):
                retry = None
            raise RateLimitError(
                "OpenAI quota exceeded; no automatic retry.", retry_after=retry
            ) from None
        if code in (401, 403, 404):
            raise ModelAccessError("OpenAI rejected credentials or model access.") from None
        if code in (400, 422):
            raise ProviderRequestError(
                "OpenAI rejected the request or output schema (HTTP 400/422)."
            ) from None
        if code == 502:
            raise ProviderBadGatewayError("OpenAI returned HTTP 502.") from None
        if code == 503:
            raise ProviderServiceUnavailableError("OpenAI returned HTTP 503.") from None
        if "Timeout" in type(exc).__name__:
            raise RequestTimeoutError(
                "Provider request timed out; outcome may be unknown."
            ) from None
        raise ProviderError(
            "OpenAI request failed; no automatic retry or raw provider response logged."
        ) from None

    def generate_json(
        self, *, model: str, prompt: str, schema: type[T], images: list[ImageInput] | None = None
    ) -> T:
        try:
            content = [{"type": "input_text", "text": prompt}]
            for image in images or []:
                encoded = base64.b64encode(image.data).decode("ascii")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{image.mime_type};base64,{encoded}",
                    }
                )
            response = self.client.responses.parse(
                model=model,
                input=[{"role": "user", "content": content}],
                text_format=schema,
                max_output_tokens=self.settings.max_output_tokens,
                store=False,
            )
            if response.status != "completed":
                raise InvalidResponseError("OpenAI output incomplete or refused.")
            if any(
                getattr(c, "type", None) == "refusal"
                for out in getattr(response, "output", [])
                for c in getattr(out, "content", [])
            ):
                raise InvalidResponseError("OpenAI refused the request; no output finalized.")
            return validate_json(response.output_text, schema)
        except InvalidResponseError:
            raise
        except Exception as exc:
            if type(exc).__name__ in (
                "ValidationError",
                "LengthFinishReasonError",
                "ContentFilterFinishReasonError",
            ):
                raise InvalidResponseError(
                    "OpenAI output failed structured schema validation."
                ) from None
            self._error(exc)

    def close(self):
        self.client.close()
