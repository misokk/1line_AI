"""Google Gen AI SDK adapter. Original domain schemas remain the final validator."""

import asyncio
import json
import math
import os
import re
import struct
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

import httpx
from google import genai
from google.genai import errors, types
from websockets.exceptions import ConnectionClosed, InvalidStatus

from api.base import ImageInput, T, validate_json
from api.config import Settings
from api.recording import redact
from src.errors import (
    AIError,
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

_TRACE = ContextVar("gemini_diagnostic_trace", default=None)
_DIAGNOSTIC_LOCK = Lock()
_RPC_STATUSES = {
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
}


@contextmanager
def diagnostic_scope(settings, feature, model_id, *, secrets=(), credential_source=None):
    parent = _TRACE.get()
    trace = (
        dict(parent)
        if parent
        else {
            "trace_id": uuid4().hex,
            "feature": feature,
            "model_id": model_id,
            "started": time.perf_counter(),
            "secrets": (),
        }
    )
    trace["secrets"] = (*trace["secrets"], *secrets)
    if not trace.get("credential_source") and credential_source in {
        "request_header",
        "server_environment",
        "sdk_argument",
        "missing",
    }:
        trace["credential_source"] = credential_source
    token = _TRACE.set(trace)
    try:
        yield
    finally:
        _TRACE.reset(token)


def error_diagnostics(exc):
    """Extract codes and fixed hints; never serialize raw provider message text."""
    response = getattr(exc, "response", None)
    http_status = getattr(response, "status_code", None)
    if type(http_status) is not int or not 100 <= http_status <= 599:
        http_status = None
    code = getattr(exc, "code", None)
    code = code if type(code) is int else None
    rpc_status = getattr(exc, "status", None)
    rpc_status = rpc_status if isinstance(rpc_status, str) and rpc_status in _RPC_STATUSES else None
    target_matches = None
    try:
        target_matches = response.request.url.host == "generativelanguage.googleapis.com"
    except (AttributeError, RuntimeError):
        pass
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        origin, message = (
            "client_timeout",
            "Client deadline reached without a complete HTTP response.",
        )
    elif isinstance(exc, errors.APIError):
        origin = (
            "google_rpc_http_response"
            if http_status and rpc_status and target_matches
            else "upstream_http_unverified"
            if http_status
            else "sdk_api_error_without_http_response"
        )
        message = "Upstream HTTP/API error; raw provider text omitted."
    elif isinstance(exc, (httpx.TransportError, OSError)):
        origin, message = (
            "transport_error",
            "Transport failed; no confirmed upstream HTTP response.",
        )
    elif isinstance(exc, AIError):
        origin, message = "application_error", "Application validation or mapped provider error."
    else:
        origin, message = "sdk_or_local_error", "SDK processing or local failure; raw text omitted."
    kind = type(exc).__name__
    if kind not in {
        "APIError",
        "ClientError",
        "ServerError",
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "LocalProtocolError",
        "ValidationError",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "OSError",
        "SSLError",
        "ConnectionError",
        "ConfigurationError",
        "InputError",
        "InvalidResponseError",
        "ProviderError",
        "ProviderBadGatewayError",
        "ProviderServiceUnavailableError",
        "ProviderRequestError",
        "ModelSelectionError",
        "ModelAccessError",
        "RateLimitError",
        "RequestTimeoutError",
    }:
        kind = "OtherError"
    result = {
        "error_origin": origin,
        "original_code": code,
        "http_status": http_status,
        "rpc_status": rpc_status,
        "target_host_matches_google": target_matches,
        "error_type": kind,
        "safe_message": message,
    }
    if isinstance(exc, errors.APIError) and code in (400, 422):
        # Match only fixed vocabulary in memory. Provider text can echo private input.
        raw = getattr(exc, "message", None)
        if isinstance(raw, str):
            result["request_error_hints"] = [
                label
                for label, pattern in (
                    ("unknown_field", r"unknown (?:name|field)"),
                    ("invalid_json_payload", r"invalid json payload"),
                    ("schema_complexity", r"schema.{0,40}(?:too complex|too large|too deep)"),
                    ("unsupported_parameter", r"(?:not supported|unsupported)"),
                    ("invalid_value", r"invalid value"),
                )
                if re.search(pattern, raw, re.IGNORECASE)
            ]
            result["mentioned_request_fields"] = [
                field
                for field in (
                    "responseJsonSchema", "response_json_schema", "responseSchema",
                    "response_schema", "responseMimeType", "response_mime_type",
                    "maxOutputTokens", "max_output_tokens", "thinkingConfig",
                    "generationConfig", "additionalProperties", "anyOf", "enum",
                    "minItems", "maxItems", "required", "propertyOrdering",
                )
                if re.search(r"(?<![\w])" + re.escape(field) + r"(?![\w])", raw)
            ]
    return result


def response_diagnostics(response):
    """Only enum codes, counts and usage; never content, finish messages or safety text."""
    candidates = response.candidates or []
    reasons = {reason.value for reason in types.FinishReason}
    blocked = {reason.value for reason in types.BlockedReason}
    feedback = response.prompt_feedback
    reason = feedback.block_reason if feedback else None
    result = {
        "candidate_count": len(candidates),
        "finish_reasons": [
            c.finish_reason if c.finish_reason in reasons else "UNKNOWN" for c in candidates
        ],
        "prompt_block_reason": reason if reason in blocked else None,
    }
    usage = response.usage_metadata
    result["token_usage"] = {
        field: value
        for field in (
            "prompt_token_count", "candidates_token_count", "thoughts_token_count",
            "total_token_count", "cached_content_token_count",
        )
        if type(value := getattr(usage, field, None)) is int and value >= 0
    }
    return result


def diagnostic_event(settings, stage, *, exc=None, response=None, **counts):
    """Development-only JSONL, separate from the unchanged input/output body recorder."""
    if settings.environment != "development":
        return
    try:
        trace = _TRACE.get() or {
            "trace_id": uuid4().hex,
            "feature": "sdk_initialization",
            "model_id": None,
            "started": time.perf_counter(),
            "secrets": (),
        }
        feature = trace["feature"]
        model_id = trace["model_id"]
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace["trace_id"],
            "feature": feature if re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", feature) else "unknown",
            "model_id": model_id
            if isinstance(model_id, str)
            and model_id.startswith(("gemini:", "gemini-"))
            and re.fullmatch(r"(?:gemini:)?[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", model_id)
            else None,
            "credential_source": trace.get("credential_source"),
            "stage": stage,
            "elapsed_ms": round((time.perf_counter() - trace["started"]) * 1000, 3),
            "sdk_version": genai.__version__,
            "timeout_seconds": settings.timeout_seconds,
            "server_deadline_seconds": math.ceil(settings.timeout_seconds),
            "max_attempts": 1,
            "proxy_environment_present": any(
                os.environ.get(k) for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
            ),
        }
        row.update(
            {
                k: v
                for k, v in counts.items()
                if k in {"prompt_characters", "schema_characters", "max_output_tokens"}
                and type(v) is int
            }
        )
        if counts.get("thinking_level") in {"LOW", "MEDIUM", "HIGH"}:
            row["thinking_level"] = counts["thinking_level"]
        if exc is not None:
            row.update(error_diagnostics(exc))
        if response is not None:
            row.update(response_diagnostics(response))
        secrets = (
            *trace["secrets"],
            *(
                s.get_secret_value()
                for s in (
                    settings.server_api_key,
                    settings.gemini_api_key,
                    settings.openai_api_key,
                    settings.upstage_api_key,
                )
            ),
        )
        line = (
            json.dumps(
                redact(row, sorted({s for s in secrets if s}, key=len, reverse=True)),
                ensure_ascii=False,
            )
            + "\n"
        )
        target = settings.recording_root / ".cache/diagnostics/gemini.jsonl"
        with _DIAGNOSTIC_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception:
        # Diagnostic disk/format errors must not alter the original API outcome.
        pass


def compatible_schema(original: dict) -> dict:
    """Only the wire schema changes. Unsupported rules remain enforced by Pydantic."""
    definitions = original.get("$defs", {})
    supported = {
        "type",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "properties",
        "additionalProperties",
        "required",
    }

    def visit(node, ancestors=()):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            reference = node["$ref"]
            name = reference.removeprefix("#/$defs/")
            if name not in definitions or name in ancestors:
                raise ConfigurationError(
                    "Unsupported recursive or external output schema reference."
                )
            return visit(
                {**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}},
                (*ancestors, name),
            )
        result = {}
        for key, value in node.items():
            if key not in supported:
                continue
            if key == "properties":
                result[key] = {k: visit(v, ancestors) for k, v in value.items()}
            elif isinstance(value, dict):
                result[key] = visit(value, ancestors)
            elif key in ("anyOf", "prefixItems"):
                result[key] = [visit(v, ancestors) for v in value]
            else:
                result[key] = value
        if "const" in node:
            result["enum"] = [node["const"]]
        if "oneOf" in node:
            result["anyOf"] = [visit(v, ancestors) for v in node["oneOf"]]
        if node.get("format") in ("date", "time", "date-time"):
            result["format"] = node["format"]
        constraints = [
            f"{key}={node[key]}"
            for key in (
                "minLength",
                "maxLength",
                "pattern",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "uniqueItems",
            )
            if key in node
        ]
        if constraints:
            result["description"] = (
                result.get("description", "")
                + " Application also validates: "
                + "; ".join(constraints)
            ).strip()
        return result

    return visit(original)


def provider_error(exc):
    """Never include a raw SDK exception: it may contain credentials or source content."""
    code = getattr(exc, "code", None)
    if isinstance(exc, InvalidStatus):
        code = exc.response.status_code
    if isinstance(exc, ConnectionClosed):
        code = getattr(getattr(exc, "rcvd", None), "code", None)
        code = {1008: 403, 1013: 503, 1011: 503}.get(code)
    if code == 429:
        headers = getattr(getattr(exc, "response", None), "headers", {})
        retry = headers.get("retry-after") if hasattr(headers, "get") else None
        try:
            retry = min(86400, max(0, math.ceil(float(retry))))
        except (ValueError, TypeError, OverflowError):
            retry = None
        raise RateLimitError(
            "Gemini quota or rate limit exceeded; no automatic retry.", retry_after=retry
        ) from None
    if code in (401, 403, 404):
        raise ModelAccessError("Gemini rejected credentials or model access.") from None
    if code in (400, 422):
        raise ProviderRequestError("Gemini rejected the request or output schema.") from None
    if code == 502:
        raise ProviderBadGatewayError("Gemini returned HTTP 502.") from None
    if code == 503:
        raise ProviderServiceUnavailableError("Gemini returned HTTP 503.") from None
    if code == 504:
        raise RequestTimeoutError(
            "Gemini endpoint returned HTTP 504; no automatic retry."
        ) from None
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        raise RequestTimeoutError("Gemini request timed out; no automatic retry.") from None
    if isinstance(code, int) and 400 <= code <= 599:
        raise ProviderError(f"Gemini request failed (HTTP {code}); no automatic retry.") from None
    if isinstance(exc, httpx.NetworkError):
        raise ProviderError("Gemini network connection failed; no automatic retry.") from None
    if isinstance(exc, httpx.ProtocolError):
        raise ProviderError("Gemini network protocol failed; no automatic retry.") from None
    # Class names only, never exception text, URLs, SDK response bodies or source material.
    kind = type(exc).__name__
    if kind not in {
        "ClientError",
        "ServerError",
        "APIError",
        "ValidationError",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "SSLError",
        "ConnectionError",
        "OSError",
    }:
        kind = "unclassified SDK error"
    raise ProviderError(
        f"Gemini request failed ({kind}); no automatic retry or raw error logged."
    ) from None


def sdk_client(api_key, settings, *, live=False):
    return genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options=types.HttpOptions(
            base_url="https://generativelanguage.googleapis.com",
            api_version="v1alpha" if live else "v1beta",
            timeout=int(settings.timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


class SDKClient:
    def __init__(self, provider: str, api_key: str, settings: Settings):
        self.settings = settings
        self._diagnostic_secrets = (api_key,)
        diagnostic_event(settings, "client_initialization_started")
        if not settings.live_enabled:
            raise ConfigurationError("Provider calls are disabled by AI_LIVE_ENABLED.")
        if provider != "gemini":
            raise ConfigurationError("Only Gemini is enabled for text requests.")
        if not api_key.strip():
            raise ConfigurationError("GEMINI_API_KEY or request-scoped X-AI-Key is required.")
        self.client = sdk_client(api_key, settings)
        diagnostic_event(settings, "client_initialized")

    def check_model_access(self, model):
        with diagnostic_scope(
            self.settings, "model_access_check", model, secrets=self._diagnostic_secrets
        ):
            diagnostic_event(self.settings, "model_access_request_started")
            try:
                self.client.models.get(model=model)
            except Exception as exc:
                diagnostic_event(self.settings, "model_access_request_failed", exc=exc)
                provider_error(exc)
            diagnostic_event(self.settings, "model_access_request_completed")

    def generate_json(
        self, *, model: str, prompt: str, schema: type[T], images: list[ImageInput] | None = None,
        thinking_level: str | None = None,
    ) -> T:
        with diagnostic_scope(
            self.settings,
            "structured_generation",
            model,
            secrets=self._diagnostic_secrets,
            credential_source="sdk_argument",
        ):
            return self._generate_json(
                model=model, prompt=prompt, schema=schema, images=images,
                thinking_level=thinking_level,
            )

    def _generate_json(self, *, model, prompt, schema, images, thinking_level=None):
        diagnostic_event(self.settings, "sdk_preparation_started")
        try:
            content = [types.Part.from_text(text=prompt)]
            content += [
                types.Part.from_bytes(data=i.data, mime_type=i.mime_type) for i in images or []
            ]
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=compatible_schema(schema.model_json_schema()),
                max_output_tokens=self.settings.max_output_tokens,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=thinking_level)
                    if thinking_level is not None else None
                ),
            )
        except Exception as exc:
            diagnostic_event(self.settings, "sdk_preparation_failed", exc=exc)
            raise
        diagnostic_event(
            self.settings,
            "sdk_request_started",
            prompt_characters=len(prompt),
            schema_characters=len(json.dumps(config.response_json_schema)),
            max_output_tokens=self.settings.max_output_tokens,
            thinking_level=(
                config.thinking_config.thinking_level.value
                if config.thinking_config is not None else None
            ),
        )
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=content,
                config=config,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            diagnostic_event(self.settings, "sdk_request_failed", exc=exc)
            provider_error(exc)
        diagnostic_event(self.settings, "sdk_response_received", response=response)
        try:
            result = validate_json(completed_text(response), schema)
        except Exception as exc:
            diagnostic_event(self.settings, "output_validation_failed", exc=exc)
            raise
        diagnostic_event(self.settings, "output_schema_validated")
        return result

    def close(self):
        self.client.close()


def completed_text(response):
    candidates = response.candidates or []
    if len(candidates) != 1 or candidates[0].finish_reason != types.FinishReason.STOP:
        raise InvalidResponseError("Gemini output incomplete, blocked or empty.")
    parts = candidates[0].content.parts if candidates[0].content else []
    if any(p.function_call or p.inline_data for p in parts or []):
        raise InvalidResponseError("Gemini returned non-text output parts.")
    return "".join(p.text for p in parts or [] if p.text and not p.thought)


class LiveClient:
    """Translate Gemini Live events into the existing browser protocol, without retries."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def check_configuration(self):
        if not self.settings.gemini_live_enabled:
            raise ConfigurationError("Enable GEMINI_LIVE_ENABLED explicitly for voice calls.")
        if not self.settings.gemini_api_key.get_secret_value().strip():
            raise ConfigurationError("Set GEMINI_API_KEY on the server for voice sessions.")

    def connect_config(self, session):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=session.instructions(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.settings.gemini_live_voice
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=False),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
        )

    async def bridge(self, browser, session):
        self.check_configuration()
        config = self.connect_config(session)
        began = time.perf_counter()
        client = sdk_client(
            self.settings.gemini_api_key.get_secret_value(), self.settings, live=True
        )
        try:
            async with client.aio.live.connect(
                model=self.settings.gemini_live_model, config=config
            ) as provider:
                if provider.setup_complete is None:
                    raise InvalidResponseError("Gemini Live did not acknowledge session setup.")
                await self._stream(browser, provider, session)
        except (InvalidResponseError, InputError, ValueError, KeyError, TypeError):
            session.error_code = "INVALID_OUTPUT"
        except Exception as exc:
            try:
                if isinstance(
                    exc,
                    (
                        RateLimitError,
                        ModelAccessError,
                        ProviderRequestError,
                        ProviderBadGatewayError,
                        ProviderServiceUnavailableError,
                        RequestTimeoutError,
                        ProviderError,
                    ),
                ):
                    raise exc
                provider_error(exc)
            except (
                RateLimitError,
                ModelAccessError,
                ProviderRequestError,
                ProviderBadGatewayError,
                ProviderServiceUnavailableError,
                RequestTimeoutError,
                ProviderError,
            ) as safe:
                session.error_code = {
                    RateLimitError: "RATE_LIMITED",
                    ModelAccessError: "PROVIDER_AUTH",
                    ProviderRequestError: "PROVIDER_REQUEST_INVALID",
                    ProviderBadGatewayError: "PROVIDER_BAD_GATEWAY",
                    ProviderServiceUnavailableError: "PROVIDER_SERVICE_UNAVAILABLE",
                    RequestTimeoutError: "PROVIDER_TIMEOUT",
                }.get(type(safe), "PROVIDER_UNAVAILABLE")
        finally:
            try:
                await client.aio.aclose()
                client.close()
            except Exception:
                session.error_code = session.error_code or "PROVIDER_UNAVAILABLE"
            if session.error_code or not session.provider_finalized:
                session.status = "failed"
                session.provider_finalized = False
                session.error_code = session.error_code or "INCOMPLETE_FINALIZATION"
            from api.recording import record_exchange

            await record_exchange(
                self.settings,
                "teachback_voice",
                str(session.id),
                session.request.model_dump(mode="json"),
                session.result().model_dump(mode="json"),
                status=200 if session.status == "closed" else 502,
                duration_ms=(time.perf_counter() - began) * 1000,
                secrets=[
                    s.get_secret_value()
                    for s in (
                        self.settings.server_api_key,
                        self.settings.gemini_api_key,
                        self.settings.upstage_api_key,
                        self.settings.openai_api_key,
                    )
                    if s.get_secret_value()
                ],
            )

    async def _stream(self, browser, provider, session):
        # Existing browser audio is 24kHz PCM16. Gemini accepts and resamples this rate.
        await browser.send_json({"type": "ready", "sample_rate": 24000})
        started = time.monotonic()
        ending = asyncio.Event()
        boundary = asyncio.Event()
        buffers = {"user": "", "assistant": ""}
        sequence, total_bytes = 0, 0
        has_input = False
        speech_until = 0.0

        async def commit(role):
            nonlocal sequence
            if not buffers[role]:
                return
            sequence += 1
            now = int((time.monotonic() - started) * 1000)
            # Gemini does not promise GPT-Live timestamp/event_id fields. Use local receipt time.
            fragment = session.add_transcript(
                {
                    "type": "session.input_transcript.delta"
                    if role == "user"
                    else "session.output_transcript.delta",
                    "event_id": f"gemini-{sequence}",
                    "delta": buffers[role],
                    "start_ms": now,
                    "end_ms": now,
                }
            )
            buffers[role] = ""
            if fragment:
                await browser.send_json({"type": "transcript", **fragment.model_dump()})

        async def end():
            if not ending.is_set():
                # audio_stream_end flushes cached audio, but does not promise a new
                # turn_complete when the provider already acknowledged an idle turn.
                # New speech/content clears boundary separately; preserve that acknowledgement.
                ending.set()
                await browser.send_json({"type": "playback.clear"})
                await provider.send_realtime_input(audio_stream_end=True)

        async def receive_browser():
            nonlocal total_bytes, has_input, speech_until
            while not ending.is_set():
                message = await browser.receive()
                if message["type"] == "websocket.disconnect":
                    raise ProviderError("Client disconnected before finalization.")
                if message.get("bytes") is not None:
                    audio = message["bytes"]
                    if not audio or len(audio) > 9600 or len(audio) % 2:
                        raise InvalidResponseError("Expected mono PCM16 frames up to 200 ms.")
                    total_bytes += len(audio)
                    if total_bytes > 48000 * (time.monotonic() - started + 2):
                        raise InvalidResponseError("Microphone stream exceeds realtime rate.")
                    samples = struct.unpack(f"<{len(audio) // 2}h", audio)
                    voiced = math.sqrt(sum(s * s for s in samples) / len(samples)) > 820
                    if voiced:
                        now = time.monotonic()
                        if now >= speech_until:
                            await browser.send_json({"type": "playback.clear"})
                        speech_until = now + 0.35
                    if not has_input or voiced:
                        boundary.clear()
                    has_input = True
                    await provider.send_realtime_input(
                        audio=types.Blob(data=audio, mime_type="audio/pcm;rate=24000")
                    )
                else:
                    if json.loads(message.get("text", "{}")) == {"type": "end"}:
                        await end()
                    else:
                        raise InvalidResponseError(
                            "Only PCM audio and the end command are accepted."
                        )

        async def receive_provider():
            interrupted_turn = False
            while True:
                # SDK receive() yields ONE turn. It must be called again for following turns.
                received = False
                async for message in provider.receive():
                    received = True
                    if message.go_away:
                        raise ProviderServiceUnavailableError("Gemini Live session expiring.")
                    if message.tool_call:
                        raise InvalidResponseError("Unexpected Gemini tool request.")
                    if message.usage_metadata:
                        session.usage = message.usage_metadata.model_dump(
                            mode="json", exclude_none=True
                        )
                    content = message.server_content
                    if content is None:
                        continue
                    if not ending.is_set() and (content.model_turn or content.interrupted):
                        boundary.clear()
                    if content.interrupted:
                        interrupted_turn = True
                        await browser.send_json({"type": "playback.clear"})
                    for role, transcription in (
                        ("user", content.input_transcription),
                        ("assistant", content.output_transcription),
                    ):
                        if transcription:
                            if transcription.text and not ending.is_set():
                                boundary.clear()
                            buffers[role] += transcription.text or ""
                            if len(buffers[role]) > 4000:
                                raise InvalidResponseError("Live transcript segment is too large.")
                            if transcription.finished:
                                await commit(role)
                    if not ending.is_set() and not content.interrupted and content.model_turn:
                        for part in content.model_turn.parts or []:
                            if part.inline_data and not part.thought:
                                blob = part.inline_data
                                if not re.fullmatch(
                                    r"audio/pcm(?:;rate=24000)?", blob.mime_type or ""
                                ):
                                    raise InvalidResponseError(
                                        "Unsupported Gemini output audio format."
                                    )
                                if not blob.data or len(blob.data) % 2:
                                    raise InvalidResponseError("Incomplete Gemini PCM16 output.")
                                await browser.send_bytes(blob.data)
                    if content.turn_complete or content.waiting_for_input:
                        if interrupted_turn:
                            # This boundary closes the interrupted AI turn, not the new user turn.
                            await commit("assistant")
                            interrupted_turn = False
                            continue
                        await commit("user")
                        await commit("assistant")
                        boundary.set()
                        if ending.is_set():
                            return
                if not received:
                    raise ProviderError("Gemini Live disconnected before completion.")

        async def finish():
            await ending.wait()
            if has_input:
                # Never treat an input send or a timeout as a provider completion acknowledgement.
                await asyncio.wait_for(boundary.wait(), 15)
            session.provider_finalized = True
            session.status = "closed"

        async def deadline():
            await asyncio.sleep(session.duration)
            await end()

        tasks = [
            asyncio.create_task(f()) for f in (receive_browser, receive_provider, finish, deadline)
        ]
        try:
            while True:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if tasks[2] in done:
                    return
                if tasks[1] in done and not ending.is_set():
                    raise ProviderError("Gemini Live ended unexpectedly.")
                # Browser/deadline complete after sending the end signal; continue draining.
                tasks_to_wait = [tasks[1], tasks[2]]
                done, _ = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if tasks[2] not in done:
                    await tasks[2]
                return
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
