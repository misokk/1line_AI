"""Opt-in development ASGI recorder. Never stores headers, binary bodies or voice streams."""

import asyncio
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs
from uuid import uuid4

from api.config import Settings

MAX_JSON_BYTES = 32 * 1024 * 1024
WRITE_LOCK = Lock()
FEATURE_NAMES = {
    "health": "health",
    "internal_models": "models",
    "internal_models_validate-key": "validate_key",
    "internal_materials_extract": "extract",
    "internal_materials_analyze": "analysis",
    "internal_explanations_generate": "explanation",
    "internal_questions_generate": "questions_generate",
    "internal_questions_grade": "questions_grade",
    "internal_evaluations_generate": "evaluation",
    "internal_evaluations_detailed": "evaluation_detailed",
    "internal_keywords_suggest": "keywords",
    "internal_study_time_estimate": "study_time",
    "internal_study_plan_generate": "study_plan",
    "internal_teachback_sessions": "teachback_session_create",
    "internal_teachback_session_result": "teachback_session",
    "internal_teachback_session_evaluate": "teachback_evaluation",
    "internal_teachback_session_evaluate_detailed": "teachback_evaluation_detailed",
    "teachback_voice": "teachback_voice",
}
SENSITIVE = {
    "key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "cookie",
    "setcookie",
    "xaikey",
    "xaiapikey",
    "credential",
    "credentials",
    "audio",
    "audiodata",
    "audiobase64",
    "audiochunk",
    "audiostream",
    "rawaudio",
    "voicestream",
    "pcm",
    "pcmdata",
}
TOKEN_PATTERN = re.compile(
    r"AIza[\w-]{25,}|gsk_[\w-]{20,}|\bup_[\w-]{20,}|"
    r"github_pat_[\w]{20,}|gh[pousr]_[\w]{20,}|sk-[\w-]{20,}|"
    r"\beyJ[\w-]+\.[\w-]+\.[\w-]+|\b(?:Bearer|Basic)\s+[^\s\"<>]+|"
    r"data:audio/[^;\s]+;base64,[A-Za-z0-9+/=]+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.IGNORECASE,
)


def redact(value, secrets):
    def text(s):
        for secret in secrets:
            s = s.replace(secret, "[REDACTED]")
        return TOKEN_PATTERN.sub("[REDACTED]", s)

    if isinstance(value, str):
        return text(value)
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            sensitive = normalized in SENSITIVE or normalized.endswith(
                ("key", "secret", "password", "token", "credentials")
            )
            result[text(key)] = "[REDACTED]" if sensitive else redact(item, secrets)
        return result
    return value


def save_pair(root: Path, name: str, incoming, outgoing, secrets):
    """Replace each feature's latest pair; concurrent writes cannot mix two requests."""
    pending = []
    with WRITE_LOCK:
        try:
            bodies = [
                json.dumps(redact(v, secrets), ensure_ascii=False, indent=2)
                for v in (incoming, outgoing)
            ]
            for directory, body in zip(("input", "output"), bodies, strict=True):
                folder = root / directory
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                target = folder / f"{name}_{directory}.json"
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False
                ) as handle:
                    pending.append((Path(handle.name), target))
                    handle.write(body)
            for temporary, target in pending:
                temporary.replace(target)
        except Exception:
            # Do not log exception messages: paths and payloads may contain private data.
            pass
        finally:
            for temporary, _ in pending:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


PRIVATE_TEXT_FIELDS = {
    "text",
    "content",
    "delta",
    "excerpt",
    "summary",
    "criterion",
    "prompt",
    "title",
    "name",
    "description",
    "feedback",
    "explanation",
    "explanationmarkdown",
    "learnerquote",
    "correctexplanation",
    "improvement",
    "nextaction",
    "checklistitems",
    "acceptedanswers",
    "correctanswer",
    "selectedoptionids",
    "filename",
    "materialname",
    "keywords",
    "definition",
    "rules",
    "cautions",
    "examples",
}


def minimize_content(value, field=""):
    """Keep structure and diagnostic counts, omit private learning content by default."""
    if isinstance(value, str) and field in PRIVATE_TEXT_FIELDS:
        return {"redacted": True, "characters": len(value)}
    if isinstance(value, list):
        return [minimize_content(v, field) for v in value]
    if isinstance(value, dict):
        return {
            k: minimize_content(v, re.sub(r"[^a-z0-9]", "", k.lower())) for k, v in value.items()
        }
    return value


async def record_exchange(
    settings,
    feature,
    request_id,
    incoming,
    outgoing,
    *,
    status,
    duration_ms,
    method="WS",
    secrets=(),
):
    if settings.environment != "development" or not settings.record_io:
        return
    try:
        if not settings.record_content:
            incoming, outgoing = minimize_content(incoming), minimize_content(outgoing)
        await asyncio.to_thread(
            save_pair,
            settings.recording_root,
            FEATURE_NAMES.get(feature, "unknown"),
            incoming,
            outgoing,
            secrets,
        )
    except Exception:
        pass


class IORecordingMiddleware:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.enabled = settings.environment == "development" and settings.record_io
        self.root = settings.recording_root
        self.settings = settings
        self.secrets = [
            v.get_secret_value()
            for v in (
                settings.server_api_key,
                settings.upstage_api_key,
                settings.openai_api_key,
                settings.gemini_api_key,
            )
            if v.get_secret_value()
        ]

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            not self.enabled
            or scope["type"] != "http"
            or not (path == "/health" or path.startswith("/internal/"))
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        started = time.perf_counter()
        request_id = uuid4().hex
        secrets = list(self.secrets)
        for key, raw_value in headers.items():
            name = re.sub(r"[^a-z0-9]", "", key.decode("latin-1").lower())
            if not name.endswith(("key", "token", "authorization", "cookie")):
                continue
            value = raw_value.decode("latin-1")
            if value:
                secrets.append(value)
                if value.lower().startswith(("bearer ", "basic ")):
                    secrets.append(value.split(" ", 1)[1])
        secrets = sorted(set(secrets), key=len, reverse=True)
        mime = headers.get(b"content-type", b"").decode("latin-1").split(";")[0].lower()
        request_json = mime == "application/json" or mime.endswith("+json")
        incoming, outgoing = bytearray(), bytearray()
        digest = hashlib.sha256()
        size, output_size = 0, 0
        complete, response_json = False, False
        status, failure = None, None

        async def observed_receive():
            nonlocal size, complete
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                size += len(body)
                digest.update(body)
                if request_json and len(incoming) + len(body) <= MAX_JSON_BYTES:
                    incoming.extend(body)
                complete = not message.get("more_body", False)
            return message

        async def observed_send(message):
            nonlocal status, response_json, output_size
            if message["type"] == "http.response.start":
                status = message["status"]
                content_type = dict(message.get("headers", [])).get(b"content-type", b"")
                response_json = content_type.split(b";")[0].endswith((b"/json", b"+json"))
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                output_size += len(body)
                if response_json and len(outgoing) + len(body) <= MAX_JSON_BYTES:
                    outgoing.extend(body)
            await send(message)

        def decode(data, total):
            if len(data) == total:
                try:
                    return json.loads(data)
                except (ValueError, UnicodeError):
                    pass
            return {"recording": "body_not_available_as_json", "size_bytes": total}

        try:
            await self.app(scope, observed_receive, observed_send)
        except Exception as exc:
            failure = {"error": {"code": "UNHANDLED_EXCEPTION", "type": type(exc).__name__}}
            raise
        finally:
            try:
                query = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="replace"))
                if request_json:
                    request_data = decode(incoming, size)
                elif not mime and not size:
                    request_data = None
                else:
                    request_data = {
                        "filename": query.get("filename", [None])[0],
                        "size_bytes": size,
                        "mime_type": mime,
                        "sha256": digest.hexdigest(),
                        "body_complete": complete,
                        "query": {
                            k: v[0]
                            for k, v in query.items()
                            if k
                            in {
                                "filename",
                                "material_id",
                                "retry_failed",
                            }
                        },
                    }
                response_data = failure or (
                    decode(outgoing, output_size)
                    if response_json
                    else {
                        "recording": "non_json_response",
                        "status": status,
                        "size_bytes": output_size,
                    }
                )
                # API paths are fixed; an arbitrary URL segment must never become a filename.
                known = {
                    "/health",
                    "/internal/models",
                    "/internal/models/validate-key",
                    "/internal/materials/extract",
                    "/internal/materials/analyze",
                    "/internal/questions/generate",
                    "/internal/questions/grade",
                    "/internal/teachback/sessions",
                    "/internal/evaluations/generate",
                    "/internal/keywords/suggest",
                }
                feature = path.strip("/").replace("/", "_") if path in known else "unknown"
                if re.fullmatch(r"/internal/teachback/sessions/[0-9a-fA-F-]{36}", path):
                    feature = "internal_teachback_session_result"
                elif re.fullmatch(r"/internal/teachback/sessions/[0-9a-fA-F-]{36}/evaluate", path):
                    feature = "internal_teachback_session_evaluate"
                elif re.fullmatch(
                    r"/internal/teachback/sessions/[0-9a-fA-F-]{36}/evaluate-detailed", path
                ):
                    feature = "internal_teachback_session_evaluate_detailed"
                if path == "/internal/explanations/generate":
                    feature = "internal_explanations_generate"
                elif path == "/internal/evaluations/detailed":
                    feature = "internal_evaluations_detailed"
                elif path == "/internal/study-time/estimate":
                    feature = "internal_study_time_estimate"
                elif path == "/internal/study-plan/generate":
                    feature = "internal_study_plan_generate"
                await record_exchange(
                    self.settings,
                    feature,
                    request_id,
                    request_data,
                    response_data,
                    status=status or 500,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    method=scope.get("method", "HTTP"),
                    secrets=secrets,
                )
            except Exception:
                pass
