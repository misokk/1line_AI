"""One synthetic generation per explicit --live invocation; default is offline preparation."""

import argparse
import json
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types

from api.config import Settings
from api.gemini import (
    SDKClient,
    completed_text,
    diagnostic_event,
    diagnostic_scope,
    provider_error,
)
from api.recording import redact
from src.errors import AIError, InvalidResponseError
from src.models import Model


class ConnectionResult(Model):
    status: Literal["ok"]


def run(*, mode="text", live=False, settings=None, model_id=None):
    if mode not in ("text", "json"):
        raise ValueError("Probe mode must be text or json.")
    settings = settings or Settings.from_env(Path(".env"))
    selected = model_id or settings.default_model_id
    target = settings.target_for("explanation", selected)
    report = {
        "mode": mode,
        "model_id": selected,
        "sdk_version": genai.__version__,
        "timeout_seconds": settings.timeout_seconds,
        "max_output_tokens": 512,
        "key_configured": bool(settings.gemini_api_key.get_secret_value()),
        "live_requested": live,
        "sdk_calls": 0,
        "success": False,
        "status": "not_run",
    }
    if not live:
        return report
    client = None
    config = settings.model_copy(update={"max_output_tokens": 512})
    with diagnostic_scope(
        config, f"connection_probe_{mode}", selected, credential_source="server_environment"
    ):
        diagnostic_event(config, "probe_started")
        try:
            client = SDKClient(target.provider, config.gemini_api_key.get_secret_value(), config)
            if mode == "json":
                report["sdk_calls"] = 1
                client.generate_json(
                    model=target.model,
                    prompt='Return JSON with status equal to "ok".',
                    schema=ConnectionResult,
                )
            else:
                diagnostic_event(config, "sdk_request_started", max_output_tokens=512)
                report["sdk_calls"] = 1
                try:
                    response = client.client.models.generate_content(
                        model=target.model,
                        contents="한국어로 연결 확인이라고만 답하세요.",
                        config=types.GenerateContentConfig(max_output_tokens=512),
                    )
                except Exception as exc:
                    diagnostic_event(config, "sdk_request_failed", exc=exc)
                    provider_error(exc)
                diagnostic_event(config, "sdk_response_received")
                if not completed_text(response).strip():
                    raise InvalidResponseError("Probe returned no text.")
            report.update(status="passed", success=True)
            diagnostic_event(config, "probe_completed")
        except Exception as exc:
            diagnostic_event(config, "probe_failed", exc=exc)
            report.update(
                status="failed",
                safe_error=str(exc)
                if isinstance(exc, AIError)
                else "Probe failed locally; inspect safe diagnostics.",
            )
        finally:
            if client is not None:
                client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "json"), default="text")
    parser.add_argument("--model-id", help="Use the same ID as the failing request; no fallback.")
    parser.add_argument("--live", action="store_true", help="Requires prior user approval.")
    args = parser.parse_args()
    settings = Settings.from_env(Path(".env"))
    result = run(mode=args.mode, live=args.live, settings=settings, model_id=args.model_id)
    secrets = [
        s.get_secret_value()
        for s in (
            settings.gemini_api_key,
            settings.server_api_key,
            settings.openai_api_key,
            settings.upstage_api_key,
        )
        if s.get_secret_value()
    ]
    print(json.dumps(redact(result, secrets), ensure_ascii=False, indent=2))
    return 1 if args.live and not result["success"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
