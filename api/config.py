import json
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from src.errors import ConfigurationError, ModelSelectionError


class Role(StrEnum):
    MATERIAL_ANALYSIS = "material_analysis"
    QUESTION_GENERATION = "question_generation"
    EVALUATION = "evaluation"
    QUESTION_TAGGING = "question_tagging"
    EXPLANATION = "explanation"


class ModelTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Literal["gemini", "openai"]
    model: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
    available: bool = False
    supports_images: bool = False


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    live_enabled: bool = False
    server_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    upstage_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    gemini_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    gemini_text_model: str = Field(
        default="gemini-3.8-flash", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$"
    )
    gemini_live_enabled: bool = False
    gemini_live_model: str = Field(default="gemini-3.8-live", pattern=r"^gemini-[a-zA-Z0-9.-]+$")
    gemini_live_voice: str = Field(default="Aoede", pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    # Retained for rollback tests and secret redaction; HTTP runtime only uses Gemini.
    openai_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    use_server_text_key: bool = False
    openai_text_model: str = Field(default="gpt-5.4-mini", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    question_batch_size: int = Field(default=5, ge=1, le=20)
    evaluation_batch_size: int = Field(default=8, ge=1, le=50)
    openai_live_enabled: bool = False
    openai_live_model: str = Field(default="gpt-live-1", pattern=r"^gpt-live-[a-zA-Z0-9.-]+$")
    openai_live_voice: str = Field(default="marin", pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    voice_max_seconds: int = Field(default=300, ge=30, le=900)
    upstage_model: str = Field(
        default="document-parse", pattern=r"^document-parse(?:-[a-zA-Z0-9]+)?$"
    )
    upstage_timeout_seconds: float = Field(default=120, gt=0, le=300)
    document_cache_dir: Path = Path(".cache/document-parse")
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    model_allowlist: dict[str, ModelTarget] = Field(default_factory=dict, max_length=100)
    allow_unlisted_models: bool = False
    routes: dict[Role, str] = Field(default_factory=dict)
    default_model_id: str | None = None
    environment: Literal["development", "production", "test"] = "production"
    record_io: bool = False
    record_content: bool = False
    recording_root: Path = Path(".")

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        values = dict(dotenv_values(env_file)) if env_file is not None else {}
        values.update(os.environ)
        try:
            return cls(
                environment=values.get("AI_ENV", "production"),
                record_io=values.get("AI_RECORD_IO", "false"),
                record_content=values.get("AI_RECORD_CONTENT", "false"),
                recording_root=values.get("AI_RECORDING_ROOT") or ".",
                live_enabled=values.get("AI_LIVE_ENABLED", "false"),
                server_api_key=values.get("AI_SERVER_API_KEY") or "",
                upstage_api_key=values.get("UPSTAGE_API_KEY") or "",
                gemini_api_key=values.get("GEMINI_API_KEY") or "",
                gemini_text_model=values.get("GEMINI_TEXT_MODEL") or "gemini-3.8-flash",
                gemini_live_enabled=values.get("GEMINI_LIVE_ENABLED", "false"),
                gemini_live_model=values.get("GEMINI_LIVE_MODEL") or "gemini-3.8-live",
                gemini_live_voice=values.get("GEMINI_LIVE_VOICE") or "Aoede",
                openai_api_key=values.get("OPENAI_API_KEY") or "",
                use_server_text_key=values.get("AI_USE_SERVER_TEXT_KEY", "false"),
                openai_text_model=values.get("OPENAI_TEXT_MODEL") or "gpt-5.4-mini",
                question_batch_size=values.get("AI_QUESTION_BATCH_SIZE", "5"),
                evaluation_batch_size=values.get("AI_EVALUATION_BATCH_SIZE", "8"),
                openai_live_enabled=values.get("OPENAI_LIVE_ENABLED", "false"),
                openai_live_model=values.get("OPENAI_LIVE_MODEL") or "gpt-live-1",
                openai_live_voice=values.get("OPENAI_LIVE_VOICE") or "marin",
                voice_max_seconds=values.get("VOICE_MAX_SECONDS", "300"),
                upstage_model=values.get("UPSTAGE_DOCUMENT_MODEL") or "document-parse",
                upstage_timeout_seconds=values.get("UPSTAGE_TIMEOUT_SECONDS", "120"),
                document_cache_dir=values.get("DOCUMENT_CACHE_DIR") or ".cache/document-parse",
                timeout_seconds=values.get("AI_TIMEOUT_SECONDS", "60"),
                max_output_tokens=values.get("AI_MAX_OUTPUT_TOKENS", "8192"),
                allow_unlisted_models=values.get("AI_ALLOW_UNLISTED_MODELS", "false"),
                model_allowlist=json.loads(
                    values.get("AI_MODEL_ALLOWLIST")
                    or json.dumps(
                        {
                            "gemini:" + (values.get("GEMINI_TEXT_MODEL") or "gemini-3.8-flash"): {
                                "provider": "gemini",
                                "model": values.get("GEMINI_TEXT_MODEL") or "gemini-3.8-flash",
                                "available": True,
                            }
                        }
                    )
                ),
                routes=json.loads(values.get("AI_MODEL_ROUTES") or "{}"),
                default_model_id=values.get("AI_DEFAULT_MODEL_ID")
                or "gemini:" + (values.get("GEMINI_TEXT_MODEL") or "gemini-3.8-flash"),
            )
        except (ValidationError, ValueError, TypeError):
            raise ConfigurationError(
                "Invalid AI configuration; check environment settings."
            ) from None

    def target_for(self, role: Role, model_id: str | None = None) -> ModelTarget:
        selected = model_id or self.routes.get(role) or self.default_model_id
        target = self.model_allowlist.get(selected or "")
        if target is None and self.allow_unlisted_models and isinstance(selected, str):
            provider, separator, model = selected.partition(":")
            if (
                provider == "gemini"
                and separator
                and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", model)
            ):
                return ModelTarget(provider="gemini", model=model, available=True)
        if target is None or not target.available:
            raise ModelSelectionError("model_id is not active in the configured backend allowlist.")
        if not any(selected.startswith(target.provider + separator) for separator in (":", "/")):
            raise ModelSelectionError(
                "model_id must be provider-qualified in the backend allowlist."
            )
        return target
