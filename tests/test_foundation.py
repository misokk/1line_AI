from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from api.base import validate_json
from api.config import ModelTarget, Role, Settings
from api.gemini import SDKClient
from api.router import ModelRouter, build_router
from src.errors import (
    ConfigurationError,
    InvalidResponseError,
    ModelSelectionError,
)


class SmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str


@pytest.mark.parametrize("role", list(Role))
def test_all_roles_offline(role):
    with patch("api.gemini.genai.Client", side_effect=AssertionError("network forbidden")):
        with pytest.raises((ConfigurationError, ModelSelectionError)):
            build_router(Settings()).generate_json(role, prompt="test", schema=SmokeResponse)


def test_role_override_preserves_schema():
    settings = Settings(
        model_allowlist={
            "gemini:future": ModelTarget(provider="gemini", model="future", available=True),
            "gemini:fixture-model": ModelTarget(
                provider="gemini", model="fixture-model", available=True
            ),
        },
        routes={Role.EVALUATION: "gemini:future"},
        default_model_id="gemini:fixture-model",
    )
    fake = MagicMock()
    fake.generate_json.return_value = SmokeResponse(message="evaluated")
    router = ModelRouter(settings, {"gemini": fake})
    assert (
        router.generate_json("evaluation", prompt="test", schema=SmokeResponse).message
        == "evaluated"
    )
    fake.generate_json.assert_called_once_with(model="future", prompt="test", schema=SmokeResponse)
    assert settings.target_for(Role.MATERIAL_ANALYSIS).provider == "gemini"


@pytest.mark.parametrize("raw", ["", "not json", "[]", '{"message":3}', '{"wrong":"secret"}'])
def test_invalid_response(raw):
    with pytest.raises(InvalidResponseError) as exc:
        validate_json(raw, SmokeResponse)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    "settings",
    [
        Settings(),
        Settings(live_enabled=True),
        Settings(gemini_api_key=SecretStr("fake-openai-secret"), live_enabled=False),
    ],
)
def test_live_guards(settings):
    with patch("api.gemini.genai.Client") as sdk:
        with pytest.raises((ConfigurationError, ModelSelectionError)):
            SDKClient("gemini", settings.gemini_api_key.get_secret_value(), settings).generate_json(
                model="x", prompt="test", schema=SmokeResponse
            )
        sdk.assert_not_called()


def test_openai_requires_registered_implementation():
    settings = Settings(
        model_allowlist={
            "gemini:future": ModelTarget(provider="gemini", model="future", available=True),
            "gemini:fixture-model": ModelTarget(
                provider="gemini", model="fixture-model", available=True
            ),
        },
        routes={Role.EVALUATION: "gemini:future"},
        default_model_id="gemini:fixture-model",
    )
    with pytest.raises(ConfigurationError, match="not registered"):
        ModelRouter(settings, {}).generate_json("evaluation", prompt="test", schema=SmokeResponse)


def test_bad_route_and_missing_provider():
    with pytest.raises((ConfigurationError, ModelSelectionError)):
        build_router(Settings()).generate_json("unknown", prompt="test", schema=SmokeResponse)
    with pytest.raises((ConfigurationError, ModelSelectionError)):
        ModelRouter(Settings(), {}).generate_json("evaluation", prompt="test", schema=SmokeResponse)


@pytest.mark.parametrize("provider", ["anthropic", "groq"])
def test_removed_providers_cannot_enter_model_catalog(provider):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelTarget(provider=provider, model="removed-model", available=True)


def test_env_precedence_and_secret_redaction(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=local-secret\nGEMINI_TEXT_MODEL=file-model\n")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "env-model")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings.from_env(env_file)
    assert settings.gemini_text_model == "env-model"
    assert settings.gemini_api_key.get_secret_value() == "local-secret"
    assert "local-secret" not in repr(settings)
    monkeypatch.setenv("AI_MODEL_ROUTES", '{"bad-role":{"provider":"bad","model":"secret"}}')
    with pytest.raises(ConfigurationError) as exc:
        Settings.from_env(env_file)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("seconds", [120, 300])
def test_longer_provider_timeout_from_env(tmp_path, monkeypatch, seconds):
    monkeypatch.delenv("AI_TIMEOUT_SECONDS", raising=False)
    path = tmp_path / "timeout.env"
    path.write_text(f"AI_TIMEOUT_SECONDS={seconds}\n", encoding="utf-8")
    assert Settings.from_env(path).timeout_seconds == seconds


@pytest.mark.parametrize("seconds", [0, 301])
def test_provider_timeout_remains_bounded(seconds):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(timeout_seconds=seconds)


def test_unlisted_gemini_models_require_explicit_opt_in():
    settings = Settings()
    with pytest.raises(ModelSelectionError):
        settings.target_for(Role.EXPLANATION, "gemini:some-future-model")
    enabled = settings.model_copy(update={"allow_unlisted_models": True})
    target = enabled.target_for(Role.EXPLANATION, "gemini:some-future-model")
    assert target.provider == "gemini" and target.model == "some-future-model"
    assert target.available and not target.supports_images


@pytest.mark.parametrize(
    "model_id",
    [
        "openai:gpt-5.4-mini",
        "https://attacker.invalid/model",
        "gemini:../models/x",
        "gemini:models/x",
        "gemini:",
        "gemini:*",
        "gemini:model?key=private",
        "gemini:x\ny",
    ],
)
def test_unlisted_models_cannot_change_provider_or_endpoint(model_id):
    with pytest.raises(ModelSelectionError):
        Settings(allow_unlisted_models=True).target_for(Role.EXPLANATION, model_id)


def test_disabled_catalog_model_stays_disabled_in_open_mode():
    settings = Settings(
        allow_unlisted_models=True,
        model_allowlist={
            "gemini:disabled": ModelTarget(provider="gemini", model="disabled", available=False)
        },
    )
    with pytest.raises(ModelSelectionError):
        settings.target_for(Role.EXPLANATION, "gemini:disabled")


def test_unlisted_model_flag_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_ALLOW_UNLISTED_MODELS", raising=False)
    path = tmp_path / "models.env"
    path.write_text("AI_ALLOW_UNLISTED_MODELS=true\n", encoding="utf-8")
    assert Settings.from_env(path).allow_unlisted_models
