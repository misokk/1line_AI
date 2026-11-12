from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from api.base import ImageInput, validate_json
from api.config import ModelTarget, Role, Settings
from api.openai import SDKClient
from api.router import ModelRouter
from src.errors import ConfigurationError, InvalidResponseError, ModelSelectionError


class Response(BaseModel):
    text: str


@pytest.mark.parametrize("provider", ["openai"])
def test_optional_provider_lazily_imports_and_does_not_install_sdk(provider):
    with patch("api.openai.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ConfigurationError, match="optional"):
            SDKClient(provider, "fake-key", Settings(live_enabled=True))


@pytest.mark.parametrize("provider", ["openai"])
def test_optional_sdk_native_schema_mapping_and_fixed_endpoint(provider):
    sdk = MagicMock()
    factory = sdk.OpenAI
    client = factory.return_value
    client.responses.parse.return_value = SimpleNamespace(
        status="completed", output_text='{"text":"ok"}'
    )
    with patch("api.openai.importlib.import_module", return_value=sdk):
        adapter = SDKClient(provider, "fake-key", Settings(live_enabled=True))
        result = adapter.generate_json(
            model="fixture-model",
            prompt="untrusted source",
            schema=Response,
            images=[ImageInput(b"fake-image", "image/png")],
        )
        assert result.text == "ok"
        kwargs = factory.call_args.kwargs
        assert kwargs["max_retries"] == 0 and kwargs["base_url"].startswith("https://api.")
        assert client.responses.parse.call_args.kwargs["text_format"] is Response
        assert client.responses.parse.call_args.kwargs["store"] is False
        adapter.close()
        client.close.assert_called_once()


@pytest.mark.parametrize("provider", ["openai"])
def test_role_selection_is_provider_independent(provider):
    model_id = f"{provider}:fixture-model"
    settings = Settings(
        model_allowlist={
            model_id: ModelTarget(provider=provider, model="fixture-model", available=True)
        },
        routes={Role.EVALUATION: model_id},
    )
    fake = MagicMock()
    fake.generate_json.return_value = Response(text="ok")
    assert (
        ModelRouter(settings, {provider: fake})
        .generate_json(
            "evaluation",
            prompt="input",
            schema=Response,
        )
        .text
        == "ok"
    )
    with pytest.raises(ModelSelectionError):
        settings.target_for(Role.EVALUATION, "other-provider:invented")


def test_cannot_pass_custom_provider_endpoint():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelTarget(
            provider="openai", model="fixture", available=True, base_url="https://attacker.invalid"
        )


def test_json_error_distinguished_without_echoing_private_content():
    with pytest.raises(InvalidResponseError, match="json_invalid") as error:
        validate_json("private-source-text is not JSON", Response)
    assert "private-source-text" not in str(error.value)


def test_schema_diagnostics_show_field_type_without_value():
    with pytest.raises(InvalidResponseError) as error:
        validate_json('{"text":["private-secret-value"]}', Response)
    assert "text: string_type" in str(error.value)
    assert "private-secret-value" not in str(error.value)


def test_unknown_json_property_name_is_redacted():
    from src.models import Keywords

    with pytest.raises(InvalidResponseError) as error:
        validate_json('{"question":"Example?","private-key-in-field-name":"hidden"}', Keywords)
    assert "extra_forbidden" in str(error.value)
    assert "private-key-in-field-name" not in str(error.value) and "hidden" not in str(error.value)


def test_question_semantic_schema_failure_has_specific_safe_code(stored):
    import json

    from src.models import QuestionBatch

    data = stored.model_dump(mode="json")
    data["question"]["question_type"] = "short_answer"
    with pytest.raises(InvalidResponseError) as error:
        validate_json(json.dumps({"items": [data]}), QuestionBatch)
    assert "items.0.question: question_short_answer_options_not_empty" in str(error.value)
