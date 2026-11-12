from collections.abc import Mapping
from contextlib import contextmanager

from api.base import LLMClient, T
from api.config import Role, Settings
from api.gemini import SDKClient, diagnostic_event
from src.errors import ConfigurationError, ModelAccessError, ModelSelectionError


class ModelRouter:
    def __init__(self, settings: Settings, clients: Mapping[str, LLMClient]) -> None:
        self.settings = settings
        self.clients = dict(clients)

    def output_budget(self, role: Role | str, model_id: str | None = None) -> int | None:
        self.settings.target_for(Role(role), model_id)
        return self.settings.max_output_tokens

    def check_call_count(self, role: Role | str, model_id: str | None, count: int):
        target = self.settings.target_for(Role(role), model_id)
        client = self.clients.get(target.provider)
        limit = getattr(client, "request_limit", None)
        sent = getattr(client, "requests_sent", None)
        if type(limit) is int and type(sent) is int:
            if sent + count > limit:
                raise ConfigurationError(
                    "Split operation exceeds the explicit provider call budget."
                )

    def generate_json(
        self,
        role: Role | str,
        *,
        prompt: str,
        schema: type[T],
        model_id: str | None = None,
        images=None,
    ) -> T:
        try:
            target = self.settings.target_for(Role(role), model_id)
        except ValueError:
            raise ConfigurationError("Unknown AI role.") from None
        client = self.clients.get(target.provider)
        if client is None:
            raise ConfigurationError("Provider is not registered or its optional SDK is missing.")
        if images and not target.supports_images:
            raise ConfigurationError("Configured model is not enabled for image input.")
        kwargs = {"model": target.model, "prompt": prompt, "schema": schema}
        if images:
            kwargs["images"] = images
        if target.provider == "gemini":
            diagnostic_event(self.settings, "feature_preflight_passed")
            if (
                Role(role) == Role.QUESTION_GENERATION
                and target.model in {"gemini-3.7-flash", "gemini-3.8-flash"}
                and isinstance(client, SDKClient)
            ):
                return client.generate_json(**kwargs, thinking_level="low")
        return client.generate_json(**kwargs)


def build_router(settings: Settings) -> ModelRouter:
    """Local development only. HTTP requests use request_router and a request-scoped BYOK."""
    return ModelRouter(
        settings,
        {"gemini": SDKClient("gemini", settings.gemini_api_key.get_secret_value(), settings)},
    )


@contextmanager
def request_router(settings: Settings, model_id: str, api_key: str | None):
    target = settings.target_for(Role.MATERIAL_ANALYSIS, model_id)
    if target.provider != "gemini":
        raise ModelSelectionError("Only Gemini is active for HTTP text requests.")
    if not api_key and target.provider == "gemini" and settings.use_server_text_key:
        api_key = settings.gemini_api_key.get_secret_value()
    if not api_key or not api_key.strip() or len(api_key) > 512:
        raise ModelAccessError("X-AI-Key is required for provider calls.")
    # Server-owned keys require an explicit setting; BYOK remains the default contract.
    local = settings
    client = SDKClient(target.provider, api_key, local)
    try:
        yield ModelRouter(local, {target.provider: client})
    finally:
        client.close()
