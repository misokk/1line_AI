class AIError(Exception):
    """Base exception with safe messages suitable for application logs."""


class InputError(AIError):
    """Invalid domain input or unsupported PDF."""


class ModelSelectionError(InputError):
    """Requested model is not in the active deployment allowlist."""


class ModelRequiredError(AIError):
    """Semantic grading requires a model, unlike exact answer comparison."""


class KeyValidationUnavailableError(AIError):
    """A safe non-generating provider key validation is unavailable."""


class ConfigurationError(AIError):
    """Missing or invalid runtime configuration."""


class ProviderError(AIError):
    """Upstream service or transport failure."""


class ProviderBadGatewayError(ProviderError):
    """Provider returned HTTP 502."""


class ProviderServiceUnavailableError(ProviderError):
    """Provider returned HTTP 503."""


class ProviderRequestError(ProviderError):
    """Provider rejected the generated request or its supported parameters."""


class RateLimitError(ProviderError):
    """Provider quota or request limit exceeded."""

    def __init__(self, message: str, *, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelAccessError(ProviderError):
    """Provider rejected the model, credentials or request configuration."""


class RequestTimeoutError(ProviderError):
    """Provider request exceeded its timeout."""


class InvalidResponseError(AIError):
    """Provider output failed schema or domain validation."""


class DocumentProcessingStateError(AIError):
    """Duplicate, interrupted or uncertain parse request requires backend action."""
