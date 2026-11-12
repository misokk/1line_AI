from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.errors import InvalidResponseError

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, repr=False)
class ImageInput:
    data: bytes
    mime_type: str


class LLMClient(Protocol):
    def generate_json(
        self, *, model: str, prompt: str, schema: type[T], images: list[ImageInput] | None = None
    ) -> T: ...


def validate_json(raw: str | None, schema: type[T]) -> T:
    if not raw or not raw.strip():
        raise InvalidResponseError("Provider returned no usable text (blocked or empty response).")
    try:
        return schema.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False, include_context=False)
        if any(error["type"] == "json_invalid" for error in errors):
            raise InvalidResponseError("Provider returned malformed JSON (json_invalid).") from None
        # Only schema-declared field names may appear in diagnostics. Never include
        # raw values, arbitrary extra JSON keys, validator context or source text.
        fields = set()

        def collect(node):
            if isinstance(node, dict):
                fields.update(node.get("properties", {}))
                for value in node.values():
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(schema.model_json_schema())
        details = []
        for error in errors[:5]:
            path = (
                ".".join(
                    str(part) if isinstance(part, int) or part in fields else "<field>"
                    for part in error["loc"]
                )
                or "response"
            )
            details.append(f"{path}: {error['type']}")
        raise InvalidResponseError(
            "Provider JSON failed schema validation: " + "; ".join(details)
        ) from None
    except ValueError:
        raise InvalidResponseError("Provider JSON could not be validated for the schema.") from None
