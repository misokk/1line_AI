import json
from unittest.mock import patch

import httpx
import pymupdf
import pytest

from api.upstage import UpstageDocumentParser, normalize_response
from src.errors import (
    ConfigurationError,
    DocumentProcessingStateError,
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


def test_official_multipart_request_and_snapshot(settings, pdf_data, parse_payload):
    seen = []

    def handle(request):
        seen.append(request)
        assert str(request.url) == "https://api.upstage.ai/v1/document-digitization"
        assert request.headers["Authorization"] == "Bearer fake-upstage-key"
        body = request.content.decode("latin1")
        for name, value in [
            ("model", "document-parse"),
            ("mode", "enhanced"),
            ("ocr", "force"),
            ("output_formats", json.dumps(["markdown", "html", "text"])),
        ]:
            assert f'name="{name}"\r\n\r\n{value}' in body
        assert 'name="document"; filename="material.pdf"' in body
        return httpx.Response(200, json=parse_payload(2))

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    result = parser.parse(pdf_data, "application/pdf")
    assert result.actual_models == ["document-parse-260930"] and result.estimated_cost_usd == 0.06
    assert len(seen) == 1
    again = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle)).parse(
        pdf_data, "application/pdf"
    )
    assert again.cache_hit and again.requests_sent == 0 and len(seen) == 1


@pytest.mark.parametrize(
    ("code", "error"),
    [
        (401, ModelAccessError),
        (403, ModelAccessError),
        (429, RateLimitError),
        (502, ProviderBadGatewayError),
        (503, ProviderServiceUnavailableError),
        (504, RequestTimeoutError),
        (400, ProviderRequestError),
        (413, ProviderRequestError),
        (500, ProviderError),
    ],
)
def test_safe_errors_and_no_automatic_repeat(settings, pdf_data, code, error):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(code, json={"secret": "must-not-echo"})

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    with pytest.raises(error) as exc:
        parser.parse(pdf_data, "application/pdf")
    assert "must-not-echo" not in str(exc.value)
    with pytest.raises(DocumentProcessingStateError):
        parser.parse(pdf_data, "application/pdf")
    assert len(seen) == 1
    attempt = next(settings.document_cache_dir.glob("*/attempt-0.json"))
    state = json.loads(attempt.read_text(encoding="utf-8"))["state"]
    assert state in {"failed", "uncertain"}


def test_timeout_is_uncertain_and_not_retried(settings, pdf_data):
    def handle(request):
        raise httpx.ReadTimeout("private text", request=request)

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    with pytest.raises(RequestTimeoutError, match="billing outcome"):
        parser.parse(pdf_data, "application/pdf")
    with pytest.raises(DocumentProcessingStateError):
        parser.parse(pdf_data, "application/pdf")
    assert parser.requests_sent == 1


def test_missing_key_does_not_send(settings, pdf_data):
    from pydantic import SecretStr

    settings = settings.model_copy(update={"upstage_api_key": SecretStr("")})
    with patch.object(UpstageDocumentParser, "_request") as request:
        with pytest.raises(ConfigurationError):
            UpstageDocumentParser(settings).parse(pdf_data, "application/pdf")
    request.assert_not_called()


def test_structure_order_math_html_and_coordinates(parse_payload):
    payload = parse_payload(1, text="| 값 |\n|---|\n| $x^2$ |", category="table")
    pages, model = normalize_response(payload, 1, 100)
    assert pages[0].page_number == 101 and pages[0].elements[0].page_number == 101
    assert "$x^2$" in pages[0].markdown
    assert "<table>" in pages[0].elements[0].html
    assert pages[0].elements[0].coordinates[0].x == 0.1
    assert model == "document-parse-260930"


def test_reject_bad_page_mapping_and_wrong_mode(parse_payload):
    payload = parse_payload()
    payload["elements"][0]["page"] = 0
    with pytest.raises(InvalidResponseError):
        normalize_response(payload, 1)
    payload = parse_payload()
    payload["usage"]["enhanced"] = []
    with pytest.raises(InvalidResponseError):
        normalize_response(payload, 1)


def test_split_101_pages_preserves_mapping_and_resumes_only_failed_part(settings, parse_payload):
    with pymupdf.open() as pdf:
        for _ in range(101):
            pdf.new_page()
        data = pdf.tobytes()
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=parse_payload(100 if count == 1 else 1))

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    with pytest.raises(ProviderServiceUnavailableError):
        parser.parse(data, "application/pdf")
    with pytest.raises(DocumentProcessingStateError):
        parser.parse(data, "application/pdf")
    result = parser.parse(data, "application/pdf", retry_failed=True)
    assert count == 3 and result.requests_sent == 1
    assert [p.number for p in result.document.pages] == list(range(1, 102))


def test_model_version_partitions_cache(settings, pdf_data, parse_payload):
    seen = []

    def handle(request):
        seen.append(1)
        return httpx.Response(200, json=parse_payload(2))

    transport = httpx.MockTransport(handle)
    UpstageDocumentParser(settings, transport=transport).parse(pdf_data, "application/pdf")
    pinned = settings.model_copy(update={"upstage_model": "document-parse-260930"})
    UpstageDocumentParser(pinned, transport=transport).parse(pdf_data, "application/pdf")
    assert len(seen) == 2


def test_partial_result_is_reused_as_partial(settings, pdf_data, parse_payload):
    payload = parse_payload(2)
    payload["elements"] = payload["elements"][:1]
    seen = []

    def handle(request):
        seen.append(1)
        return httpx.Response(200, json=payload)

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    first = parser.parse(pdf_data, "application/pdf")
    second = parser.parse(pdf_data, "application/pdf")
    assert first.status == second.status == "partial"
    assert second.cache_hit and len(seen) == 1


def test_pdf_over_200_pages_rejected_before_request(settings):
    with pymupdf.open() as pdf:
        for _ in range(201):
            pdf.new_page()
        data = pdf.tobytes()
    with patch.object(UpstageDocumentParser, "_request") as call:
        with pytest.raises(InputError, match="1-200"):
            UpstageDocumentParser(settings).parse(data, "application/pdf")
    call.assert_not_called()
