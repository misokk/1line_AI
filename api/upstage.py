"""Upstage recognition only: fixed endpoint, validated pages, reusable paid results."""

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Literal

import httpx
import pymupdf
from pydantic import ConfigDict, Field, ValidationError

from api.config import Settings
from src.errors import (
    AIError,
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
from src.models import Document, Model, Page
from src.pdf import MAX_PDF_BYTES

FORMATS = ("application/pdf", "image/png", "image/jpeg", "text/plain")
PRICE_USD_PER_PAGE = 0.03  # Enhanced list price, checked 2026-11-04; excludes VAT.
PRICE_SOURCE = "https://www.upstage.ai/pricing/api"


class Coordinates(Model):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ParsedElement(Model):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    source_element_id: int = Field(ge=0)
    page_number: int = Field(ge=1, le=200)
    order: int = Field(ge=0)
    category: str
    sub_category: str | None = None
    markdown: str = Field(max_length=100000)
    html: str = Field(default="", max_length=200000)
    text: str = Field(default="", max_length=100000)
    coordinates: list[Coordinates] = Field(default_factory=list, max_length=20)
    visual_interpretation: bool = False


class ParsedPage(Model):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    page_number: int = Field(ge=1, le=200)
    markdown: str = Field(max_length=100000)
    elements: list[ParsedElement]
    status: Literal["recognized", "review_required", "missing"]
    warnings: list[str] = Field(default_factory=list)


class ParseMetadata(Model):
    source_sha256: str
    requested_model: str
    actual_models: list[str]
    mode: Literal["enhanced"] = "enhanced"
    ocr: Literal["force"] = "force"
    status: Literal["recognized", "review_required", "partial"]
    estimated_cost_usd: float
    price_source: str = PRICE_SOURCE
    price_checked_on: str = "2026-11-04"
    cost_note: str = "Initial public-list-price estimate, not actual billing; excludes VAT/credits."
    cache_hit: bool = False
    requests_sent: int = 0
    warnings: list[str] = Field(default_factory=list)


class ParseResult(ParseMetadata):
    document: Document
    pages: list[ParsedPage]


def inspect_input(data: bytes, mime: str) -> int:
    if mime not in FORMATS:
        raise InputError("Supported types: PDF, TXT, PNG, JPEG only.")
    if not data or len(data) > MAX_PDF_BYTES:
        raise InputError("Material must be nonempty and at most 20 MiB.")
    if mime == "text/plain":
        if data.startswith((b"%PDF-", b"\x89PNG", b"\xff\xd8")) or b"\x00" in data[:100]:
            if not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                raise InputError("TXT MIME does not match its content.")
        return 1
    try:
        if mime == "application/pdf":
            if b"%PDF-" not in data[:1024]:
                raise InputError("PDF MIME does not match its content.")
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass or not 1 <= len(pdf) <= 200:
                    raise InputError("PDF must be unencrypted and contain 1-200 pages.")
                return len(pdf)
        if not (
            mime == "image/png"
            and data.startswith(b"\x89PNG\r\n\x1a\n")
            or mime == "image/jpeg"
            and data.startswith(b"\xff\xd8\xff")
        ):
            raise InputError("Image MIME does not match its content.")
        pix = pymupdf.Pixmap(data)
        if pix.width * pix.height > 200000000:
            raise InputError("Image exceeds 200 million pixels.")
        return 1
    except (RuntimeError, ValueError):
        raise InputError("Damaged or unreadable PDF/image.") from None


def read_txt(data: bytes) -> str:
    # BOM is authoritative. UTF-8 first, CP949 covers common Korean text files.
    encodings = (
        ["utf-16"] if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ["utf-8-sig", "cp949"]
    )
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text and text.strip():
            if len(text) > 100000:
                raise InputError("TXT exceeds the page text limit.")
            return text
    raise InputError("TXT must contain readable UTF-8, BOM UTF-16 or CP949 text.")


def normalize_response(payload: dict, count: int, offset: int = 0) -> tuple[list[ParsedPage], str]:
    """Element page/id, rather than document-level Markdown, define provenance."""
    try:
        actual = payload["model"]
        rows = payload["elements"]
        usage = payload["usage"]
        if not isinstance(actual, str) or not actual.startswith("document-parse"):
            raise ValueError("model")
        if not isinstance(rows, list) or not isinstance(usage, dict) or usage.get("pages") != count:
            raise ValueError("usage/pages")
        enhanced = usage.get("enhanced")
        if enhanced is not None and sorted(enhanced) != list(range(1, count + 1)):
            raise ValueError("not enhanced")
        by_page = {n: [] for n in range(1, count + 1)}
        ids = set()
        for row in rows:
            page, element_id = row["page"], row["id"]
            if type(page) is not int or page not in by_page or type(element_id) is not int:
                raise ValueError("element page/id")
            if element_id in ids:
                raise ValueError("duplicate element id")
            ids.add(element_id)
            content = row["content"]
            category = row["category"]
            if not isinstance(category, str) or not isinstance(content, dict):
                raise ValueError("content")
            by_page[page].append(
                ParsedElement(
                    source_element_id=element_id,
                    page_number=page + offset,
                    order=len(by_page[page]),
                    category=category,
                    sub_category=str(row["sub_category"])
                    if row.get("sub_category") is not None
                    else None,
                    markdown=content.get("markdown") or content.get("text") or "",
                    html=content.get("html") or "",
                    text=content.get("text") or "",
                    coordinates=row.get("coordinates") or [],
                    visual_interpretation=category in {"figure", "chart"},
                )
            )
        pages = []
        for n, elements in by_page.items():
            warnings = []
            status = "recognized"
            if not elements or not any(e.markdown.strip() for e in elements):
                status = "missing"
                warnings.append("No page content returned; blankness is not inferred.")
            for element in elements:
                if element.visual_interpretation:
                    status = "review_required" if status != "missing" else status
                    warnings.append(
                        "Figure/chart description is model interpretation, not a verified quote."
                    )
                if not element.markdown.strip():
                    status = "review_required" if status != "missing" else status
                    warnings.append("Element has no Markdown/text; inspect original visual/HTML.")
                if "\ufffd" in element.markdown:
                    status = "review_required"
                    warnings.append("Replacement characters require source review.")
                if len(element.markdown) > 12000:
                    status = "review_required"
                    warnings.append(
                        "Oversized element requires explicit chunk policy; not split automatically."
                    )
            markdown = "\n\n".join(e.markdown for e in elements if e.markdown)
            pages.append(
                ParsedPage(
                    page_number=n + offset,
                    markdown=markdown,
                    elements=elements,
                    status=status,
                    warnings=list(dict.fromkeys(warnings)),
                )
            )
        return pages, actual
    except (KeyError, TypeError, ValueError, ValidationError):
        raise InvalidResponseError(
            "Upstage response has invalid model, page coverage or element structure."
        ) from None


class UpstageDocumentParser:
    """A small request cache/duplicate guard, not a backend job queue or retry engine."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self._lock = RLock()
        self.requests_sent = 0

    def _request(self, data: bytes, mime: str) -> dict:
        key = self.settings.upstage_api_key.get_secret_value()
        if not key or not key.isascii() or any(c.isspace() for c in key):
            raise ConfigurationError("Set a valid UPSTAGE_API_KEY; no request sent.")
        ext = "pdf" if mime == "application/pdf" else "png" if mime == "image/png" else "jpg"
        try:
            with httpx.Client(
                timeout=self.settings.upstage_timeout_seconds,
                follow_redirects=False,
                trust_env=False,
                transport=self.transport or httpx.HTTPTransport(retries=0),
            ) as client:
                self.requests_sent += 1
                response = client.post(
                    "https://api.upstage.ai/v1/document-digitization",
                    headers={"Authorization": "Bearer " + key},
                    files={"document": ("material." + ext, data, mime)},
                    data={
                        "model": self.settings.upstage_model,
                        "mode": "enhanced",
                        "ocr": "force",
                        "output_formats": json.dumps(["markdown", "html", "text"]),
                        "coordinates": "true",
                        "merge_multipage_tables": "false",
                    },
                )
        except httpx.TimeoutException:
            raise RequestTimeoutError(
                "Upstage timed out; billing outcome may be unknown. No automatic retry."
            ) from None
        except httpx.HTTPError:
            raise ProviderError("Upstage network request failed; no automatic retry.") from None
        code = response.status_code
        if code in (401, 403):
            raise ModelAccessError("Upstage authentication/access failed.")
        if code == 429:
            raise RateLimitError("Upstage rate limit/quota exceeded.")
        if code in (408, 504):
            raise RequestTimeoutError("Upstage request timeout; no automatic retry.")
        if code == 502:
            raise ProviderBadGatewayError("Upstage returned HTTP 502.")
        if code == 503:
            raise ProviderServiceUnavailableError("Upstage returned HTTP 503.")
        if code in (400, 404, 413, 415, 422):
            raise ProviderRequestError(f"Upstage rejected the document/request (HTTP {code}).")
        if code != 200:
            raise ProviderError(f"Upstage unavailable (HTTP {code}).")
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            return payload
        except ValueError:
            raise InvalidResponseError("Upstage returned a non-JSON response.") from None

    @staticmethod
    def _save(path: Path, value: dict):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def parse(self, data: bytes, mime: str, *, retry_failed: bool = False) -> ParseResult:
        count = inspect_input(data, mime)
        if mime == "text/plain":
            raise InputError("TXT must be decoded directly without Upstage.")
        key = self.settings.upstage_api_key.get_secret_value()
        if not key:
            raise ConfigurationError("UPSTAGE_API_KEY is missing; no request sent.")
        digest = hashlib.sha256(data).hexdigest()
        identity = [
            digest,
            mime,
            self.settings.upstage_model,
            "enhanced",
            "force",
            "markdown/html/text",
            "normalizer-v1",
            hashlib.sha256(key.encode()).hexdigest(),
        ]
        cache_key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        folder = self.settings.document_cache_dir / cache_key
        with self._lock:
            result_path = folder / "result.json"
            try:
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                has_result = result_path.is_file()
            except OSError:
                raise ConfigurationError(
                    "Cannot access DOCUMENT_CACHE_DIR; check permissions."
                ) from None
            if has_result and not retry_failed:
                try:
                    cached = ParseResult.model_validate_json(
                        result_path.read_text(encoding="utf-8")
                    )
                    return cached.model_copy(
                        update={"cache_hit": True, "requests_sent": 0}, deep=True
                    )
                except (ValueError, OSError):
                    raise DocumentProcessingStateError(
                        "Cached parse is unreadable; backend must review before retry."
                    ) from None
            lock_path = folder / "request.lock"
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError:
                raise DocumentProcessingStateError(
                    "Parse is in progress or interrupted. Confirm prior request state before retry."
                ) from None
            except OSError:
                raise ConfigurationError(
                    "Document cache is not writable; check DOCUMENT_CACHE_DIR."
                ) from None
            sent_before = self.requests_sent
            active_attempt = None
            try:
                all_pages, models = [], []
                for start in range(0, count, 100):
                    end = min(count, start + 100)
                    part_path, attempt = (
                        folder / f"part-{start}.json",
                        folder / f"attempt-{start}.json",
                    )
                    if part_path.is_file():
                        stored = json.loads(part_path.read_text(encoding="utf-8"))
                        pages = [ParsedPage.model_validate(p) for p in stored["pages"]]
                        actual = stored["model"]
                        if not retry_failed or all(p.status == "recognized" for p in pages):
                            all_pages.extend(pages)
                            models.append(actual)
                            continue
                    if attempt.exists() and not retry_failed:
                        raise DocumentProcessingStateError(
                            "Prior billing is failed/unknown; backend must authorize retry_failed."
                        )
                    body = data
                    if count > 100:
                        with (
                            pymupdf.open(stream=data, filetype="pdf") as pdf,
                            pymupdf.open() as part,
                        ):
                            part.insert_pdf(pdf, from_page=start, to_page=end - 1)
                            body = part.tobytes()
                    self._save(
                        attempt,
                        {
                            "state": "submitted",
                            "start_page": start + 1,
                            "end_page": end,
                            "model": self.settings.upstage_model,
                        },
                    )
                    active_attempt = attempt
                    payload = self._request(body, mime)
                    pages, actual = normalize_response(payload, end - start, start)
                    self._save(
                        part_path, {"pages": [p.model_dump() for p in pages], "model": actual}
                    )
                    self._save(
                        attempt,
                        {
                            "state": "recognized"
                            if all(p.status == "recognized" for p in pages)
                            else "review_required"
                        },
                    )
                    all_pages.extend(pages)
                    models.append(actual)
                    active_attempt = None
                # Never merge tables across pages speculatively. Mark adjacent candidates only.
                for previous, current in zip(all_pages, all_pages[1:], strict=False):
                    if (
                        previous.elements
                        and current.elements
                        and previous.elements[-1].category
                        == current.elements[0].category
                        == "table"
                    ):
                        current.warnings.append(
                            "Tables may continue across pages; relationship unverified."
                        )
                document = Document(
                    pages=[Page(number=p.page_number, text=p.markdown) for p in all_pages]
                )
                status = (
                    "partial"
                    if any(p.status == "missing" for p in all_pages)
                    else (
                        "review_required"
                        if any(p.status != "recognized" for p in all_pages)
                        else "recognized"
                    )
                )
                result = ParseResult(
                    document=document,
                    pages=all_pages,
                    source_sha256=digest,
                    requested_model=self.settings.upstage_model,
                    actual_models=list(dict.fromkeys(models)),
                    status=status,
                    estimated_cost_usd=round(count * PRICE_USD_PER_PAGE, 6),
                    requests_sent=self.requests_sent - sent_before,
                    warnings=[
                        "Verify model-recognized quotes and visual interpretations against source."
                    ],
                )
                self._save(result_path, result.model_dump(mode="json"))
                return result
            except AIError as exc:
                if active_attempt is not None:
                    state = (
                        "failed"
                        if isinstance(
                            exc,
                            (
                                ModelAccessError,
                                RateLimitError,
                                ProviderRequestError,
                                InvalidResponseError,
                            ),
                        )
                        else "uncertain"
                    )
                    self._save(active_attempt, {"state": state, "error": type(exc).__name__})
                raise
            except (ValueError, OSError, ValidationError):
                raise InvalidResponseError(
                    "Parse cache or normalized document is incomplete; no automatic retry."
                ) from None
            finally:
                lock_path.unlink(missing_ok=True)
