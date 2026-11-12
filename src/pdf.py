import csv
import io
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pymupdf
from pydantic import Field

from src.errors import ConfigurationError, InputError
from src.models import Document, Evidence, Model, Page

MAX_PDF_BYTES = 20 * 1024 * 1024


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class PDFPageQuality(Model):
    page_number: int
    method: str
    reasons: list[str]
    raw_chars: int
    ocr_chars: int = 0
    ocr_confidence: float | None = None
    ready: bool = True
    warnings: list[str] = Field(default_factory=list)
    ocr_text: str | None = Field(default=None, max_length=100000)


def _broken(text: str) -> int:
    return sum(
        c == "\ufffd"
        or unicodedata.category(c) in {"Co", "Cs"}
        or (unicodedata.category(c) == "Cc" and c not in "\n\r\t")
        for c in text
    )


def _hangul(text: str) -> int:
    return len(re.findall(r"[가-힣]", text))


def _quality_reasons(page, text: str) -> list[str]:
    reasons = []
    if not text.strip():
        reasons.append("empty_text")
    if _broken(text):
        reasons.append("broken_characters")
    if re.search(r" {12,}", text):
        reasons.append("suspicious_spacing")
    area = page.rect.width * page.rect.height
    # Images may contain Korean absent from the text layer, even with English present.
    if any(pymupdf.Rect(i["bbox"]).get_area() / area > 0.02 for i in page.get_image_info()):
        reasons.append("image_may_contain_missing_text")
    if not _hangul(text) and len(page.get_drawings()) > 100:
        reasons.append("possible_outlined_text")
    return reasons


def _tesseract_command() -> str:
    command = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not command and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "")) / "Tesseract-OCR" / "tesseract.exe"
        if candidate.is_file():
            command = str(candidate)
    if not command:
        raise ConfigurationError(
            "Local OCR requires Tesseract with kor and eng language data. "
            "Install it or set TESSERACT_CMD."
        )
    return command


def _ocr_page(page, *, psm: int = 3, dpi: int = 300) -> tuple[str, float]:
    command = [
        _tesseract_command(),
        "stdin",
        "stdout",
        "-l",
        "kor+eng",
        "--oem",
        "1",
        "--psm",
        str(psm),
    ]
    data_dir = os.getenv("TESSDATA_PREFIX")
    local_data = Path(__file__).resolve().parents[1] / ".tessdata"
    if not data_dir and all(
        (local_data / f"{lang}.traineddata").is_file() for lang in ("kor", "eng")
    ):
        data_dir = str(local_data)
    if data_dir:
        command += ["--tessdata-dir", data_dir]
    command += ["-c", "tessedit_create_tsv=1"]
    if page.rect.get_area() * (dpi / 72) ** 2 > 30000000:
        raise InputError("Local OCR rendered page exceeds 30 million pixels.")
    image = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False).tobytes("png")
    try:
        # Tesseract's internal thread fan-out can slow small pages considerably.
        result = subprocess.run(
            command,
            input=image,
            capture_output=True,
            timeout=60,
            check=False,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
    except subprocess.TimeoutExpired:
        raise InputError("Local Tesseract OCR timed out after 60 seconds; no retry.") from None
    except OSError:
        raise ConfigurationError("Cannot run Tesseract; check TESSERACT_CMD.") from None
    if result.returncode:
        raise ConfigurationError("Tesseract OCR failed; check kor/eng data and TESSDATA_PREFIX.")
    lines, confidences = {}, []
    for row in csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE
    ):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append(row)
        confidences.extend([max(0, float(row["conf"]))] * len(text))
    output, line_confidences = [], []
    for words in lines.values():
        count = sum(len(w["text"]) for w in words)
        if count >= 10:
            line_confidences.append(sum(float(w["conf"]) * len(w["text"]) for w in words) / count)
        line, previous = "", None
        for word in words:
            text = word["text"].strip()
            gap = (
                int(word["left"]) - (int(previous["left"]) + int(previous["width"]))
                if previous
                else 0
            )
            adjacent_korean = previous and _hangul(line[-1:]) and _hangul(text[:1])
            separator = "" if adjacent_korean and gap < int(word["height"]) * 0.25 else " "
            line += (separator if line else "") + text
            previous = word
        output.append(line)
    # A high page average must not hide a badly recognized Korean line.
    mean = sum(confidences) / len(confidences) if confidences else 0
    confidence = min(mean, min(line_confidences, default=mean))
    return "\n".join(output), round(confidence, 2)


def _choose_text(raw: str, ocr: str, confidence: float) -> tuple[str, str, bool, list[str]]:
    # Never concatenate entire outputs: it duplicates headings and examples.
    raw_words = set(re.findall(r"[A-Za-z]{2,}|[가-힣]+|[0-9]+", raw.casefold()))
    ocr_words = set(re.findall(r"[A-Za-z]{2,}|[가-힣]+|[0-9]+", ocr.casefold()))
    coverage = len(raw_words & ocr_words) / max(1, len(raw_words))
    math = set(re.findall(r"[=+×÷−√∑∫≤≥²³]", raw))
    lost_math = bool(math - set(ocr))
    expressions = re.findall(r"[A-Za-z0-9²³.]+(?:\s*[=+×÷−≤≥]\s*[A-Za-z0-9²³.]+)+", raw)
    compact_ocr = re.sub(r"\s", "", ocr).casefold()
    lost_math |= any(re.sub(r"\s", "", e).casefold() not in compact_ocr for e in expressions)
    lines = [normalized(line) for line in ocr.splitlines() if line.strip()]
    raw_lines = [normalized(line) for line in raw.splitlines() if line.strip()]
    duplicated = len(lines) > 3 and len(set(lines)) / len(lines) < 0.7
    fragmented = len(lines) > 5 and sum(len(line) < 3 for line in lines) / len(lines) > 0.3
    if fragmented or (duplicated and len(set(raw_lines)) == len(raw_lines)):
        return raw, "pdf_text", False, ["OCR has fragmented or duplicate lines; review source."]
    if not ocr.strip() or confidence < 70 or _broken(ocr):
        return (
            raw,
            "pdf_text",
            False,
            ["Local OCR is empty, broken or low-confidence; review source."],
        )
    gained_korean = _hangul(ocr) >= _hangul(raw) + 3
    if lost_math:
        return (
            raw,
            "pdf_text",
            False,
            ["OCR lost mathematical symbols; original retained for review."],
        )
    if raw_words and coverage < 0.65 and not _broken(raw):
        return raw, "pdf_text", False, ["OCR lost source words; original retained for review."]
    if not raw.strip() or _broken(raw) or gained_korean or len(ocr_words - raw_words) >= 2:
        return ocr, "local_tesseract", True, ["OCR math/notation must be checked against the page."]
    # No material improvement: preserve original spelling, equations and layout text.
    return raw, "pdf_text", True, []


def extract_pdf_with_quality(
    data: bytes,
    *,
    max_ocr_pages: int = 5,
    ocr_pages: list[int] | None = None,
    strict: bool = True,
) -> tuple[Document, list[PDFPageQuality]]:
    if not data or len(data) > MAX_PDF_BYTES:
        raise InputError("PDF must be nonempty and at most 10 MiB.")
    if not 1 <= max_ocr_pages <= 10:
        raise InputError("Local OCR requires a 1-10 page budget.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass or not 1 <= pdf.page_count <= 200:
                raise InputError("Encrypted, empty or oversized PDF.")
            if ocr_pages is not None and (
                not ocr_pages
                or len(set(ocr_pages)) != len(ocr_pages)
                or any(n < 1 or n > pdf.page_count for n in ocr_pages)
            ):
                raise InputError("ocr_pages must contain distinct existing 1-based page numbers.")
            raw = [p.get_text(sort=True) for p in pdf]
            if sum(map(len, raw)) > 400000 or any(len(t) > 100000 for t in raw):
                raise InputError("Extracted PDF exceeds text limits.")
            reasons = [_quality_reasons(p, t) for p, t in zip(pdf, raw, strict=True)]
            for i, page in enumerate(pdf):
                if not raw[i].strip():
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(0.25, 0.25), colorspace=pymupdf.csGRAY, alpha=False
                    )
                    if all(v > 245 for v in pix.samples):
                        reasons[i] = []  # A truly blank page needs no OCR.
            targets = [i for i, r in enumerate(reasons) if r or i + 1 in (ocr_pages or [])]
            if len(targets) > max_ocr_pages:
                raise InputError(
                    "Local OCR exceeds page budget; split PDF or increase max_ocr_pages "
                    "(up to 10). No OCR started."
                )
            pages, reports = [], []
            for i, page in enumerate(pdf):
                text = raw[i]
                report = PDFPageQuality(
                    page_number=i + 1, method="pdf_text", reasons=reasons[i], raw_chars=len(text)
                )
                if i in targets:
                    # Sparse native text usually means a single scanned text block.
                    psm = 6 if len(text.splitlines()) < 8 else 3
                    ocr, confidence = _ocr_page(page, psm=psm)
                    text, report.method, report.ready, report.warnings = _choose_text(
                        text, ocr, confidence
                    )
                    report.ocr_chars, report.ocr_confidence = len(ocr), confidence
                    report.ocr_text = ocr
                    if not strict and not report.ready and not text.strip() and ocr.strip():
                        text = ocr
                        report.method = "local_tesseract_review"
                pages.append(Page(number=i + 1, text=text))
                reports.append(report)
            unresolved = [r.page_number for r in reports if not r.ready]
            if unresolved and strict:
                raise InputError(
                    f"PDF extraction quality unresolved on pages {unresolved}; "
                    "review original/local OCR before LLM analysis."
                )
            return Document(pages=pages), reports
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError):
        raise InputError(
            "Invalid PDF or no extractable text; local OCR could not recover it."
        ) from None


def extract_pdf(data: bytes) -> Document:
    return extract_pdf_with_quality(data)[0]


def chunks(document: Document, max_chars: int = 10000) -> list[list[Page]]:
    if max_chars < 500:
        raise InputError("Chunk size must be at least 500 characters.")
    result: list[list[Page]] = []
    current: list[Page] = []
    size = 0
    for page in document.pages:
        for start in range(0, len(page.text), max_chars):
            part = page.text[start : start + max_chars]
            if current and size + len(part) > max_chars:
                result.append(current)
                current, size = [], 0
            current.append(Page(number=page.number, text=part))
            size += len(part)
    if current:
        result.append(current)
    if len(result) > 40:
        raise InputError("Document requires more than 40 analysis chunks; split the PDF.")
    return result


def validate_evidence(evidence: list[Evidence], document: Document) -> None:
    pages = {p.number: normalized(p.text) for p in document.pages}
    for item in evidence:
        if item.page not in pages or normalized(item.quote) not in pages[item.page]:
            raise InputError("Evidence quote does not match the cited source page.")
