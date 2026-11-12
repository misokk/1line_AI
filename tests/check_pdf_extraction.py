"""Offline comparison. No API key, LLM, document upload or model router."""

import argparse
import json
import re
from pathlib import Path

import pymupdf

from src.errors import AIError
from src.pdf import extract_pdf_with_quality


def character_error_rate(expected: str, actual: str) -> float:
    expected, actual = re.sub(r"\s", "", expected), re.sub(r"\s", "", actual)
    previous = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return round(previous[-1] / max(1, len(expected)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("tests/fixtures/local_ocr.pdf"))
    parser.add_argument("--page", type=int, help="Compare just this 1-based source page")
    parser.add_argument(
        "--report", type=Path, default=Path(".cache/diagnostics/local-ocr-comparison.json")
    )
    args = parser.parse_args()
    with pymupdf.open(args.pdf) as source:
        numbers = [args.page] if args.page else list(range(1, len(source) + 1))
        if any(n < 1 or n > len(source) for n in numbers):
            parser.error("Page outside PDF")
        with pymupdf.open() as pdf:
            for n in numbers:
                pdf.insert_pdf(source, from_page=n - 1, to_page=n - 1)
            raw = [page.get_text(sort=True) for page in pdf]
            data = pdf.tobytes()
    try:
        document, quality = extract_pdf_with_quality(data, max_ocr_pages=10, strict=False)
    except AIError as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "external_api_calls": 0,
                    "ready_for_analysis": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Report: {args.report}; extraction failed: {exc}")
        return
    rows = []
    for original_number, raw_text, page, q in zip(
        numbers, raw, document.pages, quality, strict=True
    ):
        row = {
            **q.model_dump(),
            "page_number": original_number,
            "raw_text": raw_text,
            "ocr_text": q.ocr_text,
            "selected_text": page.text,
            "raw_hangul": len(re.findall(r"[가-힣]", raw_text)),
            "ocr_hangul": len(re.findall(r"[가-힣]", q.ocr_text or "")),
        }
        if args.pdf.name == "local_ocr.pdf" and original_number in (2, 3) and not args.page:
            row["ocr_character_error_rate_no_whitespace"] = character_error_rate(
                raw[0], q.ocr_text or ""
            )
        rows.append(row)
    report = {
        "external_api_calls": 0,
        "ready_for_analysis": all(q.ready for q in quality),
        "pages": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.report}; ready_for_analysis={report['ready_for_analysis']}")


if __name__ == "__main__":
    main()
