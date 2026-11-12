"""Explicit live Upstage probe: 2-page fictional PDF + PNG + JPEG, <=4 paid pages."""

import argparse
import json
from pathlib import Path

import pymupdf

from api.config import Settings
from api.upstage import UpstageDocumentParser
from src.errors import AIError, ConfigurationError, InvalidResponseError
from src.material import extract_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--report", type=Path, default=Path(".cache/diagnostics/upstage-live.json"))
    args = parser.parse_args()
    report = {"success": False, "steps": [], "page_budget": 4, "requests_sent": 0}
    client = None
    try:
        settings = Settings.from_env(Path(".env"))
        report["requested_model"] = settings.upstage_model
        if not settings.upstage_api_key.get_secret_value():
            raise ConfigurationError("UPSTAGE_API_KEY is missing; no request sent.")
        client = UpstageDocumentParser(settings)
        fixture = Path("tests/fixtures/document_parse.pdf")
        pdf_data = fixture.read_bytes()
        with pymupdf.open(stream=pdf_data, filetype="pdf") as pdf:
            pix = pdf[0].get_pixmap(dpi=150, alpha=False)
            png, jpeg = pix.tobytes("png"), pix.tobytes("jpeg")
        for name, data, mime in (
            ("pdf", pdf_data, "application/pdf"),
            ("png", png, "image/png"),
            ("jpeg", jpeg, "image/jpeg"),
        ):
            step = {"format": name, "success": False}
            report["steps"].append(step)
            result = extract_material(data, mime, parser=client, filename="fixture." + name)
            step["output"] = result.model_dump(mode="json")
            first = result.document.pages[0].text
            compact = "".join(first.split()).casefold().replace("\\", "")
            checks = {
                "korean": "평서문" in first and "한국어" in first,
                "english": "nurse" in first,
                "equations": "x+2=5" in compact and "x=3" in compact,
                "table_cells": "|영어|3|" in compact and "|수학|2|" in compact,
                "table_structure": any(
                    e.category == "table" and "<table" in e.html
                    for p in result.structured_pages
                    for e in p.elements
                ),
                "page_mapping": [p.number for p in result.document.pages]
                == ([1, 2] if name == "pdf" else [1]),
            }
            step["checks"] = checks
            if not all(checks.values()):
                raise InvalidResponseError(
                    "Live document comparison failed; inspect saved fixture output."
                )
            if name == "pdf":
                step["visual_elements"] = [
                    e.category
                    for p in result.structured_pages
                    for e in p.elements
                    if e.visual_interpretation
                ]
                step["visual_check"] = (
                    "Chart identification/interpretation still requires source review."
                )
                visual_text = "\n".join(
                    e.markdown
                    for p in result.structured_pages
                    for e in p.elements
                    if e.visual_interpretation
                )
                step["visual_checks"] = {
                    "chart_detected": bool(step["visual_elements"]),
                    "no_unsupported_sunday_claim": "일요일" not in visual_text,
                    "excluded_from_source_chunks": all(
                        "Chart Type:" not in c.text for c in result.chunk_drafts
                    ),
                }
            step["success"] = True
            print(f"OK: {name}", flush=True)
        txt = extract_material("한글 TXT 원문\nEnglish x + 2 = 5\n".encode(), "text/plain")
        report["txt_direct_success"] = (
            txt.document.pages[0].text == "한글 TXT 원문\nEnglish x + 2 = 5\n"
        )
        report["success"] = report["txt_direct_success"]
        report["api_success"] = all(step["success"] for step in report["steps"])
        report["visual_quality_verified"] = False
        report["quality_status"] = "review_required"
    except (AIError, OSError, ValueError) as exc:
        report["error"] = type(exc).__name__
        report["message"] = str(exc)
        print(f"Stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if client:
            report["requests_sent"] = client.requests_sent
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {args.report}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
