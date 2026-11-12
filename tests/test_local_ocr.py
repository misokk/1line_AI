from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from src.errors import ConfigurationError, InputError
from src.pdf import _choose_text, _tesseract_command, extract_pdf_with_quality


def test_selection_rejects_loss_and_never_concatenates():
    assert _choose_text("한국어 원문 x + 2 = 5", "한국어 원문", 95)[2] is False
    assert _choose_text("x + 2 = 5", "x + 3 = 5", 95)[2] is False
    assert _choose_text("original words retained", "different", 95)[2] is False
    assert _choose_text("원문", "다른 내용", 10)[2] is False
    text, method, ready, _ = _choose_text("She is a nurse.", "평서문 She is a nurse.", 95)
    assert ready and method == "local_tesseract"
    assert text.count("She is a nurse.") == 1


def test_native_text_never_needs_tesseract(pdf_data):
    with patch("src.pdf._ocr_page", side_effect=AssertionError("must not OCR")):
        doc, quality = extract_pdf_with_quality(pdf_data)
    assert len(doc.pages) == 2 and all(q.method == "pdf_text" for q in quality)


def test_unresolved_local_quality_still_fails_strictly(pdf_data):
    with patch("src.pdf._ocr_page", return_value=("broken", 10)):
        with pytest.raises(InputError, match="quality unresolved"):
            extract_pdf_with_quality(pdf_data, ocr_pages=[1])


def test_missing_local_engine_reports_configuration(pdf_data):
    with patch("src.pdf._tesseract_command", side_effect=ConfigurationError("Tesseract missing")):
        with pytest.raises(ConfigurationError, match="Tesseract missing"):
            extract_pdf_with_quality(pdf_data, ocr_pages=[1])


def test_real_local_diagnostic_remains_available():
    try:
        _tesseract_command()
    except ConfigurationError:
        pytest.skip("Install Tesseract kor+eng to run optional local diagnostics")
    data = (Path(__file__).parent / "fixtures" / "local_ocr.pdf").read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as source, pymupdf.open() as pdf:
        pdf.insert_pdf(source, from_page=0, to_page=0)
        pdf.insert_pdf(source, from_page=3, to_page=3)
        document, quality = extract_pdf_with_quality(pdf.tobytes())
        assert document.pages[0].text == source[0].get_text(sort=True)
    assert [p.number for p in document.pages] == [1, 2]
    assert quality[1].method == "local_tesseract"
    assert "x+2=5" in document.pages[1].text.replace(" ", "")
