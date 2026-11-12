import pymupdf
import pytest
from pydantic import ValidationError

from src.errors import InputError
from src.models import Document, Evidence, Page
from src.pdf import chunks, extract_pdf, validate_evidence


def test_extract_pages_and_quotes(pdf_data):
    doc = extract_pdf(pdf_data)
    assert [p.number for p in doc.pages] == [1, 2]
    assert "Simple Present" in doc.pages[0].text
    assert "Present Continuous" in doc.pages[1].text
    validate_evidence([Evidence(page=2, quote="actions happening now")], doc)
    with pytest.raises(InputError):
        validate_evidence([Evidence(page=1, quote="actions happening now")], doc)


def test_long_page_split_preserves_text_and_page_numbers():
    doc = Document(pages=[Page(number=1, text="abc " * 1000), Page(number=2, text="def " * 400)])
    parts = chunks(doc, 700)
    for number in (1, 2):
        assert "".join(p.text for part in parts for p in part if p.number == number) == (
            doc.pages[number - 1].text
        )
    assert all(sum(len(p.text) for p in part) <= 700 for part in parts)


@pytest.mark.parametrize("data", [b"", b"bad pdf", b"x" * (10 * 1024 * 1024 + 1)])
def test_reject_invalid_and_oversized_pdf(data):
    with pytest.raises(InputError):
        extract_pdf(data)


def test_scanned_blank_pdf_rejected():
    with pymupdf.open() as doc:
        doc.new_page()
        data = doc.tobytes()
    with pytest.raises(InputError, match="OCR"):
        extract_pdf(data)


def test_context_requires_confirmation(context):
    with pytest.raises(ValidationError):
        type(context).model_validate({**context.model_dump(), "confirmed": False})
