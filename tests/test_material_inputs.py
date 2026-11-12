from unittest.mock import MagicMock
from uuid import UUID

import httpx
import pymupdf
import pytest

from api.upstage import UpstageDocumentParser
from src.errors import InputError
from src.material import bind_source_chunks, extract_material


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp949"])
def test_txt_preserved_without_any_provider(encoding):
    text = "\ud55c\uae00 TXT  English\n\n  x + 2 = 5\n"
    unused = MagicMock()
    result = extract_material(text.encode(encoding), "text/plain", parser=unused)
    assert result.document.pages[0].text == text
    assert result.document.pages[0].number == 1 and result.recognition is None
    unused.parse.assert_not_called()
    assert all(c.text in text for c in result.chunk_drafts)


def test_long_txt_keeps_source_substrings_and_page_one():
    text = "한글 English sentence.\n" * 2000
    result = extract_material(text.encode(), "text/plain")
    assert result.document.pages[0].text == text
    assert "".join(c.text for c in result.chunk_drafts) == text
    assert all(c.page_number == 1 and len(c.text) <= 12000 for c in result.chunk_drafts)


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (b"bad", "image/png"),
        (b"bad", "image/jpeg"),
        (b"bad", "application/pdf"),
        (b"\xff", "text/plain"),
        (b"bad", "image/webp"),
    ],
)
def test_invalid_formats_fail_before_provider(data, mime):
    unused = MagicMock()
    with pytest.raises(InputError):
        extract_material(data, mime, parser=unused)
    unused.parse.assert_not_called()


def test_pdf_and_image_share_structure_and_backend_ids(settings, pdf_data, parse_payload):
    def handle(request):
        return httpx.Response(
            200, json=parse_payload(2 if b"application/pdf" in request.content else 1)
        )

    parser = UpstageDocumentParser(settings, transport=httpx.MockTransport(handle))
    pdf = extract_material(pdf_data, "application/pdf", parser=parser)
    with pymupdf.open(stream=pdf_data, filetype="pdf") as source:
        pix = source[0].get_pixmap()
        for image, mime in [(pix.tobytes("png"), "image/png"), (pix.tobytes("jpeg"), "image/jpeg")]:
            result = extract_material(image, mime, parser=parser)
            assert result.structured_pages[0].page_number == 1
    ids = [
        UUID("20000001-1111-4111-8111-111111111111"),
        UUID("20000002-1111-4111-8111-111111111111"),
    ]
    chunks = bind_source_chunks(
        pdf,
        material_id=UUID("30000001-1111-4111-8111-111111111111"),
        material_name="actual.pdf",
        chunk_ids=ids,
    )
    assert [c.id for c in chunks] == ids
    assert [c.page_number for c in chunks] == [1, 2]
    assert chunks[0].text == pdf.chunk_drafts[0].text
    assert all("id" not in c.model_dump() for c in pdf.chunk_drafts)
    with pytest.raises(InputError):
        bind_source_chunks(
            pdf, material_id=chunks[0].material_id, material_name="actual.pdf", chunk_ids=[]
        )


def test_visual_interpretation_excluded_from_learning_sources(settings, pdf_data, parse_payload):
    payload = parse_payload(2)
    payload["elements"].append(
        {
            "id": 99,
            "page": 1,
            "category": "figure",
            "content": {
                "markdown": "Model-generated visual interpretation",
                "html": "",
                "text": "",
            },
        }
    )
    parser = UpstageDocumentParser(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    result = extract_material(pdf_data, "application/pdf", parser=parser)
    assert result.review_required_pages == [1]
    assert result.ready_for_analysis
    assert "visual interpretation" not in "".join(c.text for c in result.chunk_drafts)
    assert result.structured_pages[0].elements[-1].visual_interpretation


def test_missing_page_blocks_analysis_without_faking_completion(settings, pdf_data, parse_payload):
    payload = parse_payload(2)
    payload["elements"] = payload["elements"][:1]
    parser = UpstageDocumentParser(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    result = extract_material(pdf_data, "application/pdf", parser=parser)
    assert result.recognition.status == "partial"
    assert not result.ready_for_analysis and result.chunk_drafts == []
    assert result.review_required_pages == [2]
