from pathlib import Path
from uuid import UUID

from pydantic import Field

from api.config import Settings
from api.router import ModelRouter
from api.upstage import (
    ParsedElement,
    ParsedPage,
    ParseMetadata,
    UpstageDocumentParser,
    inspect_input,
    read_txt,
)
from src.errors import InputError
from src.models import (
    ChunkDraft,
    Document,
    Model,
    Page,
    SourceChunk,
)


class OCRText(Model):
    # Retained for explicit legacy vision-client probes, never the recognition default.
    text: str
    uncertainty_notes: list[str]


class Extraction(Model):
    document: Document
    chunk_drafts: list[ChunkDraft]
    method: str
    warnings: list[str]
    ocr_processed_pages: list[int] = Field(default_factory=list)
    ocr_model_calls: int = 0
    ready_for_analysis: bool = True
    review_required_pages: list[int] = Field(default_factory=list)
    structured_pages: list[ParsedPage] = Field(default_factory=list)
    recognition: ParseMetadata | None = None
    material_id: UUID | None = None
    filename: str = Field(default="material", min_length=1, max_length=255)
    mime_type: str
    analysis_scope: str = "source_text_only; model visual interpretations excluded"


def _element_chunks(pages: list[ParsedPage]) -> list[ChunkDraft]:
    drafts = []
    for page in pages:
        current = []
        size = 0
        for element in page.elements:
            if element.visual_interpretation or not element.markdown.strip():
                if current:
                    drafts.append(
                        ChunkDraft(
                            ordinal=len(drafts) + 1,
                            page_number=page.page_number,
                            text="\n\n".join(current),
                        )
                    )
                    current, size = [], 0
                continue
            text = element.markdown
            if len(text) > 12000:
                raise InputError("Oversized document element requires source-aware chunking.")
            if current and size + len(text) + 2 > 6000:
                drafts.append(
                    ChunkDraft(
                        ordinal=len(drafts) + 1,
                        page_number=page.page_number,
                        text="\n\n".join(current),
                    )
                )
                current, size = [], 0
            current.append(text)
            size += len(text) + 2
        if current:
            drafts.append(
                ChunkDraft(
                    ordinal=len(drafts) + 1, page_number=page.page_number, text="\n\n".join(current)
                )
            )
    if len(drafts) > 200:
        raise InputError("Extracted material exceeds 200 source chunks; split material.")
    return drafts


def _txt_chunks(text: str) -> list[ChunkDraft]:
    drafts = []
    while text:
        end = min(6000, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", 0, end), text.rfind(" ", 0, end))
            if boundary > 0:
                end = boundary + 1
        piece, text = text[:end], text[end:]
        if piece.strip():
            drafts.append(ChunkDraft(ordinal=len(drafts) + 1, page_number=1, text=piece))
    return drafts


def extract_material(
    data: bytes,
    mime_type: str,
    *,
    router: ModelRouter | None = None,
    model_id: str | None = None,
    ocr_pdf: bool = False,
    max_ocr_pages: int = 5,
    ocr_pages: list[int] | None = None,
    parser: UpstageDocumentParser | None = None,
    settings: Settings | None = None,
    filename: str = "material",
    material_id: UUID | None = None,
    retry_failed: bool = False,
) -> Extraction:
    inspect_input(data, mime_type)
    filename = filename.replace("\\", "/").split("/")[-1]
    if not filename or len(filename) > 255 or material_id is not None and material_id.version != 4:
        raise InputError(
            "Filename must be 1-255 characters and material_id must be backend UUID v4."
        )
    if ocr_pages is not None:
        raise InputError(
            "Upstage parses the supplied complete file; submit a PDF subset to select pages."
        )
    if mime_type == "text/plain":
        text = read_txt(data)
        document = Document(pages=[Page(number=1, text=text)])
        elements = []
        # Blank-line boundaries retain original text. Never infer Markdown from TXT.
        import re

        blocks = re.split(r"(?<=\n)\n", text)
        for block in blocks:
            while len(block) > 12000:
                end = max(block.rfind("\n", 0, 6000), block.rfind(" ", 0, 6000))
                end = end + 1 if end > 0 else 6000
                elements.append(
                    ParsedElement(
                        source_element_id=len(elements),
                        page_number=1,
                        order=len(elements),
                        category="paragraph",
                        markdown=block[:end],
                        text=block[:end],
                    )
                )
                block = block[end:]
            if block.strip():
                elements.append(
                    ParsedElement(
                        source_element_id=len(elements),
                        page_number=1,
                        order=len(elements),
                        category="paragraph",
                        markdown=block,
                        text=block,
                    )
                )
        pages = [ParsedPage(page_number=1, markdown=text, elements=elements, status="recognized")]
        warnings, recognition, method = [], None, "direct_text"
    else:
        parser = parser or UpstageDocumentParser(settings or Settings.from_env(Path(".env")))
        recognition = parser.parse(data, mime_type, retry_failed=retry_failed)
        document, pages = recognition.document, recognition.pages
        method = "upstage_document_parse"
        warnings = recognition.warnings + [
            f"Page {p.page_number}: {w}" for p in pages for w in p.warnings
        ]
    review = [p.page_number for p in pages if p.status != "recognized"]
    critical = any(
        p.status == "missing"
        or any(
            len(e.markdown) > 12000
            or "\ufffd" in e.markdown
            or (not e.markdown.strip() and not e.visual_interpretation)
            for e in p.elements
        )
        for p in pages
    )
    drafts = (
        []
        if critical
        else (_txt_chunks(text) if mime_type == "text/plain" else _element_chunks(pages))
    )
    ready = bool(drafts) and not critical
    return Extraction(
        document=document,
        chunk_drafts=drafts,
        structured_pages=pages,
        method=method,
        warnings=warnings,
        ready_for_analysis=ready,
        review_required_pages=review,
        recognition=ParseMetadata.model_validate(
            recognition.model_dump(exclude={"document", "pages"})
        )
        if recognition is not None
        else None,
        material_id=material_id,
        filename=filename.replace("\\", "/").split("/")[-1],
        mime_type=mime_type,
        ocr_processed_pages=[] if mime_type == "text/plain" else [p.number for p in document.pages],
    )


def bind_source_chunks(
    extraction: Extraction, *, material_id: UUID, material_name: str, chunk_ids: list[UUID]
) -> list[SourceChunk]:
    """Project exact drafts onto IDs actually allocated/stored by the main backend."""
    if not extraction.ready_for_analysis or not extraction.chunk_drafts:
        raise InputError("Recognition is not ready for concept analysis.")
    if extraction.material_id is not None and extraction.material_id != material_id:
        raise InputError("Material ID differs from the recognized source.")
    if len(chunk_ids) != len(extraction.chunk_drafts) or len(set(chunk_ids)) != len(chunk_ids):
        raise InputError("Provide one distinct persisted chunk ID per draft, in ordinal order.")
    return [
        SourceChunk(
            id=identifier,
            material_id=material_id,
            material_name=material_name,
            page_number=draft.page_number,
            text=draft.text,
        )
        for identifier, draft in zip(chunk_ids, extraction.chunk_drafts, strict=True)
    ]
