from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from api.config import ModelTarget, Settings
from src.models import (
    ConceptChecklistItem,
    ConceptVersion,
    ConfirmedConcept,
    MultipleChoiceAnswer,
    Option,
    PublicQuestionDraft,
    SourceChunk,
    SourceEvidence,
    StoredQuestion,
    StudyContext,
)
from src.pdf import extract_pdf


def sample_id(number):
    return UUID(f"{number:08x}-1111-4111-8111-111111111111")


@pytest.fixture
def pdf_data():
    return (Path(__file__).parent / "fixtures" / "english_grammar.pdf").read_bytes()


def make_context(pdf_data):
    document = extract_pdf(pdf_data)
    chunks = [
        SourceChunk(
            id=sample_id(i + 1),
            material_id=sample_id(100),
            material_name="english_grammar.pdf",
            page_number=p.number,
            text=p.text,
        )
        for i, p in enumerate(document.pages)
    ]
    evidence = SourceEvidence(
        chunk_id=chunks[0].id,
        material_id=chunks[0].material_id,
        material_name=chunks[0].material_name,
        page_number=1,
        excerpt="The simple present describes habits and general truths.",
    )
    concept = ConfirmedConcept(
        concept_id=sample_id(10),
        unit_id=sample_id(12),
        material_id=sample_id(100),
        version=ConceptVersion(
            id=sample_id(11),
            version_no=1,
            name="Simple Present",
            summary=evidence.excerpt,
            is_confirmed=True,
            is_current=False,
            confirmed_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
            checklist_items=[
                ConceptChecklistItem(
                    id=sample_id(20), ordinal=1, criterion="Explain habitual actions"
                ),
                ConceptChecklistItem(
                    id=sample_id(21), ordinal=2, criterion="Explain third-person -s"
                ),
            ],
            evidence=[evidence],
        ),
    )
    return StudyContext(concepts=[concept], chunks=chunks)


@pytest.fixture
def context(pdf_data):
    return make_context(pdf_data)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        upstage_api_key="fake-upstage-key",
        document_cache_dir=tmp_path / "document-cache",
        model_allowlist={
            "gemini:fixture-model": ModelTarget(
                provider="gemini", model="fixture-model", available=True, supports_images=True
            )
        },
        default_model_id="gemini:fixture-model",
    )


@pytest.fixture
def parse_payload():
    def make(count=1, text="한국어 English x + 2 = 5", category="paragraph"):
        return {
            "model": "document-parse-260930",
            "usage": {"pages": count, "enhanced": list(range(1, count + 1)), "standard": []},
            "elements": [
                {
                    "id": i,
                    "page": i + 1,
                    "category": category,
                    "content": {
                        "markdown": text,
                        "text": text,
                        "html": "<table><tr><td>한국어</td></tr></table>"
                        if category == "table"
                        else "<p>text</p>",
                    },
                    "coordinates": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.2}],
                }
                for i in range(count)
            ],
        }

    return make


@pytest.fixture
def stored(context):
    concept = context.concepts[0]
    return StoredQuestion(
        question=PublicQuestionDraft(
            question_type="multiple_choice",
            concept_ids=[concept.concept_id],
            concept_version_ids=[concept.version.id],
            prompt="What does simple present describe?",
            options=[Option(id="A", text="Habits"), Option(id="B", text="Only future actions")],
            difficulty="easy",
            evidence=concept.version.evidence,
        ),
        correct_answer=MultipleChoiceAnswer(type="multiple_choice", selected_option_ids=["A"]),
        explanation="Simple present describes habits.",
        accepted_answers=[],
    )
