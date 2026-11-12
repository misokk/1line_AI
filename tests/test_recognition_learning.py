"""Parsed-source integration through existing functions; LLM responses are explicit fixtures."""

from unittest.mock import MagicMock
from uuid import UUID

import httpx

from api.router import ModelRouter
from api.upstage import UpstageDocumentParser
from src import analysis as concept_analysis
from src import evaluation, material, questions
from src.models import (
    AnalysisPart,
    AnalyzeRequest,
    CriterionJudgment,
    DraftConcept,
    DraftUnit,
    EvaluationJudgments,
    EvaluationRequest,
    Message,
    QuestionBatch,
    QuestionRequest,
    SourceEvidence,
)


def test_recognized_chunks_reused_by_all_learning_functions(
    settings, pdf_data, parse_payload, context, stored
):
    parser = UpstageDocumentParser(
        settings,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=parse_payload(2))),
    )
    extracted = material.extract_material(pdf_data, "application/pdf", parser=parser)
    ids = [
        UUID("20000001-1111-4111-8111-111111111111"),
        UUID("20000002-1111-4111-8111-111111111111"),
    ]
    chunks = material.bind_source_chunks(
        extracted,
        material_id=context.concepts[0].material_id,
        material_name="lesson.pdf",
        chunk_ids=ids,
    )
    evidence = SourceEvidence(
        chunk_id=chunks[0].id,
        material_id=chunks[0].material_id,
        material_name=chunks[0].material_name,
        page_number=1,
        excerpt="한국어 English x + 2 = 5",
    )
    llm = MagicMock()
    router = ModelRouter(settings, {"gemini": llm})
    draft = DraftConcept(
        name="한국어 개념",
        summary=evidence.excerpt,
        checklist_items=["설명"],
        keywords=["개념"],
        evidence=[evidence],
        estimated_difficulty="easy",
        estimated_importance=0.8,
    )
    llm.generate_json.side_effect = [
        AnalysisPart(units=[DraftUnit(title="Unit", concepts=[draft])]),
        AnalysisPart(units=[]),
    ]
    analysis = concept_analysis.analyze(AnalyzeRequest(chunks=chunks), router)
    assert analysis.state == "draft"
    assert analysis.units[0].concepts[0].evidence[0].chunk_id == ids[0]
    # Simulate the existing backend confirmation/versioning contract with persisted fixture IDs.
    context.chunks = chunks
    version = context.concepts[0].version
    version.evidence = [evidence]
    version.summary = evidence.excerpt
    llm.generate_json.side_effect = None
    stored.question.evidence = [evidence]
    llm.generate_json.return_value = QuestionBatch(items=[stored])
    generated = questions.generate(
        QuestionRequest(
            context=context,
            unit_id=context.concepts[0].unit_id,
            concept_version_ids=[version.id],
            count=1,
            difficulty="easy",
        ),
        router,
    )
    assert generated.items[0].question.evidence == [evidence]
    llm.generate_json.return_value = EvaluationJudgments(
        items=[
            CriterionJudgment(
                checklist_item_id=item.id,
                status="explained",
                description="Explained",
                confidence="high",
                evidence=[evidence],
                learner_quote="habits",
            )
            for item in version.checklist_items
        ],
        next_action="Review",
    )
    report = evaluation.evaluate(
        EvaluationRequest(
            context=context,
            concept_version_id=version.id,
            history=[Message(role="user", content="habits")],
        ),
        router,
    )
    assert report.coverage_formula == "weighted-v1" and report.coverage == 1
    assert parser.requests_sent == 1  # Learning functions never reparse the document.
