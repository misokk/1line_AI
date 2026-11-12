from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from src import evaluation, questions
from src.context import validate_context
from src.errors import InputError
from src.models import (
    CriterionJudgment,
    EvaluationJudgments,
    ExplanationContent,
    MultipleChoiceAnswer,
    QuestionResult,
    StudyContext,
)


@pytest.fixture(scope="module")
def contract():
    root = Path(__file__).resolve().parents[1]
    path = next(
        (
            p
            for p in (root / "swagger-api-final.yaml", root / "test_data/swagger-api-final.yaml")
            if p.is_file()
        ),
        None,
    )
    if path is None:
        pytest.skip("Original swagger-api-final.yaml is absent; public YAML contract not verified.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_contract(contract, name, instance):
    schema = {"$ref": f"#/components/schemas/{name}", "components": contract["components"]}
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(
        instance.model_dump(mode="json")
    )


def test_source_and_confirmed_version_match_actual_yaml(contract, context):
    evidence = context.concepts[0].version.evidence
    validate_contract(contract, "SourceEvidence", evidence[0])
    validate_contract(contract, "ConceptVersion", context.concepts[0].version)


def test_public_question_and_question_result_match_yaml(contract, stored, context):
    public = questions.public_question(stored, uuid4())
    validate_contract(contract, "Question", public)
    assert "correct_answer" not in public.model_dump()
    assert "explanation" not in public.model_dump()
    result = QuestionResult(
        is_correct=True,
        feedback="Correct",
        correct_answer=MultipleChoiceAnswer(type="multiple_choice", selected_option_ids=["A"]),
        explanation=stored.explanation,
        evidence=stored.question.evidence,
    )
    validate_contract(contract, "QuestionResult", result)


def test_explanation_content_matches_public_yaml(contract, context):
    ref = context.concepts[0].version.evidence[0]
    result = ExplanationContent(
        title="Source explanation",
        explanation_markdown=f"Habits. [근거: {ref.chunk_id}]",
        evidence=[ref],
    )
    validate_contract(contract, "LearningExplanationContent", result)


def test_report_projection_matches_yaml_without_generating_db_ids(contract, context):
    concept = context.concepts[0]
    judgments = EvaluationJudgments(
        items=[
            CriterionJudgment(
                checklist_item_id=c.id,
                status="explained",
                description="Explained",
                confidence="high",
                evidence=concept.version.evidence,
                learner_quote="habits",
            )
            for c in concept.version.checklist_items
        ],
        next_action="Practice -s",
    )
    result = evaluation.calculate_report(judgments, concept)
    report_id, session_id, action_id = uuid4(), uuid4(), uuid4()
    ids = {i.checklist_item_id: uuid4() for i in result.items}
    report = evaluation.session_report(
        result,
        report_id=report_id,
        session_id=session_id,
        version_no=concept.version.version_no,
        action_id=action_id,
        item_ids=ids,
    )
    validate_contract(contract, "SessionReport", report)
    assert report.id == report_id and report.action_id == action_id
    assert {i.id for i in report.items} == set(ids.values())
    assert report.coverage == 1 and report.coverage_formula == "weighted-v1"
    with pytest.raises(InputError):
        evaluation.session_report(
            result,
            report_id=report_id,
            session_id=session_id,
            version_no=1,
            action_id=action_id,
            item_ids={},
        )


def test_draft_rejected_and_past_confirmed_version_accepted(context):
    assert context.concepts[0].version.is_current is False
    validate_context(context)
    body = context.model_dump(mode="json")
    body["concepts"][0]["version"]["is_confirmed"] = False
    with pytest.raises(ValidationError, match="CONCEPT_NOT_CONFIRMED"):
        StudyContext.model_validate(body)


def test_forged_metadata_or_material_relations_rejected(context):
    tampered = context.model_copy(deep=True)
    tampered.concepts[0].version.evidence[0].material_name = "other file"
    with pytest.raises(InputError):
        validate_context(tampered)
