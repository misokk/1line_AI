"""Feature contract tests use response fixtures only at the external model boundary."""

from unittest.mock import MagicMock

import pytest

from api.router import ModelRouter
from src import analysis as concept_analysis
from src import evaluation, grading, questions, tagging
from src.errors import InputError, InvalidResponseError, ModelRequiredError
from src.models import (
    AnalysisPart,
    AnalyzeRequest,
    AnswerJudgment,
    CriterionJudgment,
    DraftConcept,
    DraftUnit,
    EvaluationJudgments,
    EvaluationRequest,
    GradeRequest,
    Keywords,
    Message,
    MultipleChoiceAnswer,
    QuestionBatch,
    QuestionRequest,
    ShortAnswer,
    StoredQuestion,
    TagRequest,
)


@pytest.fixture
def transport():
    return MagicMock()


@pytest.fixture
def router(settings, transport):
    return ModelRouter(settings, {"gemini": transport})


def judgment(context, ordinal=0, status="explained"):
    return CriterionJudgment(
        checklist_item_id=context.concepts[0].version.checklist_items[ordinal].id,
        status=status,
        description="Supported learning judgment",
        confidence="high",
        evidence=context.concepts[0].version.evidence,
        learner_quote="habits",
    )


def test_analyze_drafts_and_merge(router, transport, context):
    source = context.concepts[0].version.evidence
    concept = DraftConcept(
        name="Simple Present",
        summary="Describes habits",
        checklist_items=["Explain habits"],
        keywords=["habits"],
        evidence=source,
    )
    second = context.chunks[1]
    second_evidence = source[0].model_copy(
        update={
            "chunk_id": second.id,
            "page_number": second.page_number,
            "excerpt": second.text[:50],
        }
    )
    transport.generate_json.side_effect = [
        AnalysisPart(units=[DraftUnit(title="Unit 1", concepts=[concept])]),
        AnalysisPart(
            units=[
                DraftUnit(
                    title="Unit 1",
                    concepts=[concept.model_copy(update={"evidence": [second_evidence]})],
                )
            ]
        ),
    ]
    result = concept_analysis.analyze(AnalyzeRequest(chunks=context.chunks), router)
    assert result.state == "draft" and result.units[0].concepts[0].name == "Simple Present"
    assert "concept_id" not in result.model_dump_json()
    assert len(result.units[0].concepts[0].evidence) == 2


def test_analysis_budget_before_call(router, transport, context):
    with pytest.raises(InputError, match="budget"):
        concept_analysis.analyze(
            AnalyzeRequest(chunks=context.chunks, chunk_chars=500, max_calls=1), router
        )
    transport.generate_json.assert_not_called()


def test_analysis_rejects_fake_chunk(router, transport, context):
    evidence = (
        context.concepts[0].version.evidence[0].model_copy(update={"excerpt": "not in source"})
    )
    draft = DraftConcept(
        name="Bad", summary="Bad", checklist_items=["Bad"], keywords=["bad"], evidence=[evidence]
    )
    transport.generate_json.return_value = AnalysisPart(
        units=[DraftUnit(title="U", concepts=[draft])]
    )
    with pytest.raises(InvalidResponseError):
        concept_analysis.analyze(AnalyzeRequest(chunks=context.chunks), router)


def test_empty_draft_is_distinct_and_never_retried(router, transport, context):
    transport.generate_json.return_value = AnalysisPart(units=[])
    with pytest.raises(InvalidResponseError, match=r"units=\[\]"):
        concept_analysis.analyze(AnalyzeRequest(chunks=context.chunks[:1], max_calls=1), router)
    transport.generate_json.assert_called_once()
    prompt = transport.generate_json.call_args.kwargs["prompt"]
    assert "No confirmed concept version is required" in prompt
    assert "allowed source chunks" not in prompt


def test_confirmed_feature_prompt_guard_is_preserved():
    from src.context import COMMON, make_prompt

    assert make_prompt("Explain", {}).startswith(COMMON)
    assert "immutable concept version" in make_prompt("Explain", {})


def test_question_generation_and_public_projection(router, transport, context, stored):
    transport.generate_json.return_value = QuestionBatch(items=[stored])
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=1,
        difficulty="easy",
        question_types=["multiple_choice"],
    )
    result = questions.generate(request, router)
    public = questions.public_question(result.items[0], context.chunks[0].id)
    assert not {"correct_answer", "explanation", "accepted_answers"} & public.model_dump().keys()
    with pytest.raises(InvalidResponseError, match="count"):
        questions.generate(request.model_copy(update={"count": 2}), router)


def test_mixed_question_types_have_explicit_plan_and_valid_answers(
    router, transport, context, stored
):
    import json

    short = StoredQuestion(
        question=stored.question.model_copy(
            update={
                "question_type": "short_answer",
                "options": [],
                "prompt": "Name the use shown in the source.",
            }
        ),
        correct_answer=ShortAnswer(type="short_answer", text="habits"),
        explanation="The source describes habitual actions.",
    )
    transport.generate_json.return_value = QuestionBatch(items=[stored, short])
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=2,
        difficulty="easy",
    )
    result = questions.generate(request, router)
    assert [item.question.question_type for item in result.items] == [
        "multiple_choice",
        "short_answer",
    ]
    prompt = transport.generate_json.call_args.kwargs["prompt"]
    data = json.loads(prompt.split("\nUNTRUSTED_JSON:\n", 1)[1])
    assert data["question_type_counts"] == {"multiple_choice": 1, "short_answer": 1}
    schema = transport.generate_json.call_args.kwargs["schema"].model_json_schema()
    assert schema["properties"]["items"]["minItems"] == 2
    assert schema["properties"]["items"]["maxItems"] == 2
    assert data["question_type_plan"] == [
        {"ordinal": 1, "question_type": "multiple_choice"},
        {"ordinal": 2, "question_type": "short_answer"},
    ]


def test_missing_question_type_reports_which_type_without_retry(router, transport, context, stored):
    second = stored.model_copy(deep=True)
    second.question.prompt = "Which answer describes habitual actions?"
    transport.generate_json.return_value = QuestionBatch(items=[stored, second])
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=2,
        difficulty="easy",
    )
    with pytest.raises(InvalidResponseError, match="short_answer"):
        questions.generate(request, router)
    transport.generate_json.assert_called_once()


def test_unified_wire_answers_are_validated_and_restore_existing_contract(
    router, transport, context, stored
):
    import json

    schema = questions.question_batch_schema(2, ("multiple_choice", "short_answer"), "easy")
    choice = stored.model_dump(mode="json")
    choice["correct_answer"]["text"] = None
    short = stored.model_dump(mode="json")
    short["question"].update(question_type="short_answer", options=[], prompt="Name the use.")
    short["correct_answer"] = {
        "type": "short_answer",
        "selected_option_ids": None,
        "text": "habits",
    }
    transport.generate_json.return_value = schema.model_validate_json(
        json.dumps({"items": [choice, short]})
    )
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=2,
        difficulty="easy",
    )
    result = questions.generate(request, router).model_dump(mode="json")
    assert set(result["items"][0]["correct_answer"]) == {"type", "selected_option_ids"}
    assert set(result["items"][1]["correct_answer"]) == {"type", "text"}
    # The wire model may decode option labels, but the domain still rejects unknown labels.
    choice["correct_answer"]["selected_option_ids"] = ["unknown"]
    transport.generate_json.return_value = schema.model_validate_json(
        json.dumps({"items": [choice, short]})
    )
    with pytest.raises(InvalidResponseError, match="question_answer_unknown_option"):
        questions.generate(request, router)


@pytest.mark.parametrize(("choice", "correct"), [("A", True), ("B", False)])
def test_python_choice_grading_no_model(context, stored, choice, correct):
    request = GradeRequest(
        context=context,
        stored_question=stored,
        answer=MultipleChoiceAnswer(type="multiple_choice", selected_option_ids=[choice]),
    )
    assert grading.grade(request).is_correct == correct


def test_unknown_option_rejected(context, stored):
    with pytest.raises(InputError):
        grading.grade(
            GradeRequest(
                context=context,
                stored_question=stored,
                answer=MultipleChoiceAnswer(type="multiple_choice", selected_option_ids=["Z"]),
            )
        )


def test_semantic_answer_requires_model_and_evidence(router, transport, context, stored):
    short = StoredQuestion(
        question=stored.question.model_copy(
            update={"question_type": "short_answer", "options": []}
        ),
        correct_answer=ShortAnswer(type="short_answer", text="habits"),
        explanation="Habitual actions",
    )
    exact = GradeRequest(
        context=context,
        stored_question=short,
        answer=ShortAnswer(type="short_answer", text=" ＨＡＢＩＴＳ "),
    )
    assert grading.grade(exact).is_correct
    request = exact.model_copy(
        update={"answer": ShortAnswer(type="short_answer", text="regular actions")}
    )
    with pytest.raises(ModelRequiredError):
        grading.grade(request)
    transport.generate_json.return_value = AnswerJudgment(
        status="correct", feedback="Equivalent", evidence=stored.question.evidence
    )
    assert grading.grade(request, router).is_correct
    transport.generate_json.return_value = AnswerJudgment(
        status="unverified", feedback="Unknown", evidence=[]
    )
    with pytest.raises(InvalidResponseError):
        grading.grade(request, router)


def test_evaluation_only_uses_user_and_recomputes_weighted_v1(router, transport, context):
    transport.generate_json.return_value = EvaluationJudgments(
        items=[judgment(context), judgment(context, 1, "partial")],
        next_action="Practice -s.",
    )
    request = EvaluationRequest(
        context=context,
        concept_version_id=context.concepts[0].version.id,
        history=[
            Message(role="user", content="habits"),
            Message(role="assistant", content="private answer"),
        ],
    )
    result = evaluation.evaluate(request, router)
    assert result.coverage == 0.75 and result.assessable_count == 2 and result.weighted_score == 1.5
    assert "private answer" not in transport.generate_json.call_args.kwargs["prompt"]
    assert result.weak_concept_ids == [context.concepts[0].concept_id]


def test_evaluation_no_evidence_is_unverified(router, transport, context):
    items = [judgment(context, i, "error").model_copy(update={"evidence": []}) for i in range(2)]
    transport.generate_json.return_value = EvaluationJudgments(
        items=items, next_action="Review source."
    )
    result = evaluation.evaluate(
        EvaluationRequest(
            context=context,
            concept_version_id=context.concepts[0].version.id,
            history=[Message(role="user", content="habits")],
        ),
        router,
    )
    assert result.coverage is None and result.assessable_count == 0 and result.weighted_score == 0


def test_evaluation_missing_duplicate_or_fake_criterion_rejected(router, transport, context):
    for items in [[judgment(context)], [judgment(context), judgment(context)]]:
        transport.generate_json.return_value = EvaluationJudgments(
            items=items, next_action="Practice"
        )
        with pytest.raises(InvalidResponseError, match="exactly one"):
            evaluation.evaluate(
                EvaluationRequest(
                    context=context,
                    concept_version_id=context.concepts[0].version.id,
                    history=[Message(role="user", content="habits")],
                ),
                router,
            )


def test_keyword_nfkc_lowercase_and_dedup(router, transport, context, stored):
    transport.generate_json.return_value = Keywords(
        keywords=[" ＨＡＢＩＴＳ ", "habits", "Simple Present"]
    )
    result = tagging.suggest(TagRequest(question=stored.question, context=context), router)
    assert result.keywords == ["habits", "simple present"]
