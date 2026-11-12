import json
from unittest.mock import MagicMock

import pytest

from api.router import ModelRouter
from src import analysis, evaluation, questions
from src.errors import InputError, InvalidResponseError
from src.models import (
    AnalysisPart,
    AnalyzeRequest,
    CriterionJudgment,
    DraftConcept,
    DraftUnit,
    EvaluationJudgments,
    EvaluationRequest,
    Message,
    QuestionRequest,
    ShortAnswer,
)


def indexed_item(stored):
    data = (
        stored.model_dump(mode="json")
        if hasattr(stored, "model_dump")
        else json.loads(json.dumps(stored))
    )
    question = data["question"]
    question.pop("concept_ids")
    question.pop("concept_version_ids")
    question.pop("evidence")
    question.update(concept_indices=[0], evidence_indices=[0])
    answer = data["correct_answer"]
    if answer["type"] == "multiple_choice":
        answer["text"] = None
    else:
        answer["selected_option_ids"] = None
    return data


def indexed_batch(kwargs, items):
    from api.base import validate_json

    return validate_json(json.dumps({"items": [indexed_item(x) for x in items]}), kwargs["schema"])


def budget_router(settings, tokens=900):
    settings = settings.model_copy(update={"max_output_tokens": tokens})
    transport = MagicMock()
    return ModelRouter(settings, {"gemini": transport}), transport


def test_questions_split_validate_types_merge_and_reject_duplicate(settings, context, stored):
    router, transport = budget_router(settings)
    short = stored.model_copy(deep=True)
    short.question.question_type = "short_answer"
    short.question.options = []
    short.question.prompt = "Name the use."
    short.correct_answer = ShortAnswer(type="short_answer", text="habits")
    items = iter([stored, short])
    transport.generate_json.side_effect = lambda **kwargs: indexed_batch(kwargs, [next(items)])
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=2,
        difficulty="easy",
    )
    result = questions.generate(request, router)
    assert len(result.items) == 2 and transport.generate_json.call_count == 2
    assert [q.question.question_type for q in result.items] == ["multiple_choice", "short_answer"]
    for call in transport.generate_json.call_args_list:
        schema = call.kwargs["schema"].model_json_schema()
        assert schema["properties"]["items"]["maxItems"] == 1
    assert (
        str(stored.question.concept_version_ids[0])
        in transport.generate_json.call_args.kwargs["prompt"]
    )
    assert result.items[0].question.evidence == stored.question.evidence
    duplicate = short.model_copy(deep=True)
    duplicate.question.prompt = stored.question.prompt
    items = iter([stored, duplicate])
    transport.generate_json.side_effect = lambda **kwargs: indexed_batch(kwargs, [next(items)])
    with pytest.raises(InvalidResponseError, match="duplicate"):
        questions.generate(request, router)


@pytest.mark.parametrize("violation", ["evidence", "type", "version", "count"])
def test_split_question_wrong_type_or_evidence_stops_before_next_call(
    settings, context, stored, violation
):
    router, transport = budget_router(settings)

    def respond(**kwargs):
        data = indexed_item(stored)
        if violation == "evidence":
            data["question"]["evidence_indices"] = [9999]
        elif violation == "version":
            data["question"]["concept_indices"] = [9999]
        elif violation == "type":
            data["question"]["question_type"] = "short_answer"
        from api.base import validate_json

        return validate_json(
            json.dumps({"items": [data, data] if violation == "count" else [data]}),
            kwargs["schema"],
        )

    transport.generate_json.side_effect = respond
    request = QuestionRequest(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=2,
        difficulty="easy",
    )
    with pytest.raises(InvalidResponseError):
        questions.generate(request, router)
    transport.generate_json.assert_called_once()


def test_analysis_splits_source_without_losing_text_ids_or_table(settings, context):
    source = context.chunks[0].model_copy(deep=True)
    source.text = (
        "A paragraph with learning content.\n\n" * 40
    ) + "| x | y |\n|---|---|\n| 1 | 2 |"
    groups = analysis.analysis_groups([source], 12000, 900)
    assert len(groups) > 1
    assert "".join(g[0].text for g in groups) == source.text
    assert all(g[0].id == source.id and g[0].page_number == source.page_number for g in groups)
    assert sum("| 1 | 2 |" in g[0].text for g in groups) == 1
    router, transport = budget_router(settings)
    with pytest.raises(InputError, match="No provider request"):
        analysis.analyze(AnalyzeRequest(chunks=[source], max_calls=1), router)
    transport.generate_json.assert_not_called()


def test_analysis_split_merge_checks_schema_and_exact_evidence(settings, context):
    router, transport = budget_router(settings)
    source = context.chunks[0].model_copy(deep=True)
    sentence = context.concepts[0].version.evidence[0].excerpt
    source.text = (sentence + "\n\n") * 40
    evidence = context.concepts[0].version.evidence
    part = AnalysisPart(
        units=[
            DraftUnit(
                title="Present",
                concepts=[
                    DraftConcept(
                        name="Habits",
                        summary="Habits",
                        checklist_items=["Explain habits"],
                        keywords=["habits"],
                        evidence=evidence,
                    )
                ],
            )
        ]
    )

    def respond(**kwargs):
        data = part.model_dump(mode="json")
        data["units"][0]["concepts"][0]["evidence"] = [0]
        return kwargs["schema"].model_validate(data)

    transport.generate_json.side_effect = respond
    result = analysis.analyze(AnalyzeRequest(chunks=[source], max_calls=10), router)
    assert transport.generate_json.call_count > 1
    assert len(result.units) == len(result.units[0].concepts) == 1
    assert result.units[0].concepts[0].evidence == evidence


def test_evaluation_splits_only_when_needed_and_keeps_fixed_ids_and_weighted_score(
    settings, context
):
    # Two criteria fit the default 900-token estimate; 700 forces one criterion per call.
    router, transport = budget_router(settings, 700)
    concept = context.concepts[0]
    parts = [
        EvaluationJudgments(
            items=[
                CriterionJudgment(
                    checklist_item_id=c.id,
                    status=status,
                    description="Fixture",
                    confidence="high",
                    evidence=concept.version.evidence,
                    learner_quote="habits",
                )
            ],
            next_action="Review",
        )
        for c, status in zip(concept.version.checklist_items, ["explained", "partial"], strict=True)
    ]
    transport.generate_json.side_effect = parts
    request = EvaluationRequest(
        context=context,
        concept_version_id=concept.version.id,
        history=[
            Message(role="user", content="habits"),
            Message(role="assistant", content="PRIVATE ANSWER"),
        ],
    )
    result = evaluation.evaluate(request, router)
    assert transport.generate_json.call_count == 2
    assert result.coverage == 0.75 and result.weighted_score == 1.5
    assert result.concept_version_id == concept.version.id
    assert [i.checklist_item_id for i in result.items] == [
        c.id for c in concept.version.checklist_items
    ]
    assert all(
        "PRIVATE ANSWER" not in c.kwargs["prompt"] for c in transport.generate_json.call_args_list
    )
    transport.generate_json.side_effect = [parts[1]]
    with pytest.raises(InvalidResponseError, match="out-of-batch"):
        evaluation.evaluate(request, router)


@pytest.mark.parametrize(
    "field,value",
    [
        ("concept_indices", [999]),
        ("evidence_indices", [999]),
        ("concept_indices", [True]),
        ("evidence_indices", ["0"]),
        ("concept_indices", [0, 0]),
        ("evidence_indices", [0, 0]),
    ],
)
def test_question_reference_indices_are_strict_and_bounded(stored, field, value):
    from api.base import validate_json
    from src.questions import indexed_question_schema

    schema = indexed_question_schema(1, ("multiple_choice",), "easy", 1, 1)
    data = indexed_item(stored)
    data["question"][field] = value
    with pytest.raises(InvalidResponseError):
        validate_json(json.dumps({"items": [data]}), schema)


def test_question_reference_must_belong_to_selected_version(context, stored):
    from uuid import uuid4

    from src.questions import expand_question_references, indexed_question_schema, question_catalog

    first = context.concepts[0]
    second = first.model_copy(deep=True)
    second.concept_id, second.version.id = uuid4(), uuid4()
    evidence = first.version.evidence[0].model_copy(deep=True)
    evidence.chunk_id = context.chunks[1].id
    evidence.page_number = context.chunks[1].page_number
    evidence.excerpt = context.chunks[1].text
    second.version.evidence = [evidence]
    catalog, owners = question_catalog([first, second])
    schema = indexed_question_schema(1, ("multiple_choice",), "easy", 2, 2)
    data = indexed_item(stored)
    data["question"]["evidence_indices"] = [1]
    part = schema.model_validate({"items": [data]})
    with pytest.raises(InvalidResponseError, match="unselected concept version"):
        expand_question_references(part, [first, second], catalog, owners, schema)


def test_indexed_question_still_rejects_invalid_answer_option(context, stored):
    from src.questions import expand_question_references, indexed_question_schema, question_catalog

    concepts = context.concepts
    catalog, owners = question_catalog(concepts)
    schema = indexed_question_schema(1, ("multiple_choice",), "easy", 1, 1)
    data = indexed_item(stored)
    data["correct_answer"]["selected_option_ids"] = ["Z"]
    part = schema.model_validate({"items": [data]})
    with pytest.raises(InvalidResponseError, match="question_answer_unknown_option"):
        expand_question_references(part, concepts, catalog, owners, schema)
