"""Latest planning requirements, with the official SDK over an in-memory HTTP transport."""

import json
from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from api.config import ModelTarget, Settings
from api.router import ModelRouter
from src import evaluation, explanation, questions
from src.errors import InputError, InvalidResponseError
from src.models import (
    DetailedCriterionJudgment,
    DetailedEvaluationRequest,
    DetailedJudgments,
    ExplanationRequest,
    Message,
    QuestionRequest,
)
from src.study_time import WorkloadRequest, estimate_workload
from tests.test_gemini import adapter, envelope


def gemini_settings():
    return Settings(
        live_enabled=True,
        model_allowlist={
            "gemini:test": ModelTarget(provider="gemini", model="gemini-3.8-flash", available=True)
        },
        default_model_id="gemini:test",
    )


def test_all_text_roles_use_one_default_model(tmp_path, monkeypatch):
    from api.config import Role

    for key in (
        "AI_MODEL_ALLOWLIST",
        "AI_DEFAULT_MODEL_ID",
        "AI_MODEL_ROUTES",
        "GEMINI_TEXT_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env(tmp_path / "absent.env")
    assert {settings.target_for(role).model for role in Role} == {"gemini-3.8-flash"}
    assert {settings.target_for(role).provider for role in Role} == {"gemini"}


@pytest.mark.parametrize("count,fail_at", [(5, None), (20, None), (20, 4)])
def test_mixed_questions_sdk_schema_and_final_source_contract(context, count, fail_at):
    calls = []
    sequence = 0

    def handler(request):
        nonlocal sequence
        body = json.loads(request.content)
        calls.append(body)
        thinking = body["generationConfig"]["thinkingConfig"]
        assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "LOW"
        assert body["generationConfig"]["maxOutputTokens"] == 8192
        if len(calls) == fail_at:
            return httpx.Response(200, json=envelope({"items": []}, status="MAX_TOKENS"))
        schema = body["generationConfig"]["responseJsonSchema"]
        q_schema = schema["properties"]["items"]["items"]["properties"]["question"]
        level = q_schema["properties"]["difficulty"]["enum"][0]
        # Read the exact batch/type plan sent by the business function.
        prompt = body["contents"][0]["parts"][0]["text"]
        start = prompt.index('"question_type_plan":')
        plan = json.JSONDecoder().raw_decode(prompt[start:].split(":", 1)[1].lstrip())[0]
        assert len(plan) <= 5
        items = []
        for entry in plan:
            sequence += 1
            kind = entry["question_type"]
            items.append(
                {
                    "question": {
                        "question_type": kind,
                        "concept_indices": [0],
                        "evidence_indices": [0],
                        "prompt": f"Habits question {sequence}?",
                        "options": [
                            {"id": "A", "text": "Habits"},
                            {"id": "B", "text": "Only future"},
                        ]
                        if kind == "multiple_choice"
                        else [],
                        "difficulty": level,
                    },
                    "correct_answer": {
                        "type": kind,
                        "selected_option_ids": ["A"] if kind == "multiple_choice" else None,
                        "text": "habits" if kind == "short_answer" else None,
                    },
                    "explanation": "The source describes habits.",
                    "accepted_answers": ["habitual actions"] if kind == "short_answer" else [],
                }
            )
        return httpx.Response(200, json=envelope({"items": items}))

    client = adapter(handler)
    router = ModelRouter(gemini_settings(), {"gemini": client})
    levels = [["easy", "medium", "hard"][i % 3] for i in range(count)]
    concept = context.concepts[0]
    request = QuestionRequest(
        context=context,
        unit_id=concept.unit_id,
        concept_version_ids=[concept.version.id],
        count=count,
        difficulty="mixed",
        difficulty_plan=levels,
    )
    try:
        if fail_at:
            with pytest.raises(InvalidResponseError, match="incomplete"):
                questions.generate(request, router)
            assert len(calls) == fail_at
            return
        result = questions.generate(request, router)
    finally:
        client.close()
    assert len(result.items) == count and 1 < len(calls) <= 6
    assert [i.question.difficulty for i in result.items] == levels
    assert [i.question.question_type for i in result.items] == [
        ["multiple_choice", "short_answer"][i % 2] for i in range(count)
    ]
    for item in result.items:
        assert item.question.concept_version_ids == [concept.version.id]
        assert item.question.evidence == concept.version.evidence
        assert item.explanation and item.correct_answer


def test_mixed_requires_explicit_allocation(context):
    concept = context.concepts[0]
    with pytest.raises(ValidationError, match="explicit plan"):
        QuestionRequest(
            context=context,
            unit_id=concept.unit_id,
            concept_version_ids=[concept.version.id],
            count=5,
            difficulty="mixed",
        )


@pytest.mark.parametrize("role,model,low", [
    ("question_generation", "gemini-3.7-flash", True),
    ("question_generation", "gemini-3.8-flash", True),
    ("evaluation", "gemini-3.7-flash", False),
    ("material_analysis", "gemini-3.7-flash", False),
    ("explanation", "gemini-3.7-flash", False),
    ("question_tagging", "gemini-3.7-flash", False),
    ("question_generation", "gemini-2.5-flash", False),
])
def test_question_thinking_is_limited_to_supported_models_and_role(role, model, low):
    from src.models import Keywords

    calls = []

    def handler(request):
        config = json.loads(request.content)["generationConfig"]
        calls.append(config)
        if low:
            thinking = config["thinkingConfig"]
            assert set(thinking) in ({"thinkingLevel"}, {"thinking_level"})
            assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "LOW"
        else:
            assert "thinkingConfig" not in config
        return httpx.Response(200, json=envelope({"keywords": ["grammar"]}))

    settings = Settings(
        default_model_id="gemini:test",
        model_allowlist={
            "gemini:test": ModelTarget(provider="gemini", model=model, available=True)
        },
    )
    client = adapter(handler)
    try:
        assert ModelRouter(settings, {"gemini": client}).generate_json(
            role, prompt="Synthetic input", schema=Keywords,
        ).keywords == ["grammar"]
    finally:
        client.close()
    assert len(calls) == 1


def test_explanation_indices_do_not_coerce_boolean():
    with pytest.raises(ValidationError):
        explanation.GroundedText(text="Source", citations=[True])


def test_explanation_wire_bounds_do_not_mutate_original_schema():
    from api.gemini import compatible_schema

    original = explanation.ExplanationSections.model_json_schema()
    wire = compatible_schema(explanation._GeminiExplanationSections.model_json_schema())

    def count(node, key):
        if isinstance(node, dict):
            return int(key in node) + sum(count(v, key) for v in node.values())
        return sum(count(v, key) for v in node) if isinstance(node, list) else 0

    assert count(compatible_schema(original), "maxItems") == 9
    assert count(wire, "maxItems") == 0
    assert count(wire, "minItems") == 5
    assert count(wire, "anyOf") == 1
    assert wire["additionalProperties"] is False
    assert "At most 15 items" in wire["properties"]["key_points"]["description"]
    assert explanation.ExplanationSections.model_json_schema() == original


@pytest.mark.parametrize("fault", ["key_points", "examples", "citations"])
def test_explanation_wire_relaxation_still_rejects_excess_items(context, fault):
    from jsonschema import Draft202012Validator

    raw = {
        "title": "Simple present",
        "definition": {"text": "Habits.", "citations": [0]},
        "key_points": [], "rules": [], "cautions": [], "examples": [],
    }
    if fault == "citations":
        raw["definition"]["citations"] = list(range(21))
    elif fault == "examples":
        raw["examples"] = [
            {"text": "Habits.", "citations": [0], "source_example": "habits"}
            for _ in range(11)
        ]
    else:
        raw["key_points"] = [{"text": "Habits.", "citations": [0]} for _ in range(16)]
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        config = body["generationConfig"]
        assert set(config) == {"responseMimeType", "responseJsonSchema", "maxOutputTokens"}
        assert config["maxOutputTokens"] == 8192
        # The simpler wire grammar accepts it; original Pydantic must still reject it.
        Draft202012Validator(config["responseJsonSchema"]).validate(raw)
        return httpx.Response(200, json=envelope(raw))

    client = adapter(handler)
    try:
        with pytest.raises(InvalidResponseError):
            explanation.generate(
                ExplanationRequest(
                    context=context, concept_version_id=context.concepts[0].version.id
                ),
                ModelRouter(gemini_settings(), {"gemini": client}),
            )
    finally:
        client.close()
    assert len(calls) == 1


def test_explanation_nullable_definition_and_empty_sections_keep_contract(context):
    raw = {
        "title": "Simple present", "definition": None,
        "key_points": [{"text": "Habits.", "citations": [0]}],
        "rules": [], "cautions": [], "examples": [],
    }
    client = adapter(lambda request: httpx.Response(200, json=envelope(raw)))
    try:
        result = explanation.generate(
            ExplanationRequest(context=context, concept_version_id=context.concepts[0].version.id),
            ModelRouter(gemini_settings(), {"gemini": client}),
        )
    finally:
        client.close()
    assert set(result.model_dump()) == {"title", "explanation_markdown", "evidence"}
    assert result.evidence[0].chunk_id == context.chunks[0].id


def test_grounded_explanation_sdk_and_backend_content(context):
    ref = context.concepts[0].version.evidence[0]
    raw = {
        "title": "Simple present",
        "definition": {"text": "습관을 표현한다.", "citations": [0]},
        "key_points": [],
        "rules": [],
        "cautions": [],
        "examples": [],
    }
    client = adapter(lambda req: httpx.Response(200, json=envelope(raw)))
    try:
        result = explanation.generate(
            ExplanationRequest(context=context, concept_version_id=context.concepts[0].version.id),
            ModelRouter(gemini_settings(), {"gemini": client}),
        )
    finally:
        client.close()
    assert set(result.model_dump()) == {"title", "explanation_markdown", "evidence"}
    assert f"[근거: {ref.chunk_id}]" in result.explanation_markdown
    assert result.evidence[0].chunk_id == ref.chunk_id


@pytest.mark.parametrize("fault", ["citation", "fabricated_example", "html"])
def test_explanation_rejects_invalid_grounding(context, fault):
    model = MagicMock()
    raw = {
        "title": "T",
        "definition": {"text": "source", "citations": [0]},
        "key_points": [],
        "rules": [],
        "cautions": [],
        "examples": [],
    }
    if fault == "citation":
        raw["definition"]["citations"] = [999]
    elif fault == "html":
        raw["definition"]["text"] = "<script>unsafe</script>"
    else:
        raw["examples"] = [
            {"text": "invented", "citations": [0], "source_example": "not in original source"}
        ]
    model.generate_json.return_value = explanation.ExplanationSections.model_validate(raw)
    with pytest.raises(InvalidResponseError):
        explanation.generate(
            ExplanationRequest(context=context, concept_version_id=context.concepts[0].version.id),
            ModelRouter(gemini_settings(), {"gemini": model}),
        )


def test_workload_pages_concepts_and_days_no_schedule(context):
    result = estimate_workload(
        WorkloadRequest(
            total_pages=32, context=context, today=date(2026, 11, 10), exam_date=date(2026, 11, 15)
        )
    )
    assert result.remaining_days == 5 and result.recommended_daily_pages == 7
    assert result.recommended_daily_concepts == 1
    assert result.time_estimate.recommended_daily_minutes >= 0
    assert "schedule" not in result.model_dump()


def detailed_fixture(context, status="error"):
    concept = context.concepts[0]
    return DetailedJudgments(
        items=[
            DetailedCriterionJudgment(
                checklist_item_id=criterion.id,
                status=status,
                description="Source-grounded finding",
                confidence="high",
                evidence=concept.version.evidence,
                learner_quote="Only future" if status == "error" else "",
                correct_explanation="It describes habits.",
                improvement="Review habitual actions.",
            )
            for criterion in concept.version.checklist_items
        ],
        next_action="Review the source",
    )


def test_detailed_evaluation_preserves_weighted_contract_and_comparison(context):
    model = MagicMock()
    model.generate_json.return_value = detailed_fixture(context)
    router = ModelRouter(gemini_settings(), {"gemini": model})
    request = DetailedEvaluationRequest(
        context=context,
        concept_version_id=context.concepts[0].version.id,
        history=[
            Message(role="user", content="Only future"),
            Message(role="assistant", content="PRIVATE_ASSISTANT"),
        ],
    )
    first = evaluation.evaluate_detailed(request, router)
    assert first.report.coverage == 0 and first.report.coverage_formula == "weighted-v1"
    assert all(
        i.learner_quote == "Only future" and i.correct_explanation and i.improvement
        for i in first.feedback
    )
    assert "PRIVATE_ASSISTANT" not in model.generate_json.call_args.kwargs["prompt"]
    second = evaluation.evaluate_detailed(
        request.model_copy(update={"previous_result": first.report}), router
    )
    assert second.comparison.comparable and second.comparison.coverage_delta == 0
    wrong = first.report.model_copy(update={"coverage": 1})
    with pytest.raises(InputError, match="weighted-v1"):
        evaluation.compare_reports(first.report, wrong)


@pytest.mark.parametrize("detailed", [False, True])
def test_gemini_evaluation_wire_keeps_final_limits_and_grounding(context, detailed):
    from jsonschema import Draft202012Validator

    from api.gemini import compatible_schema
    from src.models import EvaluationJudgments, EvaluationRequest

    original_model = DetailedJudgments if detailed else EvaluationJudgments
    original_schema = original_model.model_json_schema()
    wire_model = (
        evaluation._GeminiDetailedJudgments
        if detailed else evaluation._GeminiEvaluationJudgments
    )
    wire = compatible_schema(wire_model.model_json_schema())
    assert '"maxItems"' not in json.dumps(wire)
    assert wire["additionalProperties"] is False
    excess = detailed_fixture(context).model_dump(mode="json")
    if not detailed:
        for item in excess["items"]:
            item.pop("correct_explanation")
            item.pop("improvement")
    excess["items"] = [excess["items"][0]] * 51
    Draft202012Validator(wire).validate(excess)
    with pytest.raises(ValidationError):
        wire_model.model_validate(excess)

    valid = detailed_fixture(context).model_dump(mode="json")
    if not detailed:
        for item in valid["items"]:
            item.pop("correct_explanation")
            item.pop("improvement")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        config = body["generationConfig"]
        assert '"maxItems"' not in json.dumps(config["responseJsonSchema"])
        assert "responseSchema" not in config
        prompt = body["contents"][0]["parts"][0]["text"]
        assert "PRIVATE_ASSISTANT" not in prompt
        return httpx.Response(200, json=envelope(valid))

    request_model = DetailedEvaluationRequest if detailed else EvaluationRequest
    request = request_model(
        context=context, concept_version_id=context.concepts[0].version.id,
        history=[Message(role="user", content="Only future"),
                 Message(role="assistant", content="PRIVATE_ASSISTANT")],
    )
    client = adapter(handler)
    try:
        fn = evaluation.evaluate_detailed if detailed else evaluation.evaluate
        result = fn(request, ModelRouter(gemini_settings(), {"gemini": client}))
    finally:
        client.close()
    report = result.report if detailed else result
    assert len(calls) == 1
    assert report.coverage_formula == "weighted-v1" and report.coverage == 0
    assert report.concept_version_id == context.concepts[0].version.id
    assert original_model.model_json_schema() == original_schema


def test_detailed_error_requires_correction(context):
    model = MagicMock()
    result = detailed_fixture(context)
    result.items[0].correct_explanation = ""
    model.generate_json.return_value = result
    request = DetailedEvaluationRequest(
        context=context,
        concept_version_id=context.concepts[0].version.id,
        history=[Message(role="user", content="Only future")],
    )
    with pytest.raises(InvalidResponseError, match="correction"):
        evaluation.evaluate_detailed(request, ModelRouter(gemini_settings(), {"gemini": model}))


def test_analysis_and_semantic_grading_official_sdk(context, stored):
    from src import analysis, grading
    from src.models import AnalyzeRequest, GradeRequest, ShortAnswer
    from tests.test_analysis_diagnostics import indexed_data

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            data = indexed_data([0])
        else:
            data = {
                "status": "correct",
                "feedback": "Equivalent habitual action meaning",
                "evidence": [e.model_dump(mode="json") for e in stored.question.evidence],
            }
        return httpx.Response(200, json=envelope(data))

    client = adapter(handler)
    router = ModelRouter(gemini_settings(), {"gemini": client})
    try:
        draft = analysis.analyze(AnalyzeRequest(chunks=context.chunks[:1]), router)
        assert (
            draft.state == "draft"
            and draft.units[0].concepts[0].evidence[0].chunk_id == context.chunks[0].id
        )
        short = stored.model_copy(deep=True)
        short.question.question_type = "short_answer"
        short.question.options = []
        short.correct_answer = ShortAnswer(type="short_answer", text="habits")
        result = grading.grade(
            GradeRequest(
                context=context,
                stored_question=short,
                answer=ShortAnswer(type="short_answer", text="regular habitual actions"),
            ),
            router,
        )
        assert result.is_correct and result.evidence == stored.question.evidence
    finally:
        client.close()
    assert len(calls) == 2


@pytest.mark.parametrize("status", ["correct", "incorrect", "unverified", "foreign_evidence"])
def test_semantic_grading_wire_preserves_verdict_and_real_evidence(context, stored, status):
    from jsonschema import Draft202012Validator

    from src import grading
    from src.models import AnswerJudgment, GradeRequest, ShortAnswer

    original_schema = AnswerJudgment.model_json_schema()
    short = stored.model_copy(deep=True)
    short.question.question_type = "short_answer"
    short.question.options = []
    short.correct_answer = ShortAnswer(type="short_answer", text="habits")
    data = {
        "status": "correct" if status == "foreign_evidence" else status,
        "feedback": "Grounded judgment.",
        "evidence": [e.model_dump(mode="json") for e in short.question.evidence],
    }
    if status == "foreign_evidence":
        data["evidence"][0]["page_number"] += 1
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        wire = body["generationConfig"]["responseJsonSchema"]
        assert "maxItems" not in wire["properties"]["evidence"]
        assert wire["properties"]["evidence"]["type"] == "array"
        assert wire["additionalProperties"] is False
        assert set(wire["required"]) == {"status", "feedback", "evidence"}
        Draft202012Validator(wire).validate(data)
        excess = {**data, "evidence": data["evidence"] * 201}
        Draft202012Validator(wire).validate(excess)
        with pytest.raises(ValidationError):
            grading._GeminiAnswerJudgment.model_validate(excess)
        return httpx.Response(200, json=envelope(data))

    request = GradeRequest(
        context=context, stored_question=short,
        answer=ShortAnswer(type="short_answer", text="regular habitual actions"),
    )
    client = adapter(handler)
    try:
        router = ModelRouter(gemini_settings(), {"gemini": client})
        if status in ("unverified", "foreign_evidence"):
            with pytest.raises(InvalidResponseError):
                grading.grade(request, router)
        else:
            result = grading.grade(request, router)
            assert result.is_correct is (status == "correct")
            assert result.evidence == short.question.evidence
            assert result.correct_answer == short.correct_answer
            assert result.explanation == short.explanation
    finally:
        client.close()
    assert len(calls) == 1
    assert AnswerJudgment.model_json_schema() == original_schema


def test_detailed_all_five_statuses_official_sdk(context):
    from uuid import uuid4

    from src.models import ConceptChecklistItem

    context = context.model_copy(deep=True)
    concept = context.concepts[0]
    statuses = ["explained", "partial", "omission", "error", "unverified"]
    concept.version.checklist_items = [
        ConceptChecklistItem(
            id=uuid4(), ordinal=i + 1, criterion=f"Habitual action criterion {i + 1}"
        )
        for i in range(5)
    ]
    data = {
        "items": [
            {
                "checklist_item_id": str(c.id),
                "status": status,
                "description": "Mocked source-grounded finding",
                "confidence": "medium",
                "evidence": [e.model_dump(mode="json") for e in concept.version.evidence]
                if status != "unverified"
                else [],
                "learner_quote": "habits"
                if status in ("explained", "partial")
                else "Only future"
                if status == "error"
                else "",
                "correct_explanation": "The source describes habitual actions.",
                "improvement": "Review the stated conditions.",
            }
            for c, status in zip(concept.version.checklist_items, statuses, strict=True)
        ],
        "next_action": "Practice the weak criteria",
    }
    client = adapter(lambda req: httpx.Response(200, json=envelope(data)))
    try:
        result = evaluation.evaluate_detailed(
            DetailedEvaluationRequest(
                context=context,
                concept_version_id=concept.version.id,
                history=[Message(role="user", content="habits; Only future")],
            ),
            ModelRouter(gemini_settings(), {"gemini": client}),
        )
    finally:
        client.close()
    assert [i.status for i in result.report.items] == statuses
    assert result.report.coverage == 0.375 and result.report.assessable_count == 4
    assert result.report.weighted_score == 1.5
    assert result.feedback[-1].correct_explanation == ""
    assert [f.checklist_item_id for f in result.feedback] == [
        c.id for c in concept.version.checklist_items
    ]
    other = result.report.model_copy(update={"concept_version_id": uuid4()})
    assert evaluation.compare_reports(result.report, other).reason == "version_mismatch"
