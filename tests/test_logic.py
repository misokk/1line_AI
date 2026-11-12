from datetime import date
from itertools import product

import pytest
from pydantic import ValidationError

from src.errors import InputError, InvalidResponseError
from src.evaluation import calculate_report
from src.grading import arithmetic, normalize_answer, verify_answer
from src.models import (
    CriterionJudgment,
    EvaluationJudgments,
    MultipleChoiceAnswer,
    Option,
    StudyTimeRequest,
    TimeConcept,
)
from src.study_time import estimate_study_time


@pytest.mark.parametrize(("expression", "expected"), [("1/3 + 2/3", 1), ("2**3", 8), ("-5+2", -3)])
def test_safe_arithmetic(expression, expected):
    assert arithmetic(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "2**1000000", "1/0", "[1]*100", "x+1"])
def test_arithmetic_rejects_unbounded_or_executable_expressions(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        arithmetic(expression)


def test_math_check_is_explicitly_limited(stored):
    assert verify_answer(stored).status == "not_verified"
    q = stored.model_copy(deep=True)
    q.question.options = [Option(id="A", text="5"), Option(id="B", text="6")]
    q.arithmetic_expression = "2+3"
    assert verify_answer(q).status == "consistent"
    q.correct_answer = MultipleChoiceAnswer(type="multiple_choice", selected_option_ids=["B"])
    assert verify_answer(q).status == "inconsistent"


def test_answer_nfkc_matches_backend_normalization():
    assert normalize_answer("  ＨＡＢＩＴＳ \n NOW ") == "habits now"


@pytest.mark.parametrize(
    ("statuses", "coverage", "assessable", "weighted"),
    [
        (["explained", "partial"], 0.75, 2, 1.5),
        (["explained", "unverified"], 1, 1, 1),
        (["omission", "error"], 0, 2, 0),
        (["unverified", "unverified"], None, 0, 0),
    ],
)
def test_weighted_v1(statuses, coverage, assessable, weighted, context):
    c = context.concepts[0]
    result = calculate_report(
        EvaluationJudgments(
            items=[
                CriterionJudgment(
                    checklist_item_id=item.id,
                    status=statuses[i],
                    description="Learning judgment",
                    confidence="medium",
                    evidence=[],
                    learner_quote="",
                )
                for i, item in enumerate(c.version.checklist_items)
            ],
            next_action="Review",
        ),
        c,
    )
    assert result.coverage == coverage
    assert result.assessable_count == assessable and result.weighted_score == weighted


@pytest.mark.parametrize(
    "statuses", list(product(("explained", "partial", "omission", "error", "unverified"), repeat=2))
)
def test_all_25_status_combinations_preserved_from_legacy_runner(context, statuses):
    weights = {"explained": 1, "partial": 0.5, "omission": 0, "error": 0, "unverified": 0}
    concept = context.concepts[0]
    result = calculate_report(
        EvaluationJudgments(
            items=[
                CriterionJudgment(
                    checklist_item_id=criterion.id,
                    status=status,
                    description="Calculation fixture",
                    confidence="medium",
                    evidence=concept.version.evidence,
                    learner_quote="",
                )
                for criterion, status in zip(concept.version.checklist_items, statuses, strict=True)
            ],
            next_action="Review",
        ),
        concept,
    )
    count = sum(status != "unverified" for status in statuses)
    score = sum(weights[status] for status in statuses)
    assert result.coverage == (round(score / count, 4) if count else None)
    assert result.weighted_score == score and result.assessable_count == count
    assert all(item.concept_version_id == concept.version.id for item in result.items)


def test_weighted_v1_rounds_four_places_and_counts_all_criteria(context):
    c = context.concepts[0]
    items = [
        CriterionJudgment(
            checklist_item_id=i.id,
            status="omission",
            description="Missing",
            confidence="high",
            evidence=[],
            learner_quote="",
        )
        for i in c.version.checklist_items
    ]
    with pytest.raises(InvalidResponseError):
        calculate_report(EvaluationJudgments(items=items[:1], next_action="Review"), c)


@pytest.mark.parametrize("exam", [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)])
def test_imminent_study_time_without_daily_time_input(exam, context):
    request = StudyTimeRequest(
        exam_date=exam,
        today=date(2026, 11, 3),
        concepts=[
            TimeConcept(
                concept_version_id=context.concepts[0].version.id,
                name="Present",
                remaining_criteria=2,
                remaining_questions=3,
                review_cycles=1,
                weakness=1,
            )
        ],
    )
    result = estimate_study_time(request)
    assert result.total_minutes == 18 + 9 + 6
    assert result.recommended_daily_minutes == 33 and result.estimate
    assert result.concepts[0].learning_minutes == 18


def test_study_time_spreads_remaining_work_and_does_not_create_plans(context):
    concept = TimeConcept(
        concept_version_id=context.concepts[0].version.id,
        name="Present",
        remaining_criteria=2,
        remaining_questions=2,
        review_cycles=1,
    )
    request = StudyTimeRequest(
        exam_date=date(2026, 11, 10), today=date(2026, 11, 3), concepts=[concept]
    )
    result = estimate_study_time(request)
    assert result.total_minutes == 22 and result.recommended_daily_minutes == 4
    assert "tasks" not in result.model_dump()
    assert estimate_study_time(request) == result


def test_negative_progress_rejected(context):
    with pytest.raises(ValidationError):
        TimeConcept(
            concept_version_id=context.concepts[0].version.id, name="Present", remaining_criteria=-1
        )


def test_weighted_v1_rounds_ratio_to_four_decimals(context):
    from uuid import uuid4

    from src.models import ConceptChecklistItem

    concept = context.concepts[0].model_copy(deep=True)
    concept.version.checklist_items.append(
        ConceptChecklistItem(
            id=uuid4(),
            ordinal=3,
            criterion="Explain negatives",
        )
    )
    judgments = EvaluationJudgments(
        items=[
            CriterionJudgment(
                checklist_item_id=item.id,
                status="explained" if i == 0 else "omission",
                description="Judgment",
                confidence="medium",
                evidence=[],
                learner_quote="",
            )
            for i, item in enumerate(concept.version.checklist_items)
        ],
        next_action="Review",
    )
    result = calculate_report(judgments, concept)
    assert result.coverage == 0.3333 and result.assessable_count == 3


def test_completed_concept_can_still_have_review_time(context):
    request = StudyTimeRequest(
        exam_date=date(2026, 11, 10),
        today=date(2026, 11, 3),
        concepts=[
            TimeConcept(
                concept_version_id=context.concepts[0].version.id,
                name="Present",
                remaining_criteria=0,
                review_criteria=3,
                review_cycles=2,
            )
        ],
    )
    result = estimate_study_time(request)
    assert result.concepts[0].learning_minutes == 0
    assert result.concepts[0].review_minutes == 12


def test_confirmed_context_workload_includes_teachback_and_no_daily_time_input(context):
    from src.study_time import estimate_context_workload

    result = estimate_context_workload(
        context, exam_date=date(2026, 11, 10), today=date(2026, 11, 3)
    )
    assert result.total_minutes == 12 + 6 + 5 + 4
    assert result.recommended_daily_minutes == 4
    assert result.concepts[0].teachback_minutes == 5
    assert result.estimate


def test_workload_difficulty_importance_and_progress(context):
    from src.study_time import estimate_context_workload

    version = context.concepts[0].version
    progress = TimeConcept(
        concept_version_id=version.id,
        name=version.name,
        remaining_criteria=1,
        remaining_questions=2,
        teachback_sessions=1,
        difficulty="hard",
        importance=1,
        weakness=1,
    )
    result = estimate_context_workload(
        context, exam_date=date(2026, 11, 10), today=date(2026, 11, 3), progress=[progress]
    )
    assert result.concepts[0].learning_minutes == 14
    assert result.concepts[0].question_minutes == 9
    assert result.concepts[0].teachback_minutes == 8
    progress.remaining_criteria = 50
    with pytest.raises(InputError):
        estimate_context_workload(
            context, exam_date=date(2026, 11, 10), today=date(2026, 11, 3), progress=[progress]
        )
