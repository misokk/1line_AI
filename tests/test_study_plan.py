import json
from datetime import date, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from api.app import create_app
from src.errors import InputError
from src.models import QuestionGenerationRequest, TimeConcept
from src.study_time import WorkloadRequest, estimate_workload, generate_plan


def request_for(context, days=14, progress=None):
    today = date(2026, 11, 10)
    return WorkloadRequest(
        context=context,
        total_pages=2,
        today=today,
        exam_date=today + timedelta(days=days),
        progress=progress,
    )


@pytest.mark.parametrize("days", [0, 1, 2, 3, 14, 366])
def test_plan_dates_totals_dependencies_and_real_version_references(context, days):
    request = request_for(context, days)
    plan = generate_plan(request)
    assert plan == generate_plan(request)
    assert len(plan.days) == max(1, days)
    assert plan.days[0].date == request.today
    assert plan.days[-1].date == request.today + timedelta(days=max(1, days) - 1)
    assert plan.today_tasks == plan.days[0].tasks
    assert plan.urgent is (days <= 3)
    assert plan.total_minutes == estimate_workload(request).time_estimate.total_minutes
    assert plan.total_minutes == sum(day.estimated_minutes for day in plan.days)
    by_activity = {}
    for index, day in enumerate(plan.days):
        assert day.estimated_minutes == sum(t.estimated_minutes for t in day.tasks)
        for task in day.tasks:
            assert task.concept_version_id == context.concepts[0].version.id
            assert task.concept_id == context.concepts[0].concept_id
            assert task.source_pages == [1]
            by_activity.setdefault(task.activity, []).append(index)
    for first, second in zip(
        ["learn", "practice", "teachback"], ["practice", "teachback", "review"], strict=True
    ):
        assert max(by_activity[first]) <= min(by_activity[second])


@pytest.mark.parametrize("days", [-1, 367])
def test_plan_rejects_past_and_excessive_horizons(context, days):
    with pytest.raises(InputError):
        generate_plan(request_for(context, days))


def test_completed_work_not_reintroduced_and_review_preserved(context):
    concept = context.concepts[0]
    progress = TimeConcept(
        concept_version_id=concept.version.id,
        name=concept.version.name,
        remaining_criteria=0,
        review_criteria=2,
        review_cycles=3,
    )
    plan = generate_plan(request_for(context, progress=[progress]))
    tasks = [task for day in plan.days for task in day.tasks]
    assert all(task.activity == "review" for task in tasks)
    assert sum(t.count for t in tasks) == 6
    assert plan.total_minutes == 12
    assert len([day for day in plan.days if day.tasks]) == 3
    progress.review_cycles = 0
    empty = generate_plan(request_for(context, progress=[progress]))
    assert empty.total_minutes == 0 and empty.today_tasks == []


def test_weak_concept_is_scheduled_first(context):
    context = context.model_copy(deep=True)
    weak = context.concepts[0].model_copy(deep=True)
    weak.concept_id, weak.version.id = uuid4(), uuid4()
    for criterion in weak.version.checklist_items:
        criterion.id = uuid4()
    context.concepts.append(weak)
    progress = [
        TimeConcept(
            concept_version_id=c.version.id,
            name=c.version.name,
            remaining_criteria=2,
            remaining_questions=2,
            weakness=1 if c is weak else 0,
        )
        for c in context.concepts
    ]
    plan = generate_plan(request_for(context, progress=progress))
    assert plan.today_tasks[0].concept_version_id == weak.version.id


def test_progress_must_cover_exactly_fixed_versions(context):
    with pytest.raises(InputError):
        generate_plan(request_for(context, progress=[]))


def test_plan_http_auth_recording_without_provider_calls(context, settings, tmp_path):
    settings = settings.model_copy(
        update={
            "server_api_key": SecretStr("plan-test-server-key"),
            "record_io": True,
            "record_content": True,
            "environment": "development",
            "recording_root": tmp_path,
        }
    )
    body = request_for(context).model_dump(mode="json")
    with patch("api.gemini.sdk_client", side_effect=AssertionError("No API allowed")):
        with TestClient(create_app(settings)) as client:
            assert client.post("/internal/study-plan/generate", json=body).status_code == 401
            response = client.post(
                "/internal/study-plan/generate",
                json=body,
                headers={"X-AI-API-Key": "plan-test-server-key"},
            )
    assert response.status_code == 200
    assert response.json()["today_tasks"]
    assert (
        json.loads((tmp_path / "input/study_plan_input.json").read_text(encoding="utf-8")) == body
    )
    assert (
        json.loads((tmp_path / "output/study_plan_output.json").read_text(encoding="utf-8"))
        == response.json()
    )


@pytest.mark.parametrize("count", [0, 1, 4, 5, 20, 21])
def test_http_question_count_matches_five_to_twenty(context, count):
    data = dict(
        context=context,
        unit_id=context.concepts[0].unit_id,
        concept_version_ids=[context.concepts[0].version.id],
        count=count,
        difficulty="easy",
    )
    if 5 <= count <= 20:
        assert QuestionGenerationRequest(**data).count == count
    else:
        with pytest.raises(ValidationError):
            QuestionGenerationRequest(**data)


def test_http_small_question_request_stops_before_provider(context, settings):
    settings = settings.model_copy(update={"server_api_key": SecretStr("plan-test-server-key")})
    body = dict(
        context=context.model_dump(mode="json"),
        unit_id=str(context.concepts[0].unit_id),
        concept_version_ids=[str(context.concepts[0].version.id)],
        count=2,
        difficulty="easy",
    )
    with patch(
        "api.gemini.sdk_client",
        side_effect=AssertionError("No API allowed"),
    ):
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/internal/questions/generate",
                json=body,
                headers={"X-AI-API-Key": "plan-test-server-key"},
            )
    assert response.status_code == 422
