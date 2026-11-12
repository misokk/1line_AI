from datetime import date, datetime, timedelta, timezone
from math import ceil
from typing import Literal
from uuid import UUID

from pydantic import Field

from src.errors import InputError
from src.models import (
    ConceptTime,
    Model,
    StudyContext,
    StudyTimeEstimate,
    StudyTimeRequest,
    TimeConcept,
)


class WorkloadRequest(Model):
    total_pages: int = Field(ge=1, le=100000)
    context: StudyContext
    exam_date: date
    today: date = Field(default_factory=lambda: datetime.now(timezone(timedelta(hours=9))).date())
    progress: list[TimeConcept] | None = None


class WorkloadEstimate(Model):
    remaining_days: int
    total_pages: int
    concept_count: int
    recommended_daily_pages: int
    recommended_daily_concepts: int
    time_estimate: StudyTimeEstimate
    assumptions: list[str]


class PlanTask(Model):
    concept_id: UUID
    concept_version_id: UUID
    name: str
    activity: Literal["learn", "practice", "teachback", "review"]
    count: int = Field(ge=1)
    estimated_minutes: int = Field(ge=1)
    source_pages: list[int]
    priority: float


class PlanDay(Model):
    date: date
    tasks: list[PlanTask]
    estimated_minutes: int = Field(ge=0)


class StudyPlan(Model):
    today: date
    exam_date: date
    total_pages: int
    concept_count: int
    urgent: bool
    days: list[PlanDay]
    today_tasks: list[PlanTask]
    total_minutes: int
    assumptions: list[str]


def generate_plan(request: WorkloadRequest) -> StudyPlan:
    remaining = (request.exam_date - request.today).days
    if remaining < 0:
        raise InputError("Exam date has passed; choose today or a future date.")
    if remaining > 366:
        raise InputError("Study plans support up to 366 days before the exam.")
    estimate = estimate_workload(request)
    progress = _context_progress(request.context, request.progress)
    progress_by_id = {p.concept_version_id: p for p in progress}
    concepts = {c.version.id: c for c in request.context.concepts}
    count = max(1, remaining)
    days = [
        PlanDay(date=request.today + timedelta(days=i), tasks=[], estimated_minutes=0)
        for i in range(count)
    ]
    primary, reviews = [], []
    for timing in estimate.time_estimate.concepts:
        concept = concepts[timing.concept_version_id]
        state = progress_by_id[timing.concept_version_id]
        base = dict(
            concept_id=concept.concept_id,
            concept_version_id=concept.version.id,
            name=concept.version.name,
            priority=timing.priority,
            source_pages=sorted({e.page_number for e in concept.version.evidence}),
        )
        for activity, units, minutes in (
            ("learn", state.remaining_criteria, timing.learning_minutes),
            ("practice", state.remaining_questions, timing.question_minutes),
            ("teachback", state.teachback_sessions, timing.teachback_minutes),
            ("review", state.review_cycles, timing.review_minutes),
        ):
            if not units or not minutes:
                continue
            quotient, extra = divmod(minutes, units)
            for index in range(units):
                task = PlanTask(
                    **base,
                    activity=activity,
                    count=(
                        state.remaining_criteria
                        if state.review_criteria is None
                        else state.review_criteria
                    )
                    if activity == "review"
                    else 1,
                    estimated_minutes=quotient + (index < extra),
                )
                (reviews if activity == "review" else primary).append(task)

    def add(day_index, task):
        day = days[day_index]
        existing = next(
            (
                t
                for t in day.tasks
                if t.concept_version_id == task.concept_version_id and t.activity == task.activity
            ),
            None,
        )
        if existing:
            existing.count += task.count
            existing.estimated_minutes += task.estimated_minutes
        else:
            day.tasks.append(task.model_copy(deep=True))
        day.estimated_minutes += task.estimated_minutes

    # Reserve the last fifth for review when possible; no daily-time input is required.
    review_days = max(1, count // 5) if reviews and count >= 3 else 0
    primary_days = max(1, count - review_days) if primary else 0
    primary_total = sum(t.estimated_minutes for t in primary)
    elapsed, last_day = 0, {}
    for task in primary:
        index = min(primary_days - 1, elapsed * primary_days // primary_total)
        add(index, task)
        last_day[task.concept_version_id] = index
        elapsed += task.estimated_minutes
    for version_id in dict.fromkeys(t.concept_version_id for t in reviews):
        tasks = [t for t in reviews if t.concept_version_id == version_id]
        earliest = min(count - 1, max(primary_days, last_day.get(version_id, -1) + 1))
        available = count - earliest
        for index, task in enumerate(tasks):
            start = earliest + index * available // len(tasks)
            stop = min(count, max(start + 1, earliest + (index + 1) * available // len(tasks)))
            slot = min(range(start, stop), key=lambda i: (days[i].estimated_minutes, i))
            add(slot, task)
    total = sum(d.estimated_minutes for d in days)
    if total != estimate.time_estimate.total_minutes:
        raise InputError("Plan workload does not match the time estimate.")
    return StudyPlan(
        today=request.today,
        exam_date=request.exam_date,
        total_pages=request.total_pages,
        concept_count=estimate.concept_count,
        urgent=remaining <= 3,
        days=days,
        today_tasks=days[0].tasks,
        total_minutes=total,
        assumptions=[
            "확정된 개념과 입력 진도만 배치하며 DB 일정·사용자 기록은 저장하지 않습니다.",
            "취약도·미학습 여부·중요도 순으로 학습→문제 풀이→Teach-back을 배치합니다.",
            "가능하면 마지막 약 20% 날짜를 복습에 사용하며 시험일 전날까지 배치합니다.",
            "시험 당일은 오늘에 압축하고 임박 시 같은 날 여러 활동이 배정될 수 있습니다.",
            "페이지는 개념 근거 위치이며 전체 페이지의 모든 내용 학습을 보장하지 않습니다.",
            "시간은 기존 추정 기준을 사용하며 실제 학습 속도·완료를 보장하지 않습니다.",
        ],
    )


def estimate_workload(request: WorkloadRequest) -> WorkloadEstimate:
    from src.context import validate_context

    validate_context(request.context)
    days = max(0, (request.exam_date - request.today).days)
    count = len(request.context.concepts)
    return WorkloadEstimate(
        remaining_days=days,
        total_pages=request.total_pages,
        concept_count=count,
        recommended_daily_pages=ceil(request.total_pages / max(1, days)),
        recommended_daily_concepts=ceil(count / max(1, days)),
        time_estimate=estimate_context_workload(
            request.context,
            exam_date=request.exam_date,
            today=request.today,
            progress=request.progress,
        ),
        assumptions=[
            "단순 학습량 추정이며 날짜별 계획·일정·DB 레코드를 생성하지 않습니다.",
            "시험 전 남은 일수로 나누며 당일·시험 경과는 1일로 계산합니다.",
            "복습일 배정·코스 압축 비율·너무 먼 시험일 기준은 팀 협의 사항입니다.",
        ],
    )


def estimate_study_time(request: StudyTimeRequest) -> StudyTimeEstimate:
    """Independent estimate; does not create tasks, plans, DB records or learner diagnoses."""
    ids = [c.concept_version_id for c in request.concepts]
    if len(ids) != len(set(ids)):
        raise InputError("Duplicate concept versions in time estimate.")
    remaining = max(0, (request.exam_date - request.today).days)
    results = []
    for c in request.concepts:
        factor = {"easy": 0.75, "medium": 1.0, "hard": 1.5}[c.difficulty]
        learning = ceil(c.remaining_criteria * 6 * factor * (1 + c.weakness * 0.5))
        solving = ceil(c.remaining_questions * 3 * factor)
        teachback = ceil(c.teachback_sessions * 5 * factor)
        review_criteria = c.remaining_criteria if c.review_criteria is None else c.review_criteria
        review = ceil(review_criteria * c.review_cycles * 2 * factor * (1 + c.weakness * 0.5))
        results.append(
            ConceptTime(
                concept_version_id=c.concept_version_id,
                name=c.name,
                learning_minutes=learning,
                question_minutes=solving,
                review_minutes=review,
                teachback_minutes=teachback,
                total_minutes=learning + solving + review + teachback,
                priority=round(
                    c.weakness * 2 + (1 if c.remaining_criteria else 0) + c.importance - 0.5, 2
                ),
            )
        )
    results.sort(key=lambda item: -item.priority)
    total = sum(c.total_minutes for c in results)
    daily = ceil(total / max(1, remaining))
    guidance = (
        "Prioritize weak and unlearned concepts, then essential problems; reduce scope with "
        "the learner if the estimated daily workload is impractical."
    )
    if request.exam_date < request.today:
        guidance = "Exam date has passed; this is a remaining-work estimate for today. " + guidance
    elif daily > 180:
        guidance = (
            "Workload exceeds 180 minutes/day; prioritize rather than promise completion. "
            + guidance
        )
    return StudyTimeEstimate(
        remaining_days=remaining,
        total_minutes=total,
        recommended_daily_minutes=daily,
        concepts=results,
        assumptions=[
            "Heuristic estimate, not observed learning speed or a completion guarantee.",
            "6 minutes per remaining criterion, 3 per problem, 2 per criterion per review cycle.",
            "Criterion/review time multiplied by 1 + 0.5 * supplied weakness (0-1).",
            "Difficulty: easy 0.75, medium 1, hard 1.5; 5 minutes per teach-back session.",
            "Importance (0-1) affects priority, not minutes; estimates are not a diagnosis.",
            "Divide by days before exam; today/past uses one day. No time input or diagnosis.",
        ],
        priority_guidance=guidance,
    )


def estimate_context_workload(
    context: StudyContext,
    *,
    exam_date: date,
    today: date,
    progress: list[TimeConcept] | None = None,
) -> StudyTimeEstimate:
    """Use confirmed extracted concepts; backend may supply actual remaining work."""
    supplied = progress is not None
    progress = _context_progress(context, progress)
    result = estimate_study_time(
        StudyTimeRequest(exam_date=exam_date, today=today, concepts=progress)
    )
    if not supplied:
        result.assumptions.append(
            "Without progress: all confirmed criteria unlearned, 2 questions, 1 teach-back, "
            "1 review per concept; medium difficulty and neutral importance."
        )
    return result


def _context_progress(context, progress):
    by_id = {c.version.id: c for c in context.concepts}
    if progress is not None:
        if len(progress) != len(by_id) or set(by_id) != {p.concept_version_id for p in progress}:
            raise InputError("Progress must cover exactly the confirmed versions in context.")
        if any(
            p.remaining_criteria > len(by_id[p.concept_version_id].version.checklist_items)
            for p in progress
        ):
            raise InputError("Remaining criteria exceed the confirmed checklist.")
    else:
        progress = [
            TimeConcept(
                concept_version_id=c.version.id,
                name=c.version.name,
                remaining_criteria=len(c.version.checklist_items),
                remaining_questions=2,
                teachback_sessions=1,
                review_cycles=1,
            )
            for c in context.concepts
        ]
    return progress
