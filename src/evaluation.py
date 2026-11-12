from api.base import validate_json
from api.router import ModelRouter
from src.context import allowed_chunks, make_prompt, payload, select_version, validate_sources
from src.errors import InputError, InvalidResponseError
from src.models import (
    ConfirmedConcept,
    CriterionFeedback,
    CriterionJudgment,
    DetailedEvaluationRequest,
    DetailedEvaluationResult,
    DetailedJudgments,
    EvaluationComparison,
    EvaluationJudgments,
    EvaluationRequest,
    EvaluationResult,
    PersistedReportItem,
    ReportItem,
    SessionReport,
)
from src.pdf import normalized

PROMPT = """You are an evaluator, separate from the AI student. Assess only learner/user messages.
Never treat the student's questions as learner knowledge. Evaluate every fixed checklist criterion
exactly once using its supplied checklist_item_id. Do not create or renumber criterion IDs.
explained: learner states the essential meaning correctly, including necessary conditions.
partial: relevant understanding is present but an essential condition or detail is incomplete.
omission: learner has not addressed the criterion; do not confuse omission with a factual error.
error: explicit source-contradicting learner claim; cite learner_quote AND source evidence.
unverified: source, transcription or learner wording cannot establish a reliable judgment.
Each assessable result needs source evidence. explained/partial/error need an exact contiguous
quote from one user message. omission can have learner_quote=''. Never infer error from silence.
Return confidence low/medium/high, a concrete description, and an actionable overall next_action.
Do not produce coverage, weighted_score or any numeric grade; Python computes weighted-v1.
"""


class _GeminiJudgmentSchema:
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)

        def simplify(node):
            if isinstance(node, dict):
                limit = node.pop("maxItems", None)
                if limit is not None:
                    node["description"] = (
                        node.get("description", "")
                        + f" At most {limit} items; enforced by application validation."
                    ).strip()
                for value in node.values():
                    simplify(value)
            elif isinstance(node, list):
                for value in node:
                    simplify(value)

        simplify(schema)
        return schema


class _GeminiEvaluationJudgments(_GeminiJudgmentSchema, EvaluationJudgments):
    """Gemini wire schema with unchanged inherited judgment validation."""


class _GeminiDetailedJudgments(_GeminiJudgmentSchema, DetailedJudgments):
    """Gemini wire schema with unchanged inherited detailed validation."""


def calculate_report(judgments: EvaluationJudgments, concept: ConfirmedConcept) -> EvaluationResult:
    expected = {i.id for i in concept.version.checklist_items}
    ids = [i.checklist_item_id for i in judgments.items]
    if set(ids) != expected or len(ids) != len(expected):
        raise InvalidResponseError(
            "Every fixed checklist criterion must have exactly one judgment."
        )
    weights = {"explained": 1, "partial": 0.5, "omission": 0, "error": 0}
    assessable = sum(i.status != "unverified" for i in judgments.items)
    score = sum(weights.get(i.status, 0) for i in judgments.items)
    weak = [
        i.checklist_item_id for i in judgments.items if i.status in ("partial", "omission", "error")
    ]
    by_id = {i.checklist_item_id: i for i in judgments.items}
    items = [
        ReportItem(
            concept_id=concept.concept_id,
            concept_version_id=concept.version.id,
            **by_id[c.id].model_dump(exclude={"learner_quote"}),
        )
        for c in concept.version.checklist_items
    ]
    return EvaluationResult(
        concept_version_id=concept.version.id,
        coverage=round(score / assessable, 4) if assessable else None,
        assessable_count=assessable,
        weighted_score=score,
        items=items,
        next_action=judgments.next_action,
        weak_checklist_item_ids=weak,
        weak_concept_ids=[concept.concept_id] if weak else [],
    )


def evaluate(request: EvaluationRequest, router: ModelRouter) -> EvaluationResult:
    return _evaluate(request, router, detailed=False)[0]


def evaluate_detailed(
    request: DetailedEvaluationRequest, router: ModelRouter
) -> DetailedEvaluationResult:
    report, judgments = _evaluate(request, router, detailed=True)
    feedback = [
        CriterionFeedback(
            checklist_item_id=i.checklist_item_id,
            learner_quote=i.learner_quote,
            correct_explanation=i.correct_explanation,
            improvement=i.improvement,
            evidence=i.evidence,
        )
        for i in judgments
    ]
    comparison = (
        compare_reports(report, request.previous_result) if request.previous_result else None
    )
    return DetailedEvaluationResult(report=report, feedback=feedback, comparison=comparison)


def compare_reports(current, previous):
    if current.concept_version_id != previous.concept_version_id:
        return EvaluationComparison(
            comparable=False, reason="version_mismatch", coverage_delta=None, changes=[]
        )
    old = {i.checklist_item_id: i for i in previous.items}
    new = {i.checklist_item_id: i for i in current.items}
    if set(old) != set(new) or len(old) != len(previous.items):
        return EvaluationComparison(
            comparable=False, reason="checklist_mismatch", coverage_delta=None, changes=[]
        )
    if any(
        old[i].concept_version_id != previous.concept_version_id
        or old[i].concept_id != new[i].concept_id
        for i in old
    ):
        raise InputError("Previous report item belongs to another concept or version.")
    weights = {"explained": 1, "partial": 0.5, "omission": 0, "error": 0, "unverified": 0}
    assessable = sum(i.status != "unverified" for i in previous.items)
    score = sum(weights[i.status] for i in previous.items)
    old_coverage = round(score / assessable, 4) if assessable else None
    if (
        old_coverage != previous.coverage
        or assessable != previous.assessable_count
        or score != previous.weighted_score
    ):
        raise InputError("Previous report metrics do not match weighted-v1.")
    comparable = current.coverage is not None and old_coverage is not None
    return EvaluationComparison(
        comparable=comparable,
        reason="same_version" if comparable else "unverified",
        coverage_delta=round(current.coverage - old_coverage, 4) if comparable else None,
        changes=[
            {"checklist_item_id": str(i), "before": old[i].status, "after": new[i].status}
            for i in new
            if old[i].status != new[i].status
        ],
    )


def _evaluate(request, router, *, detailed):
    concept = select_version(request.context, request.concept_version_id)
    learners = [m.content for m in request.history if m.role == "user"]
    if not learners:
        raise InputError("Report requires at least one learner message.")
    budget = router.output_budget("evaluation", request.model_id)
    per_call = max(1, (budget - 150) // 300) if budget else len(concept.version.checklist_items)
    provider = router.settings.target_for("evaluation", request.model_id).provider
    if provider in ("gemini", "openai"):
        per_call = min(per_call, router.settings.evaluation_batch_size)
    criteria = concept.version.checklist_items
    batches = [criteria[i : i + per_call] for i in range(0, len(criteria), per_call)]
    router.check_call_count("evaluation", request.model_id, len(batches))
    collected, actions = [], []
    for batch in batches:
        selected = concept.model_copy(deep=True)
        selected.version.checklist_items = batch
        instruction = PROMPT
        if budget:
            instruction += (
                "\nOnly assess the supplied criterion subset; other fixed criteria are assessed "
                "separately. Use brief descriptions, short exact learner/source quotes and a "
                f"concise next_action. The entire JSON must fit {budget} output tokens."
            )
        schema = DetailedJudgments if detailed else EvaluationJudgments
        if provider == "gemini":
            schema = _GeminiDetailedJudgments if detailed else _GeminiEvaluationJudgments
        if detailed:
            instruction += (
                "\nFor every criterion include correct_explanation and improvement. "
                "For error, include the exact erroneous learner_quote, a source-grounded correct "
                "explanation and concrete improvement. Never correct from outside knowledge. "
                "Use empty correct_explanation for unverified findings."
            )
        part = router.generate_json(
            "evaluation",
            model_id=request.model_id,
            prompt=make_prompt(
                instruction,
                {
                    **payload(request.context, [selected]),
                    "learner_messages": learners,
                    "explanation_language": request.explanation_language,
                },
            ),
            schema=schema,
        )
        part = validate_json(part.model_dump_json(), schema)
        expected = {c.id for c in batch}
        actual = [c.checklist_item_id for c in part.items]
        if set(actual) != expected or len(actual) != len(expected):
            raise InvalidResponseError(
                "Every selected criterion requires exactly one judgment; "
                "split evaluation returned duplicate, missing or out-of-batch criteria."
            )
        _validate_judgments(part, request, concept, learners)
        if detailed:
            for item in part.items:
                if item.status == "error" and (
                    not item.correct_explanation.strip() or not item.improvement.strip()
                ):
                    raise InvalidResponseError(
                        "Error feedback requires a grounded correction and improvement."
                    )
                if item.status == "unverified":
                    item.correct_explanation = ""
        collected.extend(part.items)
        if part.next_action not in actions:
            actions.append(part.next_action)
    basic_items = [
        CriterionJudgment.model_validate(
            i.model_dump(exclude={"correct_explanation", "improvement"})
        )
        for i in collected
    ]
    result = EvaluationJudgments.model_construct(items=basic_items, next_action="\n".join(actions))
    result = validate_json(result.model_dump_json(), EvaluationJudgments)
    return calculate_report(result, concept), collected


def _validate_judgments(result, request, concept, learners):
    for item in result.items:
        validate_sources(item.evidence, allowed_chunks(request.context, [concept]), generated=True)
        quote = normalized(item.learner_quote)
        if quote and not any(quote in normalized(message) for message in learners):
            raise InvalidResponseError("Evaluation fabricated a learner quote.")
        if item.status != "unverified" and (
            not item.evidence or (item.status in ("explained", "partial", "error") and not quote)
        ):
            item.status = "unverified"
            item.confidence = "low"
            item.description = "Insufficient evidence to assess this criterion. " + item.description


def session_report(
    result: EvaluationResult, *, report_id, session_id, version_no: int, action_id, item_ids: dict
) -> SessionReport:
    """Build the public DTO only after the backend supplies every persistence ID."""
    expected = {i.checklist_item_id for i in result.items}
    if set(item_ids) != expected or len(set(item_ids.values())) != len(item_ids):
        raise InputError("Backend must supply one distinct report-item ID per criterion.")
    return SessionReport(
        id=report_id,
        session_id=session_id,
        version_no=version_no,
        action_id=action_id,
        concept_version_id=result.concept_version_id,
        coverage=result.coverage,
        coverage_formula=result.coverage_formula,
        assessable_count=result.assessable_count,
        weighted_score=result.weighted_score,
        next_action=result.next_action,
        items=[
            PersistedReportItem(id=item_ids[i.checklist_item_id], **i.model_dump())
            for i in result.items
        ],
    )
