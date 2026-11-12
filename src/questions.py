import json
from collections import Counter
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, create_model

from api.base import validate_json
from api.router import ModelRouter
from src.context import make_prompt, payload, select_version, validate_question
from src.errors import InputError, InvalidResponseError
from src.grading import normalize_answer, verify_answer
from src.models import (
    GeneratedQuestions,
    Model,
    Option,
    PublicQuestion,
    PublicQuestionDraft,
    QuestionBatch,
    QuestionRequest,
    StoredQuestion,
    Text,
)

GENERATION_PROMPT = """
역할: 제공된 학습자료와 확정된 개념을 바탕으로 정확한 시험문제를 생성한다.

[문제 생성]
1. 요청된 count만큼 서로 다른 문제를 생성한다.
2. 선택한 단원과 확정된 개념 버전의 내용만 사용한다.
   외부 지식이 필요한 문제나 원문에 근거가 없는 문제는 만들지 않는다.
3. 난이도에 맞춰 출제한다.
   - easy: 개념 이해 및 직접 적용
   - medium: 주어진 상황에서 규칙 적용
   - hard: 여러 단계의 판단이나 개념 구별
4. question_type_plan의 순서대로 문제 유형을 배정하고,
   question_type_counts의 유형별 개수를 지킨다.
   count=2이고 두 유형이 요청되었다면 객관식과 단답형을 각각 1개 생성한다.
5. 지정된 concept_id와 대응하는 concept_version_id를 그대로 사용한다.
   새로운 ID를 생성하지 않는다.

[정답 및 해설]
1. 모든 문제는 원문으로 검증할 수 있는 명확한 정답을 가져야 한다.
2. 문제, 정답, 해설은 서로 모순되지 않아야 한다.
3. 객관식은 서로 다른 선택지 2~5개를 A~E로 표시한다.
   정답은 반드시 하나이며, 다른 선택지가 정답으로 해석될 여지가 없어야 한다.
4. 단답형은 질문에서 요구한 모든 요소를 정답에 포함한다.
   accepted_answers에는 의미와 필수 요소가 모두 동일한 대체 정답만 넣는다.
5. 해설은 정답이 맞는 이유를 원문 근거로 설명한다.
   문제에 없는 선택지를 언급하거나 근거 없는 내용을 추가하지 않는다.
6. 각 문제에 해당 학습자료의 SourceEvidence를 포함한다.
7. 반환 전에 정답의 정확성, 선택지의 모호성, 해설과 근거의 일치 여부를 확인한다.

[출력 형식]
- 최상위 JSON 객체의 유일한 키는 "items"이다.
- items에는 정확히 count개의 객체를 넣는다.
- 각 객체는 question, correct_answer, explanation,
  accepted_answers, arithmetic_expression을 포함한다.
- question과 evidence의 세부 필드는 제공된 JSON 스키마를 따른다.
- correct_answer와 explanation은 공개 question 객체와 분리한다.

객관식:
  correct_answer = {
    "type": "multiple_choice",
    "selected_option_ids": ["실제 정답 선택지 라벨"],
    "text": null
  }

단답형:
  correct_answer = {
    "type": "short_answer",
    "selected_option_ids": null,
    "text": "정답"
  }
  options = []

- 대체 정답이 없으면 accepted_answers=[]를 사용한다.
- 산술 계산 문제가 아니면 arithmetic_expression=null을 사용한다.
- 산술 계산 문제에는 숫자 계산식만 사용한다 (+, -, *, /, **).
  일반 방정식이나 기호 증명은 포함하지 않는다.
- 수식 검증 결과만으로 문제의 의미적 정확성이 보장되지는 않는다.
- 출력 언어는 요청된 explanation_language를 따른다.
- JSON 스키마에 정의되지 않은 필드는 추가하지 않는다.
"""


class GenerationAnswer(Model):
    """Single wire shape avoids object-union ambiguity in provider decoders."""

    type: Literal["multiple_choice", "short_answer"]
    selected_option_ids: list[str] | None = None
    text: str | None = None


@lru_cache(maxsize=120)
def question_batch_schema(
    count: int, kinds: tuple[str, ...], difficulty: str
) -> type[QuestionBatch]:
    """Constrain the wire response without changing the public QuestionBatch contract."""
    question = create_model(
        "GenerationQuestion",
        __base__=PublicQuestionDraft,
        question_type=(Literal[kinds], ...),
        difficulty=(Literal[difficulty], ...),
    )
    item = create_model(
        "GenerationItem",
        __base__=StoredQuestion,
        question=(question, ...),
        correct_answer=(GenerationAnswer, ...),
    )
    return create_model(
        f"QuestionBatch{count}",
        __base__=QuestionBatch,
        items=(list[item], Field(min_length=count, max_length=count)),
    )


@lru_cache(maxsize=120)
def indexed_question_schema(count, kinds, difficulty, concept_count, evidence_count):
    question = create_model(
        "IndexedGenerationQuestion",
        __base__=Model,
        question_type=(Literal[kinds], ...),
        difficulty=(Literal[difficulty], ...),
        prompt=(Text, ...),
        options=(list[Option], Field(max_length=5)),
        concept_indices=(
            list[Annotated[int, Field(strict=True, ge=0, le=concept_count - 1)]],
            Field(min_length=1, max_length=concept_count),
        ),
        evidence_indices=(
            list[Annotated[int, Field(strict=True, ge=0, le=evidence_count - 1)]],
            Field(min_length=1, max_length=evidence_count),
        ),
    )
    item = create_model(
        "IndexedGenerationItem",
        __base__=StoredQuestion,
        question=(question, ...),
        correct_answer=(GenerationAnswer, ...),
    )
    return create_model(
        f"IndexedQuestionBatch{count}",
        __base__=QuestionBatch,
        items=(list[item], Field(min_length=count, max_length=count)),
    )


def question_catalog(concepts):
    catalog, owners = [], []
    for index, concept in enumerate(concepts):
        for evidence in concept.version.evidence:
            if evidence in catalog:
                owners[catalog.index(evidence)].add(index)
            else:
                catalog.append(evidence)
                owners.append({index})
    if not catalog:
        raise InputError("Question generation requires confirmed source evidence.")
    return catalog, owners


def expand_question_references(part, concepts, catalog, owners, schema):
    data = validate_json(part.model_dump_json(), schema).model_dump(mode="json")
    for item in data["items"]:
        question = item["question"]
        indices = question.pop("concept_indices")
        refs = question.pop("evidence_indices")
        if len(indices) != len(set(indices)) or len(refs) != len(set(refs)):
            raise InvalidResponseError("Generated duplicate concept/evidence indices.")
        if any(not owners[r].intersection(indices) for r in refs):
            raise InvalidResponseError("Evidence index belongs to an unselected concept version.")
        question["concept_ids"] = list(dict.fromkeys(str(concepts[i].concept_id) for i in indices))
        question["concept_version_ids"] = [str(concepts[i].version.id) for i in indices]
        question["evidence"] = [catalog[r].model_dump(mode="json") for r in refs]
        answer = item["correct_answer"]
        if answer["type"] == "multiple_choice":
            if answer.get("text") is not None:
                raise InvalidResponseError(
                    "Multiple-choice answer includes conflicting answer text."
                )
            answer.pop("text", None)
        else:
            if answer.get("selected_option_ids") not in (None, []):
                raise InvalidResponseError("Short-answer result includes selected option IDs.")
            answer.pop("selected_option_ids", None)
    return validate_json(json.dumps(data, ensure_ascii=False), QuestionBatch)


def _stored_question(item: StoredQuestion) -> StoredQuestion:
    data = item.model_dump(mode="json")
    answer = data["correct_answer"]
    if answer["type"] == "multiple_choice":
        if answer.get("text") is not None:
            raise InvalidResponseError("Multiple-choice answer includes conflicting answer text.")
        answer.pop("text", None)
    else:
        if answer.get("selected_option_ids") not in (None, []):
            raise InvalidResponseError("Short-answer result includes selected option IDs.")
        answer.pop("selected_option_ids", None)
    return validate_json(json.dumps(data, ensure_ascii=False), StoredQuestion)


def generate(request: QuestionRequest, router: ModelRouter) -> GeneratedQuestions:
    if len(set(request.concept_version_ids)) != len(request.concept_version_ids):
        raise InputError("Duplicate requested version IDs.")
    concepts = [select_version(request.context, v) for v in request.concept_version_ids]
    if len({c.material_id for c in concepts}) != 1 or any(
        c.unit_id != request.unit_id for c in concepts
    ):
        raise InputError("Question set must use versions from the requested unit and material.")
    if len(set(request.question_types)) != len(request.question_types):
        raise InputError("Question types must be unique.")
    type_plan = [
        request.question_types[i % len(request.question_types)] for i in range(request.count)
    ]
    budget = router.output_budget("question_generation", request.model_id)
    catalog, owners = question_catalog(concepts) if budget else (None, None)
    per_call = max(1, budget // 650) if budget else request.count
    target = router.settings.target_for("question_generation", request.model_id)
    if target.provider in ("gemini", "openai"):
        per_call = min(per_call, router.settings.question_batch_size)
    levels = (
        request.difficulty_plan
        if request.difficulty == "mixed"
        else [request.difficulty] * request.count
    )
    grouped = {}
    for ordinal, (kind, level) in enumerate(zip(type_plan, levels, strict=True)):
        grouped.setdefault(level, []).append((ordinal, kind))
    plans = [
        (level, batch[i : i + per_call])
        for level, batch in grouped.items()
        for i in range(0, len(batch), per_call)
    ]
    router.check_call_count("question_generation", request.model_id, len(plans))
    collected = []
    seen_prompts = set()
    ordered = {}
    for level, entries in plans:
        plan = [kind for _, kind in entries]
        data = {
            **payload(request.context, concepts),
            **request.model_dump(mode="json", exclude={"context"}),
            "count": len(plan),
            "difficulty": level,
            "difficulty_plan": None,
            "question_types": list(dict.fromkeys(plan)),
            "question_type_plan": [
                {"ordinal": i + 1, "question_type": kind} for i, kind in enumerate(plan)
            ],
            "question_type_counts": dict(Counter(plan)),
            "previous_question_prompts": [q.question.prompt for q in collected],
        }
        instruction = GENERATION_PROMPT
        schema = question_batch_schema(len(plan), tuple(dict.fromkeys(plan)), level)
        if budget:
            schema = indexed_question_schema(
                len(plan),
                tuple(dict.fromkeys(plan)),
                level,
                len(concepts),
                len(catalog),
            )
            data["concept_catalog"] = [
                {"index": i, "concept": c.model_dump(mode="json")} for i, c in enumerate(concepts)
            ]
            data.pop("confirmed_concepts", None)
            data["citation_catalog"] = [
                {"index": i, "concept_indices": sorted(owners[i]), "text": e.excerpt}
                for i, e in enumerate(catalog)
            ]
            instruction = instruction.replace(
                "5. 지정된 concept_id와 대응하는 concept_version_id를 그대로 사용한다.\n"
                "   새로운 ID를 생성하지 않는다.",
                "5. question.concept_indices에는 concept_catalog의 index 정수만 선택한다. "
                "concept_id, concept_version_id 등 UUID 필드는 생성하거나 반환하지 않는다.",
            )
            instruction = instruction.replace(
                "6. 각 문제에 해당 학습자료의 SourceEvidence를 포함한다.",
                "6. question.evidence_indices에는 해당 개념을 실제로 뒷받침하는 citation_catalog의 "
                "index 정수만 선택한다. 각 인용의 concept_indices 중 하나 이상을 선택해야 한다. "
                "인용문·자료명·페이지·UUID는 반환하지 않는다. 번호는 중복 선택하지 않는다.",
            )
            instruction += (
                "\nKeep each question concise: a short prompt, short distinct options, "
                "one-sentence explanation, minimal exact evidence quotations and only "
                "necessary concept references. Avoid all previous_question_prompts. "
                f"The complete JSON must fit {budget} output tokens; never omit required fields."
            )
        part = router.generate_json(
            "question_generation",
            model_id=request.model_id,
            prompt=make_prompt(instruction, data),
            schema=schema,
        )
        if catalog and part.items and not hasattr(part.items[0].question, "concept_ids"):
            part = expand_question_references(part, concepts, catalog, owners, schema)
        subrequest = request.model_copy(
            update={
                "count": len(plan),
                "question_types": list(dict.fromkeys(plan)),
                "difficulty": level,
                "difficulty_plan": None,
            }
        )
        checked = _validate_batch(part, subrequest, plan)
        by_type = {}
        for item in checked.items:
            by_type.setdefault(item.question.question_type, []).append(item)
        for ordinal, kind in entries:
            item = by_type[kind].pop(0)
            ordered[ordinal] = item
            key = normalize_answer(item.question.prompt)
            if key in seen_prompts:
                raise InvalidResponseError("Generated duplicate questions across split calls.")
            seen_prompts.add(key)
            collected.append(item)
    return _validate_batch(
        QuestionBatch(items=[ordered[i] for i in range(request.count)]), request, type_plan
    )


def _validate_batch(batch, request, type_plan):
    if len(batch.items) != request.count:
        raise InvalidResponseError(
            f"Generated question count differs: requested {request.count}, "
            f"received {len(batch.items)}."
        )
    batch = QuestionBatch(items=[_stored_question(item) for item in batch.items])
    seen = set()
    kinds = set()
    levels = (
        request.difficulty_plan
        if request.difficulty == "mixed"
        else [request.difficulty] * request.count
    )
    for ordinal, stored in enumerate(batch.items):
        question = stored.question
        try:
            validate_question(question, request.context)
        except InputError as error:
            raise InvalidResponseError(
                "Generated question has invalid concept/evidence relations: " + str(error)
            ) from None
        if not set(question.concept_version_ids) <= set(request.concept_version_ids):
            raise InvalidResponseError("Generated question uses an unselected version.")
        if (
            question.question_type not in request.question_types
            or question.difficulty != levels[ordinal]
        ):
            raise InvalidResponseError("Generated type or difficulty differs from the request.")
        normalized = normalize_answer(question.prompt)
        if normalized in seen:
            raise InvalidResponseError("Generated duplicate questions.")
        seen.add(normalized)
        kinds.add(question.question_type)
        if verify_answer(stored).status == "inconsistent":
            raise InvalidResponseError("Generated arithmetic answer is inconsistent.")
    if request.count >= len(request.question_types) and kinds != set(request.question_types):
        missing = sorted(set(request.question_types) - kinds)
        counts = dict(Counter(q.question.question_type for q in batch.items))
        raise InvalidResponseError(
            f"Generated questions omit requested types {missing}; received counts {counts}. "
            "No automatic retry was made."
        )
    actual_counts = Counter(q.question.question_type for q in batch.items)
    if actual_counts != Counter(type_plan):
        raise InvalidResponseError("Generated question type counts differ from the request plan.")
    return GeneratedQuestions(
        items=batch.items, verification=[verify_answer(q) for q in batch.items]
    )


def public_question(stored: StoredQuestion, question_id) -> PublicQuestion:
    """The caller supplies the DB ID; no answer fields can appear in this projection."""
    return PublicQuestion(id=question_id, **stored.question.model_dump())
