from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

Text = Annotated[str, Field(min_length=1, max_length=12000)]
Name = Annotated[str, Field(min_length=1, max_length=200)]
Language = Literal["ko", "en"]
Difficulty = Literal["easy", "medium", "hard"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def version_four_ids(self):
        for value in self.__dict__.values():
            values = value if isinstance(value, list) else [value]
            if any(isinstance(v, UUID) and v.version != 4 for v in values):
                raise ValueError("Backend identifiers must be UUID v4.")
        return self


class Page(Model):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    number: int = Field(ge=1, le=500)
    text: str = Field(max_length=100000)


class Document(Model):
    pages: list[Page] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def valid_pages(self):
        numbers = [p.number for p in self.pages]
        if numbers != sorted(set(numbers)):
            raise ValueError("Page numbers must be unique and increasing.")
        if sum(len(p.text) for p in self.pages) > 400000:
            raise ValueError("Document exceeds 400,000 characters.")
        if not any(p.text.strip() for p in self.pages):
            raise ValueError("No extractable text; image OCR must be requested explicitly.")
        return self


class Evidence(Model):
    page: int = Field(ge=1)
    quote: str = Field(min_length=3, max_length=2000)


class ChunkDraft(Model):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    ordinal: int = Field(ge=1)
    page_number: int = Field(ge=1)
    text: Text


class SourceChunk(Model):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    id: UUID
    material_id: UUID
    material_name: Name
    page_number: int = Field(ge=1)
    text: Text


class SourceEvidence(Model):
    chunk_id: UUID
    material_id: UUID
    material_name: Name
    page_number: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=2000)


class ConceptChecklistItem(Model):
    id: UUID
    ordinal: int = Field(ge=1)
    criterion: str = Field(min_length=1, max_length=2000)


class ConceptVersion(Model):
    id: UUID
    version_no: int = Field(ge=1)
    name: Name
    summary: str | None = Field(max_length=12000)
    is_confirmed: bool
    is_current: bool
    confirmed_at: datetime | None
    checklist_items: list[ConceptChecklistItem] = Field(max_length=50)
    evidence: list[SourceEvidence] = Field(max_length=200)

    @model_validator(mode="after")
    def valid_snapshot(self):
        ids = [i.id for i in self.checklist_items]
        ordinals = [i.ordinal for i in self.checklist_items]
        if len(ids) != len(set(ids)) or ordinals != list(range(1, len(ids) + 1)):
            raise ValueError("Checklist IDs must be unique and ordinals contiguous from 1.")
        if self.is_confirmed and (
            not self.summary
            or not self.checklist_items
            or not self.evidence
            or self.confirmed_at is None
            or self.confirmed_at.utcoffset() is None
        ):
            raise ValueError(
                "Confirmed version requires summary, checklist, evidence and timestamp."
            )
        return self


class ConfirmedConcept(Model):
    concept_id: UUID
    unit_id: UUID
    material_id: UUID
    version: ConceptVersion


class StudyContext(Model):
    concepts: list[ConfirmedConcept] = Field(min_length=1, max_length=100)
    chunks: list[SourceChunk] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def valid_context(self):
        if any(not c.version.is_confirmed for c in self.concepts):
            raise ValueError("CONCEPT_NOT_CONFIRMED")
        for values in (
            [c.concept_id for c in self.concepts],
            [c.version.id for c in self.concepts],
            [c.id for c in self.chunks],
        ):
            if len(set(values)) != len(values):
                raise ValueError("Duplicate context IDs.")
        if sum(len(c.text) for c in self.chunks) > 400000:
            raise ValueError("Context exceeds 400,000 characters.")
        return self


class ModelRequest(Model):
    model_id: str | None = Field(default=None, min_length=1, max_length=100)


class AnalyzeRequest(ModelRequest):
    chunks: list[SourceChunk] = Field(min_length=1, max_length=200)
    explanation_language: Language = "ko"
    chunk_chars: int = Field(default=10000, ge=500, le=12000)
    max_calls: int = Field(default=10, ge=1, le=40)

    @model_validator(mode="after")
    def unique_chunks(self):
        if len({c.id for c in self.chunks}) != len(self.chunks):
            raise ValueError("Duplicate chunk IDs.")
        if len({c.material_id for c in self.chunks}) != 1:
            raise ValueError("One material per analysis request.")
        if sum(len(c.text) for c in self.chunks) > 400000:
            raise ValueError("Material exceeds 400,000 characters.")
        return self


class DraftConcept(Model):
    name: Name
    summary: Text
    checklist_items: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(
        min_length=1, max_length=50
    )
    keywords: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1, max_length=5
    )
    evidence: list[SourceEvidence] = Field(min_length=1, max_length=200)
    estimated_difficulty: Difficulty | None = None
    estimated_importance: float | None = Field(default=None, ge=0, le=1)


class DraftUnit(Model):
    title: Name
    concepts: list[DraftConcept] = Field(min_length=1, max_length=100)


class AnalysisPart(Model):
    units: list[DraftUnit] = Field(max_length=200)


class DraftAnalysis(Model):
    state: Literal["draft"] = "draft"
    units: list[DraftUnit] = Field(min_length=1, max_length=200)


class Option(Model):
    id: str = Field(min_length=1, max_length=40)
    text: Text


class MultipleChoiceAnswer(Model):
    type: Literal["multiple_choice"]
    selected_option_ids: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1, max_length=1
    )


class ShortAnswer(Model):
    type: Literal["short_answer"]
    text: Text


QuestionAnswer = Annotated[MultipleChoiceAnswer | ShortAnswer, Field(discriminator="type")]


class PublicQuestionDraft(Model):
    question_type: Literal["multiple_choice", "short_answer"]
    concept_ids: list[UUID] = Field(min_length=1, max_length=100)
    concept_version_ids: list[UUID] = Field(min_length=1, max_length=100)
    prompt: Text
    options: list[Option] = Field(max_length=5)
    difficulty: Difficulty
    evidence: list[SourceEvidence] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def valid_question(self):
        if len(self.concept_ids) != len(set(self.concept_ids)) or len(
            self.concept_version_ids
        ) != len(set(self.concept_version_ids)):
            raise PydanticCustomError(
                "question_duplicate_concept_ids", "Question concept and version IDs must be unique."
            )
        ids = [o.id for o in self.options]
        texts = [o.text.casefold() for o in self.options]
        if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
            raise PydanticCustomError(
                "question_duplicate_options", "Option IDs and texts must be unique."
            )
        if self.question_type == "multiple_choice" and not 2 <= len(ids) <= 5:
            raise PydanticCustomError(
                "question_choice_option_count", "Multiple choice requires 2-5 options."
            )
        if self.question_type == "short_answer" and self.options:
            raise PydanticCustomError(
                "question_short_answer_options_not_empty", "Short answer requires empty options."
            )
        return self


class PublicQuestion(PublicQuestionDraft):
    id: UUID


class AnswerVerification(Model):
    status: Literal["consistent", "inconsistent", "not_verified"]
    scope: str


class StoredQuestion(Model):
    question: PublicQuestionDraft
    correct_answer: QuestionAnswer
    explanation: Text
    accepted_answers: list[Text] = Field(default_factory=list, max_length=20)
    arithmetic_expression: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def valid_answer(self):
        if self.correct_answer.type != self.question.question_type:
            raise PydanticCustomError(
                "question_answer_type_mismatch", "Answer type differs from question type."
            )
        if isinstance(self.correct_answer, MultipleChoiceAnswer):
            if not set(self.correct_answer.selected_option_ids) <= {
                o.id for o in self.question.options
            }:
                raise PydanticCustomError(
                    "question_answer_unknown_option", "Answer refers to an unknown option."
                )
        return self


class QuestionBatch(Model):
    items: list[StoredQuestion] = Field(min_length=1, max_length=20)


class QuestionRequest(ModelRequest):
    context: StudyContext
    unit_id: UUID
    concept_version_ids: list[UUID] = Field(min_length=1, max_length=100)
    count: int = Field(ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard", "mixed"]
    difficulty_plan: list[Difficulty] | None = Field(default=None, min_length=1, max_length=20)
    question_types: list[Literal["multiple_choice", "short_answer"]] = Field(
        default_factory=lambda: ["multiple_choice", "short_answer"], min_length=1, max_length=2
    )
    explanation_language: Language = "ko"

    @model_validator(mode="after")
    def valid_difficulty_plan(self):
        if self.difficulty == "mixed":
            if self.difficulty_plan is None or len(self.difficulty_plan) != self.count:
                raise ValueError("Mixed difficulty requires an explicit plan matching count.")
            if len(set(self.difficulty_plan)) < 2:
                raise ValueError("Mixed difficulty requires at least two levels.")
        elif self.difficulty_plan is not None:
            raise ValueError("difficulty_plan is only used with mixed difficulty.")
        return self


class QuestionGenerationRequest(QuestionRequest):
    """Public internal HTTP input; smaller batches remain an implementation detail."""

    count: int = Field(ge=5, le=20)


class GeneratedQuestions(QuestionBatch):
    verification: list[AnswerVerification]


class GradeRequest(ModelRequest):
    context: StudyContext
    stored_question: StoredQuestion
    answer: QuestionAnswer


class AnswerJudgment(Model):
    status: Literal["correct", "incorrect", "unverified"]
    feedback: Text
    evidence: list[SourceEvidence] = Field(max_length=200)


class QuestionResult(Model):
    is_correct: bool
    feedback: Text
    correct_answer: QuestionAnswer
    explanation: Text
    evidence: list[SourceEvidence] = Field(min_length=1, max_length=200)


class Message(Model):
    role: Literal["user", "assistant"]
    content: Text


class EvaluationRequest(ModelRequest):
    context: StudyContext
    concept_version_id: UUID
    history: list[Message] = Field(min_length=1, max_length=100)
    explanation_language: Language = "ko"


class ExplanationRequest(ModelRequest):
    context: StudyContext
    concept_version_id: UUID
    explanation_language: Language = "ko"


class ExplanationContent(Model):
    title: Name
    explanation_markdown: str = Field(min_length=1, max_length=24000)
    evidence: list[SourceEvidence] = Field(min_length=1, max_length=200)


class CriterionJudgment(Model):
    checklist_item_id: UUID
    status: Literal["explained", "partial", "omission", "error", "unverified"]
    description: Text
    confidence: Literal["low", "medium", "high"]
    evidence: list[SourceEvidence] = Field(max_length=200)
    learner_quote: str = Field(max_length=2000)


class EvaluationJudgments(Model):
    items: list[CriterionJudgment] = Field(max_length=50)
    next_action: Text


class ReportItem(Model):
    concept_id: UUID
    concept_version_id: UUID
    checklist_item_id: UUID
    status: Literal["explained", "partial", "omission", "error", "unverified"]
    description: Text
    confidence: Literal["low", "medium", "high"]
    evidence: list[SourceEvidence]


class EvaluationResult(Model):
    concept_version_id: UUID
    coverage: float | None = Field(ge=0, le=1)
    coverage_formula: Literal["weighted-v1"] = "weighted-v1"
    assessable_count: int = Field(ge=0)
    weighted_score: float = Field(ge=0)
    items: list[ReportItem] = Field(max_length=50)
    next_action: Text
    weak_checklist_item_ids: list[UUID]
    weak_concept_ids: list[UUID]


class DetailedEvaluationRequest(EvaluationRequest):
    previous_result: EvaluationResult | None = None


class DetailedCriterionJudgment(CriterionJudgment):
    correct_explanation: str = Field(max_length=2000)
    improvement: str = Field(max_length=2000)


class DetailedJudgments(Model):
    items: list[DetailedCriterionJudgment] = Field(max_length=50)
    next_action: Text


class CriterionFeedback(Model):
    checklist_item_id: UUID
    learner_quote: str
    correct_explanation: str
    improvement: str
    evidence: list[SourceEvidence]


class EvaluationComparison(Model):
    comparable: bool
    reason: Literal["same_version", "version_mismatch", "checklist_mismatch", "unverified"]
    coverage_delta: float | None
    changes: list[dict[str, str]]


class DetailedEvaluationResult(Model):
    report: EvaluationResult
    feedback: list[CriterionFeedback]
    comparison: EvaluationComparison | None = None


class PersistedReportItem(ReportItem):
    id: UUID


class SessionReport(Model):
    id: UUID
    session_id: UUID
    concept_version_id: UUID
    version_no: int = Field(ge=1)
    coverage: float | None = Field(ge=0, le=1)
    coverage_formula: Literal["weighted-v1"]
    assessable_count: int = Field(ge=0)
    weighted_score: float = Field(ge=0)
    items: list[PersistedReportItem] = Field(max_length=50)
    next_action: Text
    action_id: UUID


class TagRequest(ModelRequest):
    question: PublicQuestionDraft
    context: StudyContext
    subject_hint: Name | None = None


class Keywords(Model):
    keywords: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1, max_length=5
    )


class TimeConcept(Model):
    concept_version_id: UUID
    name: Name
    remaining_criteria: int = Field(ge=0, le=50)
    remaining_questions: int = Field(default=0, ge=0, le=200)
    review_cycles: int = Field(default=1, ge=0, le=10)
    review_criteria: int | None = Field(default=None, ge=0, le=50)
    weakness: float = Field(default=0, ge=0, le=1)
    difficulty: Difficulty = "medium"
    importance: float = Field(default=0.5, ge=0, le=1)
    teachback_sessions: int = Field(default=0, ge=0, le=10)


class StudyTimeRequest(Model):
    exam_date: date
    today: date
    concepts: list[TimeConcept] = Field(min_length=1, max_length=100)


class ConceptTime(Model):
    concept_version_id: UUID
    name: Name
    learning_minutes: int
    question_minutes: int
    review_minutes: int
    total_minutes: int
    priority: float
    teachback_minutes: int = 0


class StudyTimeEstimate(Model):
    estimate: Literal[True] = True
    remaining_days: int
    total_minutes: int
    recommended_daily_minutes: int
    concepts: list[ConceptTime]
    assumptions: list[str]
    priority_guidance: str
