import unicodedata

from api.router import ModelRouter
from src.context import make_prompt, payload
from src.errors import InvalidResponseError
from src.models import Keywords, TagRequest
from src.questions import validate_question

PROMPT = """Suggest 1-5 relevant search keywords for this question using its subject, unit,
concepts and source context. Output short terms, not sentences or generic words like 'question'.
Prefer specific concept names. Do not include private material titles, user identities, IDs,
source quotations or answers merely for exposure. Remove irrelevant and duplicate suggestions.
The backend performs exact keyword search, not vector or personalized recommendations.
"""


def normalize_keyword(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().lower()


def suggest(request: TagRequest, router: ModelRouter) -> Keywords:
    concepts = validate_question(request.question, request.context)
    result = router.generate_json(
        "question_tagging",
        model_id=request.model_id,
        prompt=make_prompt(
            PROMPT,
            {
                **payload(request.context, concepts),
                "question": request.question.model_dump(mode="json"),
                "subject_hint": request.subject_hint,
            },
        ),
        schema=Keywords,
    )
    # These are AI suggestions; remove duplicates before a user submits publication keywords.
    values = list(dict.fromkeys(normalize_keyword(k) for k in result.keywords))
    if not values or any(not v or len(v) > 40 for v in values):
        raise InvalidResponseError("Keyword suggestions violate normalization/length constraints.")
    return Keywords(keywords=values)
