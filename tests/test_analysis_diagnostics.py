"""Analysis prompt and source-reference regressions, no external calls."""

import json

import pytest

from src.analysis import PROMPT
from src.models import AnalysisPart


def test_analysis_prompt_names_match_contract():
    assert "estimated_difficulty" in PROMPT and "estimated_importance" in PROMPT
    schema = AnalysisPart.model_json_schema()["$defs"]["DraftConcept"]["properties"]
    assert "estimated_difficulty" in schema and "estimated_importance" in schema
    assert "difficulty" not in schema and "importance" not in schema


def indexed_data(refs):
    return {
        "units": [
            {
                "title": "Unit",
                "concepts": [
                    {
                        "name": "Concept",
                        "summary": "Summary",
                        "checklist_items": ["Criterion"],
                        "keywords": ["word"],
                        "evidence": refs,
                        "estimated_difficulty": None,
                        "estimated_importance": None,
                    }
                ],
            }
        ]
    }


def test_indexed_evidence_preserves_server_owned_metadata_and_verbatim_text(context):
    from src.analysis import evidence_catalog, expand_evidence, indexed_analysis_schema
    from src.context import validate_sources

    sources = context.chunks
    catalog = evidence_catalog(sources)
    schema = indexed_analysis_schema(len(catalog))
    expanded = expand_evidence(schema.model_validate(indexed_data([0])), catalog, schema)
    evidence = expanded.units[0].concepts[0].evidence
    assert evidence == [catalog[0]]
    validate_sources(evidence, sources, generated=True)
    assert evidence[0].chunk_id == sources[0].id
    assert evidence[0].material_id == sources[0].material_id
    assert evidence[0].material_name == sources[0].material_name
    assert evidence[0].excerpt in sources[0].text
    assert evidence[0].page_number == sources[0].page_number


@pytest.mark.parametrize("refs", [[-1], [9999], ["0"], [True], [{"excerpt": "invented"}]])
def test_indexed_evidence_does_not_accept_invalid_indices(context, refs):
    from api.base import validate_json
    from src.analysis import evidence_catalog, indexed_analysis_schema
    from src.errors import InvalidResponseError

    schema = indexed_analysis_schema(len(evidence_catalog(context.chunks)))
    with pytest.raises(InvalidResponseError):
        validate_json(json.dumps(indexed_data(refs)), schema)


def test_indexed_evidence_duplicate_and_corrupted_catalog_rejected(context):
    from src.analysis import evidence_catalog, expand_evidence, indexed_analysis_schema
    from src.context import validate_sources
    from src.errors import InvalidResponseError

    catalog = evidence_catalog(context.chunks)
    schema = indexed_analysis_schema(len(catalog))
    with pytest.raises(InvalidResponseError, match="duplicate"):
        expand_evidence(schema.model_validate(indexed_data([0, 0])), catalog, schema)
    corrupted = catalog[0].model_copy(update={"excerpt": "NOT IN SOURCE"})
    with pytest.raises(InvalidResponseError):
        validate_sources([corrupted], context.chunks, generated=True)


@pytest.mark.parametrize("field,limit", [
    ("checklist_items", 50), ("keywords", 5), ("evidence", 20),
    ("concepts", 100), ("units", 200),
])
def test_simplified_analysis_wire_preserves_all_array_limits(field, limit):
    from jsonschema import Draft202012Validator

    from api.base import validate_json
    from api.gemini import compatible_schema
    from src.analysis import indexed_analysis_schema
    from src.errors import InvalidResponseError

    data = indexed_data([0])
    concept = data["units"][0]["concepts"][0]
    if field == "units":
        data["units"] *= limit + 1
    elif field == "concepts":
        data["units"][0]["concepts"] *= limit + 1
    else:
        concept[field] *= limit + 1
    model = indexed_analysis_schema(1)
    wire = compatible_schema(model.model_json_schema())
    Draft202012Validator(wire).validate(data)
    with pytest.raises(InvalidResponseError):
        validate_json(json.dumps(data), model)


def test_analyze_sdk_wire_and_final_source_relations(context, settings):
    import httpx
    from jsonschema import Draft202012Validator

    from api.router import ModelRouter
    from src.analysis import analyze
    from src.models import AnalyzeRequest
    from tests.test_gemini import adapter, envelope

    calls = []
    original = AnalysisPart.model_json_schema()

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        config = body["generationConfig"]
        wire = config["responseJsonSchema"]
        assert '"maxItems"' not in json.dumps(wire)
        assert '"minItems"' in json.dumps(wire)
        assert wire["additionalProperties"] is False
        assert "responseSchema" not in config
        assert config["maxOutputTokens"] == 8192
        Draft202012Validator.check_schema(wire)
        Draft202012Validator(wire).validate(indexed_data([0]))
        return httpx.Response(200, json=envelope(indexed_data([0])))

    client = adapter(handler)
    try:
        result = analyze(
            AnalyzeRequest(
                model_id=settings.default_model_id, chunks=[context.chunks[0]], max_calls=1,
            ),
            ModelRouter(settings, {"gemini": client}),
        )
    finally:
        client.close()
    assert len(calls) == 1
    assert result.state == "draft"
    evidence = result.units[0].concepts[0].evidence[0]
    assert evidence.chunk_id == context.chunks[0].id
    assert evidence.excerpt in context.chunks[0].text
    assert AnalysisPart.model_json_schema() == original
