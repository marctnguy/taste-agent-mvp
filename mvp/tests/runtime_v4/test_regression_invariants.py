from __future__ import annotations

from mvp.src.generative.schemas import RecommendationIntent, RecommendationItem, RecommendationResponse
from mvp.src.generative_v4.schemas import RuntimeV4Metadata, RuntimeV4Result, SelectionRecord
from mvp.src.generative_v4.validators import validate_runtime_v4_result


def _baseline_result(text: str) -> RuntimeV4Result:
    intent = RecommendationIntent(intent_type="GENERAL_DISCOVERY")
    item = RecommendationItem(
        candidate_id="1",
        title="Test Film",
        year=2000,
        why_it_may_fit=text,
        taste_signals=[],
        request_match="Grounded request match.",
        caveat=None,
    )
    response = RecommendationResponse(
        intent=intent,
        recommendations=[item],
        response_summary=text,
        methodology_note="Grounded selection only.",
    )
    metadata = RuntimeV4Metadata(
        run_id="test",
        generation_status="generated",
        validation_passed=True,
        hard_filters_applied=False,
    )
    return RuntimeV4Result(
        response=response,
        runtime_metadata=metadata,
        retrieval_diagnostics={},
        qualification_records=[],
        selection_records=[SelectionRecord(candidate_id="1", selection_rank=1, qualification_status="strong")],
        validation_passed=True,
        validation_errors=[],
        debug={"candidate_context": [{"candidate_id": "1", "qualification_status": "strong", "supported_request_aspects": ["genre:drama"], "semantic_evidence": []}]},
    )


def test_goodreads_ranking_guardrail_validator_rejects_goodreads_evidence() -> None:
    result = _baseline_result("This fits because of your books and Goodreads ranking.")
    validation = validate_runtime_v4_result(result)

    assert validation.passed is False
    assert any("Goodreads" in error for error in validation.errors)


def test_sensitive_inference_guardrail_validator_rejects_identity_claims() -> None:
    result = _baseline_result("This says about you as a person and your identity.")
    validation = validate_runtime_v4_result(result)

    assert validation.passed is False
    assert any("Sensitive inference" in error for error in validation.errors)


def test_certainty_language_guardrail_validator_rejects_probability_language() -> None:
    result = _baseline_result("This is guaranteed to be certain and likely to like.")
    validation = validate_runtime_v4_result(result)

    assert validation.passed is False
    assert any("probability or certainty" in error for error in validation.errors)


def test_validator_rejects_strong_candidates_missing_an_all_of_requirement() -> None:
    result = _baseline_result("This is a grounded request match.")
    result.debug["request_spec"] = {
        "semantic_requirement_groups": [
            {
                "group_id": "required_semantics",
                "mode": "all_of",
                "concepts": [
                    {"aspect_id": "performers", "concept": "performers"},
                    {"aspect_id": "fame", "concept": "fame"},
                ],
                "source_span": "performers and fame",
            }
        ],
        "semantic_exclusions": [],
        "structured_constraints": {},
    }
    result.debug["candidate_context"] = [
        {
            "candidate_id": "1",
            "qualification_status": "strong",
            "supported_request_aspects": ["performers"],
            "semantic_evidence": [],
        }
    ]

    validation = validate_runtime_v4_result(result)

    assert validation.passed is False
    assert any("all_of semantic group" in error for error in validation.errors)


def test_validator_rejects_watched_candidates_from_consumed_history() -> None:
    result = _baseline_result("This is a grounded request match.")
    result.debug["watched_ids"] = ["1"]

    validation = validate_runtime_v4_result(result)

    assert validation.passed is False
    assert any("consumed-film history" in error for error in validation.errors)
