from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, EvaluationResults, run_evaluator
from pydantic import BaseModel, Field

from mvp.src.generative.runtime_version import GENERATION_MODEL
from mvp.src.semantics import SEMANTIC_COLUMNS


DEFAULT_JUDGE_MODEL = GENERATION_MODEL
_GOODREADS_PATTERNS = (
    r"because of your books",
    r"because your books",
    r"reading taste",
    r"reading profile",
    r"goodreads.*rank",
    r"goodreads.*score",
    r"books.*increased",
)
_SENSITIVE_PATTERNS = (
    r"your personality",
    r"your identity",
    r"says about you",
    r"psychology",
    r"type of person",
    r"you are an",
    r"you are a",
    r"because you are",
)


class LLMJudgement(BaseModel):
    request_relevance: int = Field(ge=1, le=5)
    explanation_groundedness: int = Field(ge=1, le=5)
    explanation_usefulness: int = Field(ge=1, le=5)
    response_clarity: int = Field(ge=1, le=5)
    comment: str = ""


_LLM_SCORE_CACHE: dict[tuple[str, str], LLMJudgement] = {}


def _score_result(key: str, passed: bool, comment: str = "") -> EvaluationResult:
    return EvaluationResult(key=key, score=1.0 if passed else 0.0, comment=comment or None)


def _example_payload(example: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(example, dict):
        inputs = example.get("inputs", {}) or {}
        outputs = example.get("outputs", {}) or {}
        metadata = example.get("metadata", {}) or {}
        return inputs, outputs, metadata
    inputs = getattr(example, "inputs", {}) or {}
    outputs = getattr(example, "outputs", {}) or {}
    metadata = getattr(example, "metadata", {}) or {}
    return inputs, outputs, metadata


def _run_payload(run: Any) -> dict[str, Any]:
    outputs = getattr(run, "outputs", {}) or {}
    if isinstance(outputs, dict) and "output" in outputs and isinstance(outputs["output"], dict):
        return outputs["output"]
    if isinstance(outputs, dict):
        return outputs
    return {}


def _response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response", {})
    return response if isinstance(response, dict) else {}


def _runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime_metadata", {})
    return runtime if isinstance(runtime, dict) else {}


def _debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    debug = payload.get("debug", {})
    return debug if isinstance(debug, dict) else {}


def _candidate_context(payload: dict[str, Any]) -> list[dict[str, Any]]:
    debug = _debug_payload(payload)
    context = debug.get("candidate_context", [])
    if isinstance(context, list):
        return [item for item in context if isinstance(item, dict)]
    return []


def _candidate_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for candidate in _candidate_context(payload):
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if candidate_id:
            lookup[candidate_id] = candidate
    return lookup


def _recommendations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = _response_payload(payload)
    recs = response.get("recommendations", [])
    if isinstance(recs, list):
        return [item for item in recs if isinstance(item, dict)]
    return []


def _expected_inputs(example: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
    inputs, outputs, metadata = _example_payload(example)
    return str(inputs.get("query", "")), outputs, metadata


def _expected_behavior(outputs: dict[str, Any]) -> str:
    return str(outputs.get("expected_behavior", "")).strip()


def _expected_intent(outputs: dict[str, Any]) -> str:
    return str(outputs.get("expected_intent", "")).strip()


def _expected_constraints(outputs: dict[str, Any]) -> dict[str, Any]:
    value = outputs.get("expected_hard_constraints", {})
    return value if isinstance(value, dict) else {}


def _is_grounded_behavior(outputs: dict[str, Any]) -> bool:
    behavior = _expected_behavior(outputs)
    return behavior in {
        "five_grounded_recommendations",
        "theme_grounded_recommendations",
        "language_constraint",
        "decade_constraint",
        "genre_constraint",
        "compound_constraint",
        "novelty_grounded_selection",
        "genre_exclusion",
        "grounded_explain_exact_candidate",
        "grounded_explain_no_unsupported_facts",
        "ambiguous_theme_grounded",
    }


def _expected_count(outputs: dict[str, Any]) -> int:
    behavior = _expected_behavior(outputs)
    if behavior in {"needs_candidate_context", "explicit_no_match"}:
        return 0
    if behavior.startswith("grounded_explain"):
        return 1
    return 5


def _candidate_satisfies_constraints(candidate: dict[str, Any], constraints: dict[str, Any]) -> bool:
    genres = {str(item).lower() for item in candidate.get("genres", []) if item is not None}
    languages = str(candidate.get("original_language", "")).lower()
    countries = {str(item).lower() for item in candidate.get("production_countries", []) if item is not None}
    year = candidate.get("year")

    for genre in constraints.get("genres", []):
        if str(genre).lower() not in genres:
            return False
    for language in constraints.get("languages", []):
        if languages != str(language).lower():
            return False
    for country in constraints.get("countries", []):
        if str(country).lower() not in countries:
            return False
    for decade in constraints.get("decades", []):
        if year is None:
            return False
        if f"{int(year) // 10 * 10}s" != str(decade):
            return False
    return True


def _candidate_satisfies_request(candidate: dict[str, Any], constraints: dict[str, Any], exclusions: list[str]) -> bool:
    if not _candidate_satisfies_constraints(candidate, constraints):
        return False
    if not exclusions:
        return True
    haystack = " ".join(
        [
            str(candidate.get("title", "")).lower(),
            " ".join(str(item) for item in candidate.get("genres", [])).lower(),
            str(candidate.get("original_language", "")).lower(),
            " ".join(str(item) for item in candidate.get("production_countries", [])).lower(),
            " ".join(str(item) for item in candidate.get("candidate_provenance", [])).lower(),
        ]
    )
    return not any(str(exclusion).lower() in haystack for exclusion in exclusions)


def _exclusions_satisfied(candidate: dict[str, Any], exclusions: list[str]) -> bool:
    haystack = " ".join(
        [
            str(candidate.get("title", "")).lower(),
            " ".join(str(item) for item in candidate.get("genres", [])).lower(),
            str(candidate.get("original_language", "")).lower(),
            " ".join(str(item) for item in candidate.get("production_countries", [])).lower(),
        ]
    )
    return not any(str(exclusion).lower() in haystack for exclusion in exclusions)


def _aggregate_text(payload: dict[str, Any]) -> str:
    response = _response_payload(payload)
    chunks = [
        str(response.get("response_summary", "")),
        str(response.get("methodology_note", "")),
    ]
    for recommendation in _recommendations(payload):
        chunks.extend(
            [
                str(recommendation.get("why_it_may_fit", "")),
                str(recommendation.get("request_match", "")),
                str(recommendation.get("caveat", "")),
                " ".join(str(item) for item in recommendation.get("taste_signals", [])),
            ]
        )
    return " ".join(chunks).lower()


@run_evaluator
def candidate_compliance(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    lookup = _candidate_lookup(payload)
    passed = all(str(rec.get("candidate_id", "")) in lookup for rec in _recommendations(payload))
    return _score_result("candidate_compliance", passed)


@run_evaluator
def candidate_identity_integrity(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    lookup = _candidate_lookup(payload)
    passed = True
    for rec in _recommendations(payload):
        candidate = lookup.get(str(rec.get("candidate_id", "")))
        if not candidate:
            passed = False
            break
        if str(rec.get("title", "")) != str(candidate.get("title", "")):
            passed = False
            break
        if rec.get("year") is not None and candidate.get("year") is not None and int(rec["year"]) != int(candidate["year"]):
            passed = False
            break
    return _score_result("candidate_identity_integrity", passed)


@run_evaluator
def recommendation_count(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    behavior = _expected_behavior(outputs)
    if behavior in {"needs_candidate_context", "explicit_no_match"}:
        expected = 0
    elif behavior.startswith("grounded_explain"):
        expected = 1
    else:
        constraints = _expected_constraints(outputs)
        exclusions = [str(item) for item in constraints.get("exclusions", [])]
        eligible = [
            candidate
            for candidate in _candidate_context(payload)
            if _candidate_satisfies_request(candidate, constraints, exclusions)
        ]
        limit = int(_runtime_payload(payload).get("recommendation_count", 5) or 5)
        expected = min(limit, len(eligible))
    rec_count = len(_recommendations(payload))
    passed = rec_count == expected
    return _score_result("recommendation_count", passed, comment=f"expected={expected}, actual={rec_count}")


@run_evaluator
def duplicate_candidates(run: Any, example: Any) -> EvaluationResult:
    rec_ids = [str(rec.get("candidate_id", "")) for rec in _recommendations(_run_payload(run))]
    passed = len(rec_ids) == len(set(rec_ids))
    return _score_result("duplicate_candidates", passed)


@run_evaluator
def intent_match(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    response = _response_payload(payload)
    actual = ""
    if isinstance(response.get("intent"), dict):
        actual = str(response["intent"].get("intent_type", "")).strip()
    if not actual:
        debug = _debug_payload(payload)
        request = debug.get("request", {}) if isinstance(debug.get("request", {}), dict) else {}
        intent = request.get("intent", {}) if isinstance(request, dict) else {}
        if isinstance(intent, dict):
            actual = str(intent.get("intent_type", "")).strip()
    expected = _expected_intent(outputs)
    passed = actual == expected
    return _score_result("intent_match", passed, comment=f"expected={expected}, actual={actual}")


@run_evaluator
def hard_constraint_satisfaction(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    constraints = _expected_constraints(outputs)
    if not any(constraints.get(key) for key in ("genres", "languages", "countries", "decades")):
        return EvaluationResult(key="hard_constraint_satisfaction", score=None, comment="N/A - no structured hard constraints in this case.")
    recs = _recommendations(payload)
    if not recs:
        passed = _expected_behavior(outputs) in {"needs_candidate_context", "explicit_no_match"}
        return _score_result("hard_constraint_satisfaction", passed)
    lookup = _candidate_lookup(payload)
    passed = True
    for rec in recs:
        candidate_id = str(rec.get("candidate_id", ""))
        candidate = lookup.get(candidate_id)
        if not candidate or not _candidate_satisfies_constraints(candidate, constraints):
            passed = False
            break
    return _score_result("hard_constraint_satisfaction", passed)


@run_evaluator
def exclusion_satisfaction(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    exclusions = [str(item) for item in _expected_constraints(outputs).get("exclusions", [])]
    if not exclusions:
        return EvaluationResult(key="exclusion_satisfaction", score=None, comment="N/A - no structured exclusions in this case.")
    recs = _recommendations(payload)
    if not recs:
        passed = _expected_behavior(outputs) in {"needs_candidate_context", "explicit_no_match"}
        return _score_result("exclusion_satisfaction", passed)
    passed = all(_exclusions_satisfied(candidate, exclusions) for candidate in _candidate_context(payload))
    return _score_result("exclusion_satisfaction", passed)


@run_evaluator
def taste_signal_grounding(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    candidate_context = _candidate_context(payload)
    candidate_evidence = {
        str(candidate.get("candidate_id", "")): {
            str(item.get("dimension", "")) for item in candidate.get("semantic_evidence", []) if isinstance(item, dict)
        }
        for candidate in candidate_context
    }
    passed = True
    for rec in _recommendations(payload):
        candidate_id = str(rec.get("candidate_id", ""))
        dims = candidate_evidence.get(candidate_id, set())
        for signal in rec.get("taste_signals", []):
            signal_dim = str(signal).split(":", 1)[0].strip()
            if signal_dim not in dims:
                passed = False
                break
        if not passed:
            break
    return _score_result("taste_signal_grounding", passed)


@run_evaluator
def unsupported_taste_dimension(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    passed = True
    for rec in _recommendations(payload):
        for signal in rec.get("taste_signals", []):
            signal_dim = str(signal).split(":", 1)[0].strip()
            if signal_dim and signal_dim not in SEMANTIC_COLUMNS:
                passed = False
                break
        if not passed:
            break
    return _score_result("unsupported_taste_dimension", passed)


@run_evaluator
def goodreads_ranking_guardrail(run: Any, example: Any) -> EvaluationResult:
    text = _aggregate_text(_run_payload(run))
    passed = not any(re.search(pattern, text, flags=re.I) for pattern in _GOODREADS_PATTERNS)
    return _score_result("goodreads_ranking_guardrail", passed)


@run_evaluator
def candidate_context_required(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    runtime = _runtime_payload(payload)
    passed = _expected_behavior(outputs) == "needs_candidate_context" and runtime.get("generation_status") == "needs_candidate_context" and len(_recommendations(payload)) == 0
    return _score_result("candidate_context_required", passed)


@run_evaluator
def no_match_behavior(run: Any, example: Any) -> EvaluationResult:
    _, outputs, _ = _example_payload(example)
    payload = _run_payload(run)
    runtime = _runtime_payload(payload)
    passed = _expected_behavior(outputs) == "explicit_no_match" and runtime.get("generation_status") == "no_match" and len(_recommendations(payload)) == 0
    return _score_result("no_match_behavior", passed)


@run_evaluator
def interaction_routing_contract(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    runtime = _runtime_payload(payload)
    mode = str(runtime.get("interaction_mode", "recommend")).strip()
    recs = _recommendations(payload)
    text = _aggregate_text(payload)
    target_candidate_id = str(runtime.get("target_candidate_id", "") or "").strip()

    if mode == "explain_target":
        lookup = _candidate_lookup(payload)
        passed = (
            bool(target_candidate_id)
            and len(recs) == 1
            and str(recs[0].get("candidate_id", "")) == target_candidate_id
            and target_candidate_id in lookup
        )
        return _score_result("interaction_routing_contract", passed, comment=f"mode={mode}, target={target_candidate_id}")

    if mode == "explain_needs_target":
        passed = len(recs) == 0 and "target" in text
        return _score_result("interaction_routing_contract", passed, comment=f"mode={mode}")

    if mode == "provenance":
        passed = len(recs) == 0 and "goodreads" in text and any(token in text for token in ("descriptive only", "not used", "does not rank", "does not select", "not rank"))
        return _score_result("interaction_routing_contract", passed, comment=f"mode={mode}")

    if mode == "profile_boundary":
        passed = len(recs) == 0 and not any(re.search(pattern, text, flags=re.I) for pattern in _SENSITIVE_PATTERNS) and any(token in text for token in ("taste pattern", "cultural preference", "do not infer", "don't infer", "cannot infer", "not infer"))
        return _score_result("interaction_routing_contract", passed, comment=f"mode={mode}")

    return _score_result("interaction_routing_contract", True, comment=f"mode={mode}")


@run_evaluator
def sensitive_inference_guardrail(run: Any, example: Any) -> EvaluationResult:
    text = _aggregate_text(_run_payload(run))
    passed = not any(re.search(pattern, text, flags=re.I) for pattern in _SENSITIVE_PATTERNS)
    return _score_result("sensitive_inference_guardrail", passed)


def _judge_payload(run: Any, example: Any) -> dict[str, Any]:
    payload = _run_payload(run)
    inputs, outputs, _ = _example_payload(example)
    candidate_context = _candidate_context(payload)
    compact_context = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "title": candidate.get("title"),
            "year": candidate.get("year"),
            "genres": candidate.get("genres", []),
            "language": candidate.get("original_language"),
            "countries": candidate.get("production_countries", []),
        }
        for candidate in candidate_context
    ]
    return {
        "query": inputs.get("query", ""),
        "expected_intent": outputs.get("expected_intent", ""),
        "expected_behavior": outputs.get("expected_behavior", ""),
        "candidate_id": inputs.get("candidate_id"),
        "response": _response_payload(payload),
        "runtime_metadata": _runtime_payload(payload),
        "candidate_context": compact_context,
        "validation_passed": payload.get("validation_passed"),
        "validation_errors": payload.get("validation_errors", []),
    }


def _score_with_llm(run: Any, example: Any) -> LLMJudgement:
    keys = _judge_payload(run, example)
    prompt = (
        "You are judging a grounded recommendation runtime.\n"
        "Score each dimension from 1 to 5.\n"
        "1 = poor, 3 = acceptable, 5 = excellent.\n\n"
        f"User query: {keys['query']}\n"
        f"Expected intent: {keys['expected_intent']}\n"
        f"Expected behavior: {keys['expected_behavior']}\n"
        f"Candidate context: {keys['candidate_context']}\n"
        f"Runtime metadata: {keys['runtime_metadata']}\n"
        f"Actual response: {keys['response']}\n\n"
        "Rubric:\n"
        "- request_relevance: Does the response address the request or correctly ask for needed context?\n"
        "- explanation_groundedness: Does any explanation rely only on the supplied grounded context?\n"
        "- explanation_usefulness: Is the explanation or clarification helpful to the user?\n"
        "- response_clarity: Is the response concise, understandable, and appropriately cautious?\n"
    )
    from mvp.src.config import get_api_keys

    openai_api_key = get_api_keys().openai_api_key
    cache_key = (str(getattr(run, "id", "run")), str(getattr(example, "id", "example")))
    if cache_key in _LLM_SCORE_CACHE:
        return _LLM_SCORE_CACHE[cache_key]
    if not openai_api_key:
        judged = LLMJudgement(
            request_relevance=1,
            explanation_groundedness=1,
            explanation_usefulness=1,
            response_clarity=1,
            comment="OPENAI_API_KEY missing; LLM judge was not run.",
        )
    else:
        llm = ChatOpenAI(model=DEFAULT_JUDGE_MODEL, temperature=0, api_key=openai_api_key)
        judged = llm.with_structured_output(LLMJudgement).invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Return the structured scores now."),
            ]
        )
    _LLM_SCORE_CACHE[cache_key] = judged
    return judged


@run_evaluator
def request_relevance(run: Any, example: Any) -> EvaluationResult:
    judged = _score_with_llm(run, example)
    return EvaluationResult(key="request_relevance", score=float(judged.request_relevance), comment=judged.comment or None)


@run_evaluator
def explanation_groundedness(run: Any, example: Any) -> EvaluationResult:
    judged = _score_with_llm(run, example)
    return EvaluationResult(key="explanation_groundedness", score=float(judged.explanation_groundedness), comment=judged.comment or None)


@run_evaluator
def explanation_usefulness(run: Any, example: Any) -> EvaluationResult:
    judged = _score_with_llm(run, example)
    return EvaluationResult(key="explanation_usefulness", score=float(judged.explanation_usefulness), comment=judged.comment or None)


@run_evaluator
def response_clarity(run: Any, example: Any) -> EvaluationResult:
    judged = _score_with_llm(run, example)
    return EvaluationResult(key="response_clarity", score=float(judged.response_clarity), comment=judged.comment or None)


llm_request_relevance = request_relevance
