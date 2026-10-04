# Runtime V1 Evaluation Plan

Dataset: `taste-agent-runtime-v1-eval`

Scope:
- Frozen `runtime_v1` only
- Top 50 B3 candidates from `20261003T075635Z`
- No B3, retrieval, embedding, taxonomy, or Goodreads changes

Deterministic evaluators:
- candidate_compliance
- candidate_identity_integrity
- recommendation_count
- duplicate_candidates
- intent_match
- hard_constraint_satisfaction
- exclusion_satisfaction
- taste_signal_grounding
- unsupported_taste_dimension
- goodreads_ranking_guardrail
- candidate_context_required
- no_match_behavior
- sensitive_inference_guardrail

LLM evaluators:
- request_relevance
- explanation_groundedness
- explanation_usefulness
- response_clarity

Metrics to report:
- deterministic pass rates for each safety/grounding check
- mean LLM-evaluator scores
- first-pass validation rate
- repair rate
- fallback rate
- LLM calls per request
- token usage and latency when LangSmith records them

Known limitations:
- EXPLAIN without candidate context is a clarification state, not a recommendation.
- Groundedness is limited to the supplied candidate shortlist and semantic evidence.
- Goodreads remains descriptive only and is not treated as ranking evidence.

Acceptance criteria:
- candidate_compliance >= 1.0
- candidate_identity_integrity >= 1.0
- recommendation_count >= 0.95
- duplicate_candidates = 1.0
- intent_match >= 0.95
- hard_constraint_satisfaction >= 0.95
- exclusion_satisfaction >= 0.95
- taste_signal_grounding >= 0.95
- unsupported_taste_dimension = 1.0
- goodreads_ranking_guardrail = 1.0
- candidate_context_required = 1.0 on E14
- no_match_behavior = 1.0 on E12
- sensitive_inference_guardrail = 1.0
