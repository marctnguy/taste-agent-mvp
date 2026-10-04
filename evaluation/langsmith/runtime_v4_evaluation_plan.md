# Runtime V4 Evaluation Plan

Dataset: `taste-agent-runtime-v4-eval`

## Scope

- Frozen `runtime_v4` only
- Dynamic TMDB candidate generation at runtime
- No B3, PCA, Ridge, taxonomy, or Goodreads changes

## Deterministic checks

- candidate_compliance
- candidate_identity_integrity
- hard_constraint_satisfaction
- watched_exclusion
- qualification_compliance
- unsupported_candidate_rejection
- partial_caveat_preservation
- request_claim_grounding
- taste_claim_grounding
- duplicate_candidates
- no_match_allowed
- fewer_than_five_allowed
- sensitive_inference_guardrail
- Goodreads separation
- B3 interpretation compliance

## LLM / judgment checks

- request_relevance
- qualification_defensibility
- explanation_groundedness
- explanation_usefulness
- response_clarity

## Stage-level metrics

- retrieval coverage
- hard-constraint satisfaction
- qualification pass rate
- unsupported-candidate rejection rate
- selection pass rate
- explanation grounding rate
- abstention rate
- repair rate
- fallback rate
- LLM calls
- token usage
- latency

## Acceptance criteria

- Request relevance must be grounded before B3 is used.
- Unsupported candidates must never be selected.
- Partial matches must keep caveats.
- No-match is valid.
- Fewer than five recommendations is valid.
- Goodreads must remain descriptive only.
- Sensitive inference must remain blocked.

