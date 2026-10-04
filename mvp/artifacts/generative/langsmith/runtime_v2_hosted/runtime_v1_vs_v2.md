# runtime_v1 vs runtime_v2

Hosted v2 experiment: `taste-agent-runtime-v2-975cf538` (`414dd2a1-d63d-468d-9ce5-73bb017b9352`)

The earlier `runtime_v2_iteration` artifact is preserved as a non-comparable local-fallback run. This report covers the hosted LangSmith baseline only.

## Preconditions
- `runtime_version = runtime_v2`
- `intent_prompt_version = intent_v2`
- `recommendation_prompt_version = recommendation_v1`
- `candidate_context_size = 50`
- B3 / candidate retrieval / semantic taxonomy / Goodreads behavior unchanged

## Result Summary
- Trace coverage: 18/18
- Actual LLM calls occurred: True
- Validation pass rate: 0.9444
- Repair rate: 0.2222
- Fallback rate: 0.1111
- LLM calls/request: mean 1.1111, median 1.0, max 2.0
- Token usage: input 17218.0 / output 488.1 / total 17706.1 mean; totals 309924 / 8785 / 318709
- Latency: mean 5.578s, median 5.490s, p95 10.993s

## Critical Invariants
- candidate_compliance: v1 1.0000 -> v2 1.0000
- candidate_identity_integrity: v1 1.0000 -> v2 1.0000
- hard_constraint_satisfaction: v1 1.0000 -> v2 1.0000
- taste_signal_grounding: v1 1.0000 -> v2 1.0000
- unsupported_taste_dimension: v1 1.0000 -> v2 1.0000
- goodreads_ranking_guardrail: v1 1.0000 -> v2 1.0000
- sensitive_inference_guardrail: v1 1.0000 -> v2 1.0000

## Applicable-Case Metrics
- recommendation_count: v1 0.8333 -> v2 1.0000
- intent_match: v1 0.9444 -> v2 1.0000
- exclusion_satisfaction: v1 0.9444 -> v2 1.0000
- candidate_context_required: v1 0.0556 -> v2 0.0556
- no_match_behavior: v1 0.0556 -> v2 0.0556

## LLM Judge Metrics
- request_relevance: v1 4.7222 -> v2 4.7778
- explanation_groundedness: v1 4.8333 -> v2 4.7778
- explanation_usefulness: v1 4.6667 -> v2 4.6667
- response_clarity: v1 4.8333 -> v2 4.8889

## Case Comparison
### E06
- v1: intent CONSTRAINT, 1 recs, fallback=False, validation=True, status=generated
- v2: intent CONSTRAINT, 1 recs, fallback=False, validation=True, status=generated
- v1 titles: The Novices
- v2 titles: The Novices
### E11
- v1: intent CONSTRAINT, 5 recs, fallback=True, validation=False, status=fallback
- v2: intent CONSTRAINT, 5 recs, fallback=False, validation=True, status=generated
- v1 titles: Seven Samurai, Harakiri, High and Low, The Godfather Part II, The Godfather
- v2 titles: Directed by John Ford, Stop Making Sense, The Curse of the Dragon, 28 Up, As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty
### E17
- v1: intent GENERAL_DISCOVERY, 3 recs, fallback=False, validation=True, status=generated
- v2: intent GENERAL_DISCOVERY, 5 recs, fallback=False, validation=True, status=generated
- v1 titles: Harakiri, The Last of the Mohicans, The Grey Zone
- v2 titles: Seven Samurai, Harakiri, High and Low, The Godfather Part II, The Godfather
### E18
- v1: intent GENERAL_DISCOVERY, 5 recs, fallback=False, validation=True, status=generated
- v2: intent MOOD_THEME, 5 recs, fallback=True, validation=False, status=fallback
- v1 titles: Psycho, A Dog's Will, A Silent Voice: The Movie, Harakiri, High and Low
- v2 titles: Seven Samurai, Harakiri, High and Low, Directed by John Ford, The Godfather Part II

## Decision
FREEZE_RUNTIME_V2_FOR_MVP

Notes: E18 is the main residual risk because the corrected MOOD_THEME parse falls back to a generic grounded shortlist, but the critical grounding invariants remain satisfied and the documented E11/E17 issues are improved.