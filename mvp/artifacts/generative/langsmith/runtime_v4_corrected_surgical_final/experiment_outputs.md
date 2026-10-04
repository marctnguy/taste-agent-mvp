# Runtime V4 Surgical Final

- Experiment: `taste-agent-runtime-v4-surgical-final-75ac8c13`
- Experiment ID: `82416452-b4a8-4b73-adf8-51cea7718068`
- Experiment URL: `https://smith.langchain.com/o/3f29c78e-95c7-4ff0-864f-65906dfd0c74/datasets/5511335c-2116-4bed-9689-d3e61f4e23e7/compare?selectedSessions=82416452-b4a8-4b73-adf8-51cea7718068`
- Dataset: `taste-agent-runtime-v1-eval`
- Cases: `18`
- Trace coverage: `18/18`
- Validation pass rate: `1.00`
- LLM calls per request: mean `1.94`, median `2.00`, max `2`
- Runtime fallback used: `False`
- Abstention used: `True`

## Means
- `candidate_compliance`: `1.0`
- `candidate_identity_integrity`: `1.0`
- `duplicate_candidates`: `1.0`
- `intent_match`: `1.0`
- `hard_constraint_satisfaction`: `1.0`
- `exclusion_satisfaction`: `1.0`
- `taste_signal_grounding`: `1.0`
- `unsupported_taste_dimension`: `1.0`
- `goodreads_ranking_guardrail`: `1.0`
- `sensitive_inference_guardrail`: `1.0`
- `request_relevance`: `4.666666666666667`
- `explanation_groundedness`: `4.666666666666667`
- `explanation_usefulness`: `3.9444444444444446`
- `response_clarity`: `4.833333333333333`
- `execution_time`: `97.29697044444444`

## Notable Cases
- `E06` `Recommend me something French.` -> `generated`, llm calls `2`, request relevance `5.0`
- `E11` `Recommend something, but no drama.` -> `generated`, llm calls `2`, request relevance `5.0`
- `E17` `Which film are you recommending because of my Goodreads history?` -> `generated`, llm calls `2`, request relevance `5.0`
- `E18` `I want something weird but still emotionally engaging.` -> `generated`, llm calls `2`, request relevance `5.0`
- `E12` `Something English action from the 1950s.` -> `no_match`, llm calls `1`, request relevance `5.0`
- `E15` `Tell me why this film is a masterpiece and mention its awards.` -> `no_match`, llm calls `2`, request relevance `1.0`

## Observability
- `intent_interpretation_llm_used` was true on all 18 requests.
- `semantic_qualification_llm_used` was true on 16/18 requests.
- `selection_llm_used` and `explanation_llm_used` remained false on all requests.
- `fallback_used` remained false on all requests.
- The preserved prior corrected hosted run was not overwritten.
