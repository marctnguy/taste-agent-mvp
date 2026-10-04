# Runtime V4 LangSmith Debug

## Root cause

- The sandboxed runner could not resolve `api.smith.langchain.com`, so the earlier hosted attempt could not rely on the real LangSmith API.
- The runtime_v4 hosted path also exposed two real data-shape issues on augmented candidates:
  - missing `predicted_preference` before request scoring and qualification sorting
  - missing `raw_rank` before `RetrievedCandidate` validation
- A repeated-example upload to the frozen LangSmith dataset returned a 409 conflict, so the runner had to treat duplicate example IDs as an existing frozen dataset rather than a fallback condition.

## Fixes

- Added a hosted `runtime_v4` LangSmith runner backed by `evaluation/langsmith/runtime_v4_dataset.json`.
- Kept package imports lazy to avoid circular imports between `generative`, `generative_v4`, and `retrieval`.
- Normalized augmented TMDB candidates so `predicted_preference`, `rank`, and `raw_rank` are always available.
- Treated `example already exists` / `409 Conflict` as a frozen-dataset reuse case.
- Moved the `semantic_retrieval` trace to the runtime layer so it appears consistently in tests and real runs.

## Hosted result

- Experiment: `taste-agent-runtime-v4-hosted-1ef4feed`
- Experiment ID: `469c8480-531b-4ba1-8293-62d85c4a9a7d`
- URL: `https://smith.langchain.com/o/3f29c78e-95c7-4ff0-864f-65906dfd0c74/datasets/86d55dca-57b7-4069-9df8-9319afb925f4/compare?selectedSessions=469c8480-531b-4ba1-8293-62d85c4a9a7d`
- Trace coverage: `11/11`
- Runtime fallback used: `false`
- Runtime repairs attempted: `false`
- Runtime validation failures: `0`

## Artifact locations

- `mvp/artifacts/generative/langsmith/runtime_v4_hosted_final/experiment_summary.json`
- `mvp/artifacts/generative/langsmith/runtime_v4_hosted_final/experiment_results.csv`
- `mvp/artifacts/generative/langsmith/runtime_v4_hosted_final/runtime_v4_manifest.json`

## Observed metrics

- `candidate_compliance`: `1.0`
- `candidate_identity_integrity`: `1.0`
- `duplicate_candidates`: `1.0`
- `hard_constraint_satisfaction`: `0.7272727272727273`
- `exclusion_satisfaction`: `0.8181818181818182`
- `taste_signal_grounding`: `1.0`
- `unsupported_taste_dimension`: `1.0`
- `goodreads_ranking_guardrail`: `1.0`
- `sensitive_inference_guardrail`: `1.0`
- `request_relevance` mean: `2.8181818181818183`
- `explanation_groundedness` mean: `4.2727272727272725`
- `explanation_usefulness` mean: `3.090909090909091`
- `response_clarity` mean: `3.909090909090909`
- `execution_time` mean: `70.2385020909091`

## Runtime notes

- Hosted evaluation completed with actual LangSmith connectivity.
- The runtime itself reported `llm_call_count = 0` and `fallback_used = false` for all 11 cases.
