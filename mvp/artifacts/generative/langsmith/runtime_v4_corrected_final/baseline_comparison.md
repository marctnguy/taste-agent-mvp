# Baseline Comparison

## Summary
- Old experiment: `taste-agent-runtime-v4-hosted-1ef4feed`
- New experiment: `taste-agent-runtime-v4-corrected-bcf79a1f`
- Old trace coverage: `11/11`
- New trace coverage: `11/11`
- Old intent match: `0.00`
- New intent match: `1.00`
- Old request relevance: `2.82`
- New request relevance: `3.55`
- Old explanation usefulness: `3.09`
- New explanation usefulness: `3.55`
- Old response clarity: `3.91`
- New response clarity: `4.55`
- Old runtime validation failures: `0`
- New runtime validation failures: `0`

## What changed
- Canonical intent labels are normalized before schema validation.
- Sparse semantic evidence no longer aborts explanation generation.
- Decade constraints are validated as an allowed set rather than an impossible conjunction.
- Hosted evaluation now completes with actual runtime LLM calls (`llm_call_count = 2` per case).

## Notes
- The earlier hosted export remains preserved in `runtime_v4_hosted_final/` as the pre-fix baseline.
- The corrected run is stored in `runtime_v4_corrected_final/`.