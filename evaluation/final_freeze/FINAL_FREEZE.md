# Final Freeze

Official hosted LangSmith experiment:

- `taste-agent-runtime-v4-final-freeze-corrected-official-8c94d8ee`
- `bc736d85-d88d-46bf-a90a-a5cb01fe037d`
- `https://smith.langchain.com/o/3f29c78e-95c7-4ff0-864f-65906dfd0c74/datasets/89126411-8420-4fd4-abb7-cd8344de178f/compare?selectedSessions=bc736d85-d88d-46bf-a90a-a5cb01fe037d`

## Frozen Caseset

Exact caseset: `H01-H12 + D + E13-E17`

Coverage:

- `18/18` hosted traces
- `H04` protected contextual/reference behavior preserved
- `H01`, `H03`, `H05`, `H07` no longer collapse to the same slate
- `H10` safely abstains instead of violating old/classic intent
- `H11` no longer exhibits the earlier fame/show-business rationalization failure
- `H12` has no false reference
- B3 coverage: `51/51` selected recommendations with finite scores
- Placeholder B3 zero count: `0`
- Goodreads remains descriptive-only
- `B3`, `PCA`, `Ridge`, and taxonomy unchanged

## Classified Outcomes

### E15
`VALIDATOR_APPLICABILITY_ARTIFACT`

The `EXPLAIN_TARGET` interaction behaved correctly:

- remained on candidate `346`
- did not generate another slate
- refused unsupported awards/masterpiece claims

The failure came from a recommendation-style `all_of` validation being applied to an explanation interaction.

### H09
`REGION_CONSTRAINT_VALIDATOR_MISMATCH / MVP LIMITATION`

The runtime maps "European" to the current any-of region proxy:

- `DE`
- `ES`
- `FR`
- `GB`
- `IT`

The selected films each satisfied that configured region proxy, but the validator treated the list as jointly required and raised structured-constraint errors. The Europe mapping remains incomplete in this MVP, and the runtime is not being changed here.

### Case D
`NON_BLOCKING_DIAGNOSTIC_LIMITATION`

The system safely abstains and enforces exclusions. The internal phrase `high-B3 compatible` is still partly represented semantically rather than purely as a personalization instruction, but this is not representative end-user language and does not justify reopening the runtime.

## Preserved Metrics

- Request relevance: `4.11/5`
- Explanation groundedness: `4.44/5`
- Explanation usefulness: `3.56/5`
- Response clarity: `4.56/5`

## Latency Limitation

- Mean: `75.11 s`
- Median: `96.46 s`
- P95: `115.35 s`

## Closure Evidence

- Exact `H01-H12 + D + E13-E17` caseset
- `18/18` hosted traces
- B3 coverage `51/51` finite scores
- Placeholder B3 zero count `0`
- Goodreads descriptive-only
- No runtime/model file changes in this documentation pass

