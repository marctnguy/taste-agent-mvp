# runtime_v4 Corrected Success Criteria

These criteria are frozen before the corrected hosted regression is rerun.

## Structural

- 11/11 cases complete.
- No execution errors.
- No watched candidate selected.
- No unsupported candidate selected.
- No duplicate candidates returned.
- No Goodreads ranking influence.
- No sensitive inference.
- No explanation claim without traceable evidence.
- Runtime LLM calls are reported explicitly and are non-zero when the hosted semantic parser / qualifier are active.

## Semantic Critical Cases

- **V01 negative constraint**: no recommendation may violate the explicit exclusions for performers, fame, or show business.
- **V03 abstention**: if no grounded request match exists, abstention or fewer results is valid.
- **V06 / H11 show-business**: any recommendation must have grounded subject/content evidence for performers, fame, or show business, not associative coincidence.
- **Resolved reference**: reference context must influence retrieval and qualification through grounded reference metadata.
- **Unresolved reference**: the runtime must not invent facts about an unresolved title.
- **Novelty**: novelty behavior must be supported by the history-distance diagnostic, not by low B3 alone.

## Directional Quality Gates

- Request relevance should improve materially from the original baseline.
- Explanation usefulness should improve materially from the original baseline.
- Hard constraint and exclusion violations should be zero on applicable cases.
- `intent_match` should be meaningful after the evaluator field-path fix.

## Constraint Discipline

- Do not declare success based only on averages.
- Do not declare success based only on one or two improved cases.
- Do not tune after seeing the hosted corrected outputs.
