# Quality Hardening Pass

Branch: `runtime-v4-quality-hardening`

## What Changed

- Watched exclusions now come from the full consumed-film history and are enforced again at validation time.
- Required semantic concepts must be grounded; preferred concepts only affect ranking.
- Positive and excluded concepts are normalized into disjoint sets.
- Contextual ordering now prefers required grounding, then preferred support, then request fit, then B3.
- Novelty ranking now ignores non-positive B3 scores.
- Region validation now matches the retrieval-side any-of semantics.
- Qualification now uses grounded film evidence instead of lexical coincidence alone.

## Verification

- Runtime-v4 test bundle passed: `59 passed`.
- Frozen baseline artifacts were not overwritten.
