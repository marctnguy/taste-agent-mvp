# Exclusion D Audit

- Case: `V01`
- Prompt: `Recommend something that is very high-B3 compatible, but it must not actually be about performers, fame, or show business.`
- Verdict: `EXCLUSION_D_REAL_VIOLATION`

## Grounded Evidence
- Selected titles from the saved smoke row: `Chris Brown: Welcome to My Life`, `Michael`, `Stop Making Sense`, `Justin Bieber's Believe`, `I Am: Celine Dion`
- The saved response text explicitly described these as grounded selections tied to fame / performers / show business, which is exactly the excluded subject matter.
- No evidence in the saved smoke artifact supports an evaluator mismatch explanation; the selected works themselves are about the excluded concepts.

## Per-candidate Classification
- Chris Brown: Welcome to My Life: VIOLATION
- Michael: VIOLATION
- Stop Making Sense: VIOLATION
- Justin Bieber's Believe: VIOLATION
- I Am: Celine Dion: VIOLATION