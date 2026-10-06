# Watchlist Personalization Validity Report

Status: historical invalid-attempt report. Canonical evidence now lives at `evaluation/watchlist_personalization/final_evidence/README.md`.

This run is a diagnostic attempt only. Do not use it for human scoring or ranking interpretation.

## Why it is invalid

- Credentials are present in the project environment.
- TMDB connectivity from this execution environment failed with DNS resolution error.
- OpenAI connectivity from this execution environment failed with DNS resolution error.
- The uploaded attempt used local-hashing embeddings, zero qualification LLM calls, and an offline fallback candidate pool.
- Watchlist resolution left many titles unresolved, so the observed coverage is not representative of the intended live comparison.

## Current environment checks

- `TMDB_API_KEY`: present
- `OPENAI_API_KEY`: present
- `LANGSMITH_API_KEY`: present
- `LANGSMITH_ENDPOINT`: present
- TMDB live request: reachable and returns a valid configuration response
- OpenAI live request: reachable, but embedding request fails with `429` / `credit_balance_exhausted`

## Latest rerun after key replacement

- TMDB authenticated request succeeded.
- OpenAI authenticated embedding request reached the API, but the API returned:
  - `429 Too Many Requests`
  - `type: insufficient_quota`
  - `code: credit_balance_exhausted`
- This means the current blocker is account/project credits, not local DNS or missing environment variables.

## Existing invalid-attempt evidence

- Embedding provider: `local-hashing-1536`
- Qualification LLM calls: `0`
- Offline fallback pool used: yes
- Shared union size: 18
- Watchlist candidates resolved: 7
- Unresolved watchlist lookups: 404

## Do not score

Do not ask for human scoring, and do not interpret the produced slate order as a valid experiment result until the live TMDB/OpenAI prerequisites pass.

## Terminal command for a normal shell

```bash
cd /Users/marc.tanguy/Desktop/IRONHACK/Capstone/taste-agent
python evaluation/watchlist_personalization/run_isolated_experiment.py
```

## Required environment setup

Use the authorized project `.env` mechanism with at least:

- `TMDB_API_KEY`
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`

The harness reads `.env` from the repo root via the project config loader.
