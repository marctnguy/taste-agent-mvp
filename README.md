# Taste Agent

Taste Agent is a movie recommendation capstone with a frozen watchlist personalization path and a separate evaluation harness.

## Active Architecture

- Service entrypoint: `mvp.src.watchlist_personalization.reversible_history_watchlist_service.load_reversible_history_watchlist_service()`
- CLI / evaluation runner: `python -m mvp.src.watchlist_personalization.langsmith_evaluation`
- Frozen offline runner: `python evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py`
- Future UI should call the service facade and its `recommend(...)` method, not the evaluation runner.

## Required Inputs

- `mvp/data/processed/condition_a_enriched.csv`
- `mvp/data/processed/primary_holdout_split.csv`
- `mvp/data/processed/watched_override_confirmations.csv`
- `mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv`
- `evaluation/hitl/runtime_v4_final/05_run_manifest.json`
- The Letterboxd watchlist ZIP in the repository parent directory when running the watchlist service.

## Setup And Test

- Focused offline checks: `pytest -q mvp/tests/runtime_v4/test_hard_constraints.py`
- Watchlist service checks: `pytest -q mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py`
- Runner checks: `pytest -q mvp/tests/watchlist_personalization/test_langsmith_evaluation_runner.py`
- Offline check with service preflight: `python evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py`

## Evidence

- Canonical evidence pack: `evaluation/watchlist_personalization/final_evidence/README.md`
- Hosted comparison: `evaluation/watchlist_personalization/final_evidence/hosted/`
- Completed human review: `evaluation/watchlist_personalization/final_evidence/human_review/`
- Runtime freeze evidence: `evaluation/watchlist_personalization/final_evidence/runtime_freeze/`
- Historical cleanup evidence: `evaluation/watchlist_personalization/reversible_history_watchlist/artifacts/`

## Known Limits

- This cleanup does not claim a fresh live TMDB/OpenAI/LangSmith run.
- The watchlist service still depends on the external Letterboxd ZIP input.
- Variant A remains the default recommendation policy.
