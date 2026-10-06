# Runtime Contract

## Purpose

This pack freezes the accepted watchlist personalization runtime and its H01-H12 comparison contract.

## Entry Points

- Service: `mvp.src.watchlist_personalization.reversible_history_watchlist_service.load_reversible_history_watchlist_service()`
- CLI runner: `python -m mvp.src.watchlist_personalization.langsmith_evaluation`
- Offline experiment: `python evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py`

## Behavioral Contract

- Variant A remains the default.
- H12 must be interpreted as personalized discovery, not as a film-content exclusion.
- Watched titles must be excluded, including unrated history and explicit override confirmations.
- Sequels must be blocked when earlier installments were not watched, but allowed when prerequisites are satisfied.
- Offline verification must not require live TMDB/OpenAI/LangSmith calls.

## Required Data

- `mvp/data/processed/condition_a_enriched.csv`
- `mvp/data/processed/primary_holdout_split.csv`
- `mvp/data/processed/watched_override_confirmations.csv`
- `mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv`
- `evaluation/hitl/runtime_v4_final/05_run_manifest.json`
- External Letterboxd watchlist ZIP in the repository parent directory

## Canonical Evidence

- Summary: `reversible_history_watchlist_summary.json`
- Technical report: `reversible_history_watchlist_technical_report.md`
- Human scoring: `../human_scoring/`
