# Runtime Contract

- Default variant: `A`
- Supported slates: `full_watchlist`, `empty_watchlist`, `sparse_watchlist`
- Main entrypoint: `mvp.src.watchlist_personalization.reversible_history_watchlist_service.load_reversible_history_watchlist_service()`
- Recommendation API: `service.recommend(...)` and `service.prepare_case(...)`
- Benchmark wrapper: `python -m mvp.src.watchlist_personalization.langsmith_evaluation`

## Inputs

- Personal taste profile from the frozen training population.
- Watched history from `mvp/data/raw/taste-agent-combined-ingestion.csv`.
- Manual watched overrides from `mvp/data/processed/watched_override_confirmations.csv`.
- Watchlist from the local Letterboxd export ZIP.

## Watched State

- Exclusions are driven by TMDB IDs, canonical `title__year` keys, and source IDs where applicable.
- Unrated watched films remain excluded.
- Pending clarification rows are not imported as watched.

## Sequel Policy

- `Project A - Part II (1987)` is blocked until `Project A (1983)` is watched.
- `Home Alone 3 (1997)` is blocked until `Home Alone 2: Lost in New York (1992)` is watched.
- `Toy Story 5 (2026)` remains eligible when the prior Toy Story films are confirmed watched.

## Cache Invalidation

- Prepared-case caching includes the watched-state fingerprint.
- Watched-state changes refilter cached selections instead of reusing stale results.

## Known Limits

- `Vagabond (1985)` remains pending clarification.
- The empty-watchlist path can still legitimately return no slate.
- Live LangSmith uploads were unavailable in this environment; local service preparation succeeded.
