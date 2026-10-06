# Verification Evidence

## Offline

- `pytest -q mvp/tests/runtime_v4/test_hard_constraints.py`
  - `13 passed, 1 warning`
- `pytest -q mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py`
  - `10 passed, 1 warning`
- `pytest -q mvp/tests/watchlist_personalization/test_watchlist_personalization_integration.py`
  - `11 passed, 1 warning`

## Targeted H12 Probe

- `prepare_case(...)` for H12 produced:
  - `parsed_exclusions: []`
  - `personalization.preference: personalized discovery`
  - `full_watchlist A`: `eligible=48`, `selected=5`
  - `full_watchlist B`: `eligible=48`, `selected=5`
  - `empty_watchlist A`: `eligible=0`, `selected=0`
  - `empty_watchlist B`: `eligible=0`, `selected=0`

## Notes

- LangSmith uploads were blocked by network resolution in this environment.
- The service path itself completed locally and did not rerun the full benchmark.
