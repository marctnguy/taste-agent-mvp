# Verification Evidence

## Archive And Footprint

- Recovery archive: `/private/tmp/taste-agent-cleanup-20261006-retired-outputs.zip`
- Recovery archive SHA-256: `9b31fc95e62288c2270eab24734dcca1d4722e754f5aeefaa23c11f3138ec0ee`
- Tracked files in `HEAD`: `304`
- Tracked files still present after cleanup: `250`

## Offline Checks

- `pytest -q mvp/tests/runtime_v4/test_hard_constraints.py`
  - `13 passed, 1 warning`
- `pytest -q mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py`
  - `10 passed, 1 warning`
- `pytest -q mvp/tests/watchlist_personalization/test_langsmith_evaluation_runner.py`
  - `1 passed, 1 warning`
- Bootstrap import check:
  - `python -c "import mvp.src.watchlist_personalization.langsmith_evaluation as r; print(r.DEFAULT_RESULTS_DIR); print(r.DEFAULT_CHECKPOINT_FILENAME); print(r.DEFAULT_HUMAN_REVIEW_FILENAME)"`
  - Loaded successfully without a network call

## H12 Probe

- Offline targeted H12 preparation remained valid.
- Full watchlist: `48` eligible, `5` selected.
- Empty watchlist: `0` eligible, `0` selected.
- Parsed request preserved `personalized discovery` and did not turn H12 into a film exclusion.

## Limitation

- Live TMDB/OpenAI/LangSmith verification was not rerun during this cleanup.
