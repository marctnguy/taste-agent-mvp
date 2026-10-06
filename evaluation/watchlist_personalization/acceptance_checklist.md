# Watchlist Personalization Acceptance Checklist

Status: historical wrapper/diagnostic note. Canonical evidence now lives at `evaluation/watchlist_personalization/final_evidence/README.md`.

## 1. Service extraction
- Implemented: `mvp/src/watchlist_personalization/reversible_history_watchlist_service.py`
- Implemented: `mvp/src/watchlist_personalization/langsmith_evaluation.py`
- Evidence: reusable service facade plus LangSmith suite entrypoint.
- Verified offline: `mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py`, `mvp/tests/runtime_v4/test_qualification.py`, `mvp/tests/test_contextual_retrieval.py`, `mvp/tests/runtime_v4/test_runtime_integration.py`.
- Unverified: live end-to-end behavior against TMDB/OpenAI/LangSmith.

## 2. Frozen H01-H12 dataset
- Implemented: `build_dataset_examples()` in `mvp/src/watchlist_personalization/reversible_history_watchlist_service.py`
- Implemented: dataset export in `run_reversible_history_watchlist_langsmith_suite()`
- Evidence: stable `H01`-`H12` example IDs and prompt capture.
- Unverified: hosted dataset creation and remote run IDs.

## 3. Request/spec correctness
- Implemented: still delegated to frozen reversible-history runner in `evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py`
- Evidence: `validate_controlled_requests()`, `correct_hitl_request()`, and `evaluate_single_prompt()`
- Unverified: offline gates proving no semantic drift in the new service wrapper.

## 4. Retrieval / qualification / leakage contracts
- Implemented: `run_case()` propagates candidate context, selection report, and consumed IDs into the payload.
- Implemented: `evaluation/watchlist_personalization/langsmith_evaluators.py`
- Evidence: `watched_reference_leakage`, `structured_constraint_compliance`, `output_validity`, `requested_slate_completion`, `semantic_request_fit`.
- Unverified: real TMDB and OpenAI-backed execution.

## 5. Offline behavioral gates
- Implemented: `mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py`
- Evidence: mock-based checks for service aliases, leakage evaluator, and local suite orchestration.
- Verified: 17 offline tests passed in the focused slice.

## 6. LangSmith wiring
- Implemented: `run_reversible_history_watchlist_langsmith_suite()`
- Evidence: dataset, manifest, and four experiment labels `A/full`, `B/full`, `A/empty`, `B/empty`.
- Unverified: actual LangSmith dataset/example/run IDs and uploaded experiment links.

## 7. Connectivity preflight and canary
- Present in the service: `from_environment()` fails closed on TMDB/OpenAI preflight.
- Unverified: a successful live canary in this environment.

## 8. Conclusion
- The new service is a structured wrapper plus evaluator layer around the existing reversible-history runner.
- Its offline gates now pass, but live correctness remains blocked on TMDB/OpenAI/LangSmith DNS access in this environment.
