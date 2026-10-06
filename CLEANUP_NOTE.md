# Cleanup Note

## Retained

- `mvp/src/watchlist_personalization/`
- `mvp/tests/watchlist_personalization/`
- `evaluation/watchlist_personalization/final_evidence/`
- `evaluation/watchlist_personalization/reversible_history_watchlist/` (historical cleanup evidence)
- `mvp/data/processed/watched_override_confirmations.csv`
- The accepted recommendation behavior and the frozen H01-H12 contract

## Removed Or Retired

- Duplicate export bundles under `exports/`
- Superseded hosted-final hardening outputs under `evaluation/quality_hardening/hosted_final/`
- Generated caches and obsolete LangSmith/debug run outputs
- Duplicate correction checklist: `mvp/artifacts/watchlist_personalization/recommendation_corrections_checklist.md`

## Recovery Archive

- Archive: `/private/tmp/taste-agent-cleanup-20261006-retired-outputs.zip`
- Manifest: `/private/tmp/taste-agent-cleanup-20261006-retired-outputs.manifest.json`
- SHA-256: `9b31fc95e62288c2270eab24734dcca1d4722e754f5aeefaa23c11f3138ec0ee`
- Tracked files in `HEAD`: `304`
- Tracked files still present after cleanup: `250`
- Checklist archive: `/private/tmp/taste-agent-cleanup-20261006-checklist.zip`
- Checklist archive SHA-256: `b7df179257d5a3534f0715dc4b9e8d6b5f5b714c74a8538102ccef363ddea34c`

## Recovery Copies

- Folder: `/Users/marc.tanguy/Desktop/IRONHACK/Capstone/Cleanup recovery/`
- Retired outputs ZIP and manifest: `taste-agent-cleanup-20261006-retired-outputs.zip`, `taste-agent-cleanup-20261006-retired-outputs.manifest.json`
- Checklist ZIP and manifest: `taste-agent-cleanup-20261006-checklist.zip`, `taste-agent-cleanup-20261006-checklist.manifest.json`

## Canonical Evidence

- Canonical evidence README: `evaluation/watchlist_personalization/final_evidence/README.md`
- Historical checklist/status docs: `evaluation/watchlist_personalization/acceptance_checklist.md`, `evaluation/watchlist_personalization/validity_report.md`

## Restoration

- Unzip the archive into a clean checkout if a retired file needs to be recovered.
- The archive preserves original relative paths and SHA-256 hashes.
- Git history was not rewritten.
