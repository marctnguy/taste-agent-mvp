from __future__ import annotations

from pathlib import Path


CATALOG_TARGET = Path("mvp/src/retrieval/catalog_retrieval.py")
SERVICE_TARGET = Path(
    "mvp/src/watchlist_personalization/reversible_history_watchlist_service.py"
)

DATACLASS_ANCHOR = '''@dataclass(frozen=True)\nclass CatalogRetrievalResult:\n    request: RequestUnderstanding\n    catalog_frame: pd.DataFrame\n    documents: pd.DataFrame\n    request_embedding: np.ndarray | None\n    retrieval_diagnostics: CatalogRetrievalDiagnostics\n    augmentation_report: dict[str, Any]\n\n\n'''

DATACLASS_REPLACEMENT = DATACLASS_ANCHOR + '''@dataclass(frozen=True)\nclass QueryAwareAugmentation:\n    frame: pd.DataFrame\n    report: dict[str, Any]\n\n\n'''

HELPER_ANCHOR = '''\ndef _filter_hard_constraints(frame: pd.DataFrame, request: RequestUnderstanding) -> pd.DataFrame:\n'''

HELPER = '''\ndef build_query_aware_augmentation(\n    request: RequestUnderstanding,\n    client: TMDBClient | None,\n) -> QueryAwareAugmentation:\n    \"\"\"Build request-only catalog augmentation once for reuse across route pools.\n\n    Reference, structured-constraint, and semantic-keyword augmentation depend only\n    on the parsed request and TMDB client, not on the base candidate pool. Keeping\n    this result immutable-by-convention lets request/history routes reuse the same\n    external lookup result while preserving their independent base pools.\n    \"\"\"\n    augmentation_frames: list[pd.DataFrame] = []\n    augmentation_report: dict[str, Any] = {\"query_aware_augmentation_used\": False}\n\n    if _request_has_query_aware_signal(request):\n        ref_frame, ref_report = _augment_with_reference_candidates(request, client)\n        if not ref_frame.empty:\n            augmentation_frames.append(ref_frame)\n            augmentation_report.update(ref_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n\n        constraint_frame, constraint_report = _augment_with_constraint_candidates(\n            request, client\n        )\n        if not constraint_frame.empty:\n            augmentation_frames.append(constraint_frame)\n            augmentation_report.update(constraint_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n\n        keyword_frame, keyword_report = _augment_with_semantic_keyword_candidates(\n            request, client\n        )\n        if not keyword_frame.empty:\n            augmentation_frames.append(keyword_frame)\n            augmentation_report.update(keyword_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n\n    if augmentation_frames:\n        frame = pd.concat(augmentation_frames, ignore_index=True)\n        if \"source_id\" in frame.columns:\n            frame[\"source_id\"] = frame[\"source_id\"].astype(str)\n    else:\n        frame = pd.DataFrame(columns=[\"source_id\"])\n\n    return QueryAwareAugmentation(frame=frame, report=augmentation_report)\n\n\n'''

SIGNATURE_OLD = '''def discover_catalog_for_request(\n    request: RequestUnderstanding,\n    *,\n    candidate_limit: int = 300,\n    candidate_context_size: int = 50,\n    candidate_pool: pd.DataFrame | None = None,\n    watched_ids: Iterable[str] | None = None,\n    client: TMDBClient | None = None,\n) -> CatalogRetrievalResult:\n'''

SIGNATURE_NEW = '''def discover_catalog_for_request(\n    request: RequestUnderstanding,\n    *,\n    candidate_limit: int = 300,\n    candidate_context_size: int = 50,\n    candidate_pool: pd.DataFrame | None = None,\n    watched_ids: Iterable[str] | None = None,\n    client: TMDBClient | None = None,\n    precomputed_augmentation: QueryAwareAugmentation | None = None,\n) -> CatalogRetrievalResult:\n'''

AUGMENT_OLD = '''    augmentation_frames: list[pd.DataFrame] = []\n    augmentation_report: dict[str, Any] = {\"query_aware_augmentation_used\": False}\n    spec = _request_spec(request)\n    if _request_has_query_aware_signal(request):\n        ref_frame, ref_report = _augment_with_reference_candidates(request, client)\n        if not ref_frame.empty:\n            augmentation_frames.append(ref_frame)\n            augmentation_report.update(ref_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n        constraint_frame, constraint_report = _augment_with_constraint_candidates(request, client)\n        if not constraint_frame.empty:\n            augmentation_frames.append(constraint_frame)\n            augmentation_report.update(constraint_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n        keyword_frame, keyword_report = _augment_with_semantic_keyword_candidates(request, client)\n        if not keyword_frame.empty:\n            augmentation_frames.append(keyword_frame)\n            augmentation_report.update(keyword_report)\n            augmentation_report[\"query_aware_augmentation_used\"] = True\n\n    if augmentation_frames:\n        candidate_pool = pd.concat([candidate_pool, *augmentation_frames], ignore_index=True)\n        candidate_pool[\"source_id\"] = candidate_pool[\"source_id\"].astype(str)\n'''

AUGMENT_NEW = '''    augmentation = (\n        precomputed_augmentation\n        if precomputed_augmentation is not None\n        else build_query_aware_augmentation(request, client)\n    )\n    augmentation_report = dict(augmentation.report)\n    augmentation_frame = augmentation.frame.copy()\n    if not augmentation_frame.empty:\n        candidate_pool = pd.concat(\n            [candidate_pool, augmentation_frame], ignore_index=True\n        )\n        candidate_pool[\"source_id\"] = candidate_pool[\"source_id\"].astype(str)\n'''

SERVICE_IMPORT_OLD = '''from mvp.src.retrieval.catalog_retrieval import discover_catalog_for_request, embed_text\n'''
SERVICE_IMPORT_NEW = '''from mvp.src.retrieval.catalog_retrieval import (\n    build_query_aware_augmentation,\n    discover_catalog_for_request,\n    embed_text,\n)\n'''

SERVICE_CALLS_OLD = '''        request_result = discover_catalog_for_request(request, client=self.tmdb_client, candidate_pool=request_pool, watched_ids=watched_keys)\n        history_result = discover_catalog_for_request(request, client=self.tmdb_client, candidate_pool=history_pool, watched_ids=watched_keys)\n'''

SERVICE_CALLS_NEW = '''        shared_augmentation = build_query_aware_augmentation(\n            request, self.tmdb_client\n        )\n        request_result = discover_catalog_for_request(\n            request,\n            client=self.tmdb_client,\n            candidate_pool=request_pool,\n            watched_ids=watched_keys,\n            precomputed_augmentation=shared_augmentation,\n        )\n        history_result = discover_catalog_for_request(\n            request,\n            client=self.tmdb_client,\n            candidate_pool=history_pool,\n            watched_ids=watched_keys,\n            precomputed_augmentation=shared_augmentation,\n        )\n'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected {label} anchor was not found; refusing to patch.")
    return text.replace(old, new, 1)


def main() -> None:
    catalog = CATALOG_TARGET.read_text()
    service = SERVICE_TARGET.read_text()

    if "precomputed_augmentation: QueryAwareAugmentation" in catalog:
        print("Shared query augmentation patch is already applied.")
        return

    catalog = _replace_once(
        catalog, DATACLASS_ANCHOR, DATACLASS_REPLACEMENT, "catalog dataclass"
    )
    catalog = _replace_once(
        catalog, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "augmentation helper"
    )
    catalog = _replace_once(
        catalog, SIGNATURE_OLD, SIGNATURE_NEW, "discover signature"
    )
    catalog = _replace_once(
        catalog, AUGMENT_OLD, AUGMENT_NEW, "query augmentation block"
    )

    service = _replace_once(
        service, SERVICE_IMPORT_OLD, SERVICE_IMPORT_NEW, "service retrieval import"
    )
    service = _replace_once(
        service, SERVICE_CALLS_OLD, SERVICE_CALLS_NEW, "service retrieval calls"
    )

    compile(catalog, str(CATALOG_TARGET), "exec")
    compile(service, str(SERVICE_TARGET), "exec")
    CATALOG_TARGET.write_text(catalog)
    SERVICE_TARGET.write_text(service)

    print("Applied shared query-aware augmentation across request/history routes.")
    print(
        "Semantics preserved: each route keeps its own base pool, filtering, "
        "embedding, relevance scoring, and route membership; only request-only "
        "TMDB augmentation is fetched once and reused."
    )


if __name__ == "__main__":
    main()
