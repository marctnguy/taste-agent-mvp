from __future__ import annotations

from pathlib import Path


TARGET = Path("mvp/src/candidates.py")

IMPORT_OLD = "from __future__ import annotations\n\nimport json\n"
IMPORT_NEW = (
    "from __future__ import annotations\n\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "import json\n"
)

CONSTANT_OLD = (
    "DEFAULT_CANDIDATE_LIMIT = 300\n"
    "DEFAULT_TOP_K = 20\n"
)
CONSTANT_NEW = (
    "DEFAULT_CANDIDATE_LIMIT = 300\n"
    "DEFAULT_TOP_K = 20\n"
    "CANDIDATE_ENRICHMENT_MAX_WORKERS = 8\n"
)

INSERT_ANCHOR = "\ndef generate_candidate_pool(\n"

HELPERS = r'''

def _enrich_candidate_record(
    client: TMDBClient,
    row: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Enrich one already-selected candidate without changing candidate identity/order."""
    movie_payload = {
        "id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
        "title": row.get("title"),
    }
    if pd.notna(row.get("year")):
        movie_payload["release_date"] = f"{int(float(row['year']))}-01-01"

    enriched = _enrich_movie_details(client, movie_payload) or {}
    failed = not bool(enriched)
    if failed:
        enriched = {
            "source_id": str(row.get("source_id")),
            "tmdb_id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else pd.NA,
            "canonical_id": str(row.get("source_id")),
            "title": row.get("title"),
            "release_year": row.get("release_year"),
            "year": row.get("year"),
            "tmdb_genres": pd.NA,
            "tmdb_original_language": pd.NA,
            "tmdb_production_countries": pd.NA,
            "tmdb_overview": pd.NA,
        }
    else:
        for field in [
            "tmdb_genres",
            "tmdb_original_language",
            "tmdb_production_countries",
            "tmdb_overview",
        ]:
            if field not in enriched:
                enriched[field] = pd.NA

    enriched["candidate_sources"] = _unique_list(
        [str(value) for value in row.get("candidate_sources", [])]
    )
    enriched["candidate_source_ranks"] = _unique_list(
        [str(value) for value in row.get("candidate_source_ranks", [])]
    )
    return enriched, failed


def _enrich_candidate_records(
    client: TMDBClient,
    rows: list[dict[str, Any]],
    *,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Enrich independent TMDB candidates concurrently while preserving input order."""
    if not rows:
        return [], 0

    worker_limit = (
        CANDIDATE_ENRICHMENT_MAX_WORKERS
        if max_workers is None
        else max(1, int(max_workers))
    )
    worker_count = min(worker_limit, len(rows))

    if worker_count <= 1:
        results = [_enrich_candidate_record(client, row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map returns results in the same order as the input rows.
            # Candidate ordering and downstream ranking inputs are therefore unchanged.
            results = list(
                executor.map(
                    lambda row: _enrich_candidate_record(client, row),
                    rows,
                )
            )

    enriched_rows = [enriched for enriched, _failed in results]
    enrichment_failures = sum(1 for _enriched, failed in results if failed)
    return enriched_rows, enrichment_failures
'''

LOOP_OLD = r'''    enriched_rows: list[dict[str, Any]] = []
    enrichment_failures = 0
    for row in candidate_frame.to_dict(orient="records"):
        movie_payload = {
            "id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
            "title": row.get("title"),
        }
        if pd.notna(row.get("year")):
            movie_payload["release_date"] = f"{int(float(row['year']))}-01-01"
        enriched = _enrich_movie_details(client, movie_payload) or {}
        if not enriched:
            enrichment_failures += 1
            enriched = {
                "source_id": str(row.get("source_id")),
                "tmdb_id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else pd.NA,
                "canonical_id": str(row.get("source_id")),
                "title": row.get("title"),
                "release_year": row.get("release_year"),
                "year": row.get("year"),
                "tmdb_genres": pd.NA,
                "tmdb_original_language": pd.NA,
                "tmdb_production_countries": pd.NA,
                "tmdb_overview": pd.NA,
            }
        else:
            for field in ["tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"]:
                if field not in enriched:
                    enriched[field] = pd.NA
        enriched["candidate_sources"] = _unique_list([str(value) for value in row.get("candidate_sources", [])])
        enriched["candidate_source_ranks"] = _unique_list([str(value) for value in row.get("candidate_source_ranks", [])])
        enriched_rows.append(enriched)
'''

LOOP_NEW = r'''    enriched_rows, enrichment_failures = _enrich_candidate_records(
        client,
        candidate_frame.to_dict(orient="records"),
    )
'''


def main() -> None:
    text = TARGET.read_text()

    if "CANDIDATE_ENRICHMENT_MAX_WORKERS = 8" in text:
        print("Parallel candidate enrichment patch is already applied.")
        return

    if IMPORT_OLD not in text:
        raise RuntimeError("Expected import anchor was not found; refusing to patch.")
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)

    if CONSTANT_OLD not in text:
        raise RuntimeError("Expected candidate constants were not found; refusing to patch.")
    text = text.replace(CONSTANT_OLD, CONSTANT_NEW, 1)

    insert_at = text.find(INSERT_ANCHOR)
    if insert_at < 0:
        raise RuntimeError("generate_candidate_pool anchor was not found; refusing to patch.")
    text = text[:insert_at] + HELPERS + text[insert_at:]

    if LOOP_OLD not in text:
        raise RuntimeError("Expected serial enrichment loop was not found; refusing to patch.")
    text = text.replace(LOOP_OLD, LOOP_NEW, 1)

    compile(text, str(TARGET), "exec")
    TARGET.write_text(text)

    print(f"Applied bounded parallel candidate enrichment to {TARGET}.")
    print(
        "Concurrency: max 8 workers; candidate universe, ordering, TMDB fields, "
        "fallback behavior, and downstream ranking inputs unchanged."
    )


if __name__ == "__main__":
    main()
