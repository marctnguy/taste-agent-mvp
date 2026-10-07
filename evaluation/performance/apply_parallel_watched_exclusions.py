from __future__ import annotations

from pathlib import Path


TARGET = Path("mvp/src/candidates.py")

CONSTANT_OLD = "CANDIDATE_ENRICHMENT_MAX_WORKERS = 8\n"
CONSTANT_NEW = (
    "CANDIDATE_ENRICHMENT_MAX_WORKERS = 8\n"
    "WATCHED_EXCLUSION_SEARCH_MAX_WORKERS = 8\n"
)

SIGNATURE_OLD = '''def build_watched_exclusions(\n    consumed_films: pd.DataFrame,\n    enriched_films: pd.DataFrame,\n    client: TMDBClient | None = None,\n    watched_overrides: pd.DataFrame | None = None,\n) -> tuple[pd.DataFrame, dict[str, Any]]:\n'''

SIGNATURE_NEW = '''def build_watched_exclusions(\n    consumed_films: pd.DataFrame,\n    enriched_films: pd.DataFrame,\n    client: TMDBClient | None = None,\n    watched_overrides: pd.DataFrame | None = None,\n    *,\n    search_max_workers: int | None = None,\n) -> tuple[pd.DataFrame, dict[str, Any]]:\n'''

LOOP_OLD = '''    exclusions: list[dict[str, Any]] = []\n    tmdb_matches = 0\n    title_year_fallbacks = 0\n    for _, row in watched_frame.iterrows():\n        source_id = str(row.get("source_id", ""))\n        title = row.get("title")\n        year = row.get("year") if pd.notna(row.get("year")) else row.get("release_year")\n        tmdb_id = pd.NA\n        match_method = "title_year_fallback"\n        if not isinstance(enriched_lookup, pd.DataFrame) and source_id in enriched_lookup.index:\n            pass\n        if isinstance(enriched_lookup, pd.DataFrame) and not enriched_lookup.empty and source_id in enriched_lookup.index:\n            matched = enriched_lookup.loc[source_id]\n            if isinstance(matched, pd.Series):\n                tmdb_id = matched.get("tmdb_id", pd.NA)\n            else:\n                tmdb_id = matched.iloc[0].get("tmdb_id", pd.NA)\n            if pd.notna(tmdb_id):\n                match_method = "enriched_tmdb"\n        if pd.isna(tmdb_id) and client is not None and title:\n            try:\n                result = client.search_movie(str(title), year)\n            except Exception:\n                result = None\n            if result and result.get("id") is not None:\n                tmdb_id = int(result["id"])\n                match_method = "tmdb_search"\n        if pd.notna(tmdb_id):\n            tmdb_matches += 1\n        else:\n            title_year_fallbacks += 1\n        normalized_title = _normalize_title(title)\n        normalized_title_year = f"{normalized_title}__{'' if pd.isna(year) else int(float(year))}"\n        exclusions.append(\n            {\n                "source_id": source_id,\n                "title": title,\n                "year": year,\n                "tmdb_id": int(tmdb_id) if pd.notna(tmdb_id) else pd.NA,\n                "match_method": match_method,\n                "normalized_title": normalized_title,\n                "normalized_title_year": normalized_title_year,\n                "exclusion_key": str(int(tmdb_id)) if pd.notna(tmdb_id) else normalized_title_year,\n            }\n        )\n'''

LOOP_NEW = '''    watched_rows = watched_frame.to_dict(orient="records")\n\n    def _resolve_watched_row(row: dict[str, Any]) -> dict[str, Any]:\n        source_id = str(row.get("source_id", ""))\n        title = row.get("title")\n        year = row.get("year") if pd.notna(row.get("year")) else row.get("release_year")\n        tmdb_id = pd.NA\n        match_method = "title_year_fallback"\n\n        if not enriched_lookup.empty and source_id in enriched_lookup.index:\n            matched = enriched_lookup.loc[source_id]\n            if isinstance(matched, pd.Series):\n                tmdb_id = matched.get("tmdb_id", pd.NA)\n            else:\n                tmdb_id = matched.iloc[0].get("tmdb_id", pd.NA)\n            if pd.notna(tmdb_id):\n                match_method = "enriched_tmdb"\n\n        if pd.isna(tmdb_id) and client is not None and title:\n            try:\n                result = client.search_movie(str(title), year)\n            except Exception:\n                result = None\n            if result and result.get("id") is not None:\n                tmdb_id = int(result["id"])\n                match_method = "tmdb_search"\n\n        normalized_title = _normalize_title(title)\n        normalized_title_year = f"{normalized_title}__{'' if pd.isna(year) else int(float(year))}"\n        return {\n            "source_id": source_id,\n            "title": title,\n            "year": year,\n            "tmdb_id": int(tmdb_id) if pd.notna(tmdb_id) else pd.NA,\n            "match_method": match_method,\n            "normalized_title": normalized_title,\n            "normalized_title_year": normalized_title_year,\n            "exclusion_key": str(int(tmdb_id)) if pd.notna(tmdb_id) else normalized_title_year,\n        }\n\n    worker_limit = (\n        WATCHED_EXCLUSION_SEARCH_MAX_WORKERS\n        if search_max_workers is None\n        else max(1, int(search_max_workers))\n    )\n    worker_count = min(worker_limit, len(watched_rows)) if watched_rows else 1\n    if worker_count <= 1:\n        exclusions = [_resolve_watched_row(row) for row in watched_rows]\n    else:\n        with ThreadPoolExecutor(max_workers=worker_count) as executor:\n            # executor.map preserves input ordering, so drop_duplicates(keep="first")\n            # sees the same row order as the serial implementation.\n            exclusions = list(executor.map(_resolve_watched_row, watched_rows))\n\n    tmdb_matches = sum(1 for row in exclusions if pd.notna(row.get("tmdb_id")))\n    title_year_fallbacks = len(exclusions) - tmdb_matches\n'''


def main() -> None:
    text = TARGET.read_text()

    if "WATCHED_EXCLUSION_SEARCH_MAX_WORKERS = 8" in text:
        print("Parallel watched-exclusion search patch is already applied.")
        return

    if CONSTANT_OLD not in text:
        raise RuntimeError("Expected worker constant anchor was not found; refusing to patch.")
    text = text.replace(CONSTANT_OLD, CONSTANT_NEW, 1)

    if SIGNATURE_OLD not in text:
        raise RuntimeError("Expected build_watched_exclusions signature was not found; refusing to patch.")
    text = text.replace(SIGNATURE_OLD, SIGNATURE_NEW, 1)

    if LOOP_OLD not in text:
        raise RuntimeError("Expected serial watched-exclusion loop was not found; refusing to patch.")
    text = text.replace(LOOP_OLD, LOOP_NEW, 1)

    compile(text, str(TARGET), "exec")
    TARGET.write_text(text)

    print(f"Applied bounded parallel watched-exclusion TMDB searches to {TARGET}.")
    print(
        "Concurrency: max 8 workers; input order, enriched-ID preference, TMDB search logic, "
        "fallback keys, deduplication order, and report semantics are preserved."
    )


if __name__ == "__main__":
    main()
