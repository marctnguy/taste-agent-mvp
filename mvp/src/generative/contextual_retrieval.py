from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mvp.src.config import get_api_keys
from mvp.src.generative.context_builder import (
    DEFAULT_CANDIDATE_CONTEXT_SIZE,
    DEFAULT_RECOMMENDATION_COUNT,
    DEFAULT_RUN_DIR,
    RecommendationRuntimeContext,
    _candidate_evidence,
    _eligible_taste_profile,
    _load_candidate_frame,
    _load_candidate_semantics,
    _load_taste_profile,
    _parse_listish,
    _to_signals,
    parse_intent,
)
from mvp.src.generative.schemas import CandidateContextItem


DEFAULT_CANDIDATE_EMBEDDINGS_PATH = Path("mvp/artifacts/models/b3_mvp/candidate_embedding_cache/candidate_embeddings.csv")
DEFAULT_CANDIDATE_SEMANTIC_PATH = Path("mvp/artifacts/diagnostics/comprehensive_discovery_audit/candidate_semantic_vectors.csv")
DEFAULT_TASTE_PROFILE_PATH = Path("mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv")
RUNTIME_V3_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RRF_K = 60


@dataclass(frozen=True)
class RuntimeV3RetrievalDiagnostics:
    candidate_universe_count: int
    post_constraint_candidate_count: int
    contextual_retrieval_used: bool
    candidate_context_size: int
    shortlist_size: int
    retrieval_mode: str
    context_rank_map: dict[str, int | None]
    b3_rank_map: dict[str, int]
    rrf_rank_map: dict[str, int | None]
    final_shortlist_rank_map: dict[str, int]


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _openai_post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _vector_columns(frame: pd.DataFrame, prefix: str = "emb_") -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _load_embedding_cache(path: str | Path = DEFAULT_CANDIDATE_EMBEDDINGS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _embed_text(text: str, api_key: str, model: str = RUNTIME_V3_EMBEDDING_MODEL) -> np.ndarray:
    response = _openai_post_json(
        "https://api.openai.com/v1/embeddings",
        api_key,
        {"model": model, "input": [text]},
        timeout=180,
    )
    data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
    if not data:
        raise RuntimeError("OpenAI embedding response was empty.")
    vector = np.asarray(data[0]["embedding"], dtype=float)
    return _l2_normalize_vector(vector)


def _candidate_passes_hard_filters(row: pd.Series, intent) -> bool:
    if intent.requested_languages:
        language = str(row.get("tmdb_original_language") or "").strip().lower()
        if language and language not in {value.lower() for value in intent.requested_languages}:
            return False
    if intent.requested_countries:
        countries = {part.lower() for part in _parse_listish(row.get("tmdb_production_countries"))}
        if not countries.intersection({value.lower() for value in intent.requested_countries}):
            return False
    if intent.requested_decades and pd.notna(row.get("year")):
        decade = f"{int(row['year']) // 10 * 10}s"
        if decade not in intent.requested_decades:
            return False
    if intent.requested_genres:
        genres = {part.lower() for part in _parse_listish(row.get("tmdb_genres"))}
        if not genres.intersection({value.lower() for value in intent.requested_genres}):
            return False
    if intent.exclusions:
        haystack = " ".join(
            [
                str(row.get("title") or "").lower(),
                " ".join(_parse_listish(row.get("tmdb_genres"))).lower(),
                str(row.get("tmdb_original_language") or "").lower(),
                " ".join(_parse_listish(row.get("tmdb_production_countries"))).lower(),
                " ".join(_parse_listish(row.get("candidate_sources"))).lower(),
                str(row.get("tmdb_overview") or "").lower(),
            ]
        )
        if any(str(exclusion).lower() in haystack for exclusion in intent.exclusions):
            return False
    return True


def _has_meaningful_context(query: str, intent) -> bool:
    if intent.intent_type == "EXPLAIN":
        return False
    if intent.requested_taste_dimensions:
        return True
    tokens = re.findall(r"[a-z0-9']+", query.lower())
    return len(tokens) >= 7


def _frame_to_candidate_item(
    row: pd.Series,
    taste_profile: pd.DataFrame,
    semantic_lookup: pd.DataFrame,
) -> CandidateContextItem:
    candidate_id = str(row["source_id"])
    semantic_row = semantic_lookup.loc[candidate_id] if candidate_id in semantic_lookup.index else None
    evidence = _candidate_evidence(row, taste_profile, semantic_row)
    return CandidateContextItem(
        candidate_id=candidate_id,
        tmdb_id=int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
        title=str(row["title"]),
        year=int(row["year"]) if pd.notna(row.get("year")) else None,
        genres=_parse_listish(row.get("tmdb_genres")),
        original_language=str(row["tmdb_original_language"]) if pd.notna(row.get("tmdb_original_language")) else None,
        production_countries=_parse_listish(row.get("tmdb_production_countries")),
        predicted_preference=float(row["predicted_preference"]),
        raw_rank=int(row["rank"]),
        candidate_provenance=_parse_listish(row.get("candidate_sources")),
        candidate_source_ranks=_parse_listish(row.get("candidate_source_ranks")),
        semantic_evidence=evidence,
    )


def _build_candidate_items(
    frame: pd.DataFrame,
    taste_profile: pd.DataFrame,
    semantic_lookup: pd.DataFrame,
) -> list[CandidateContextItem]:
    items: list[CandidateContextItem] = []
    for _, row in frame.iterrows():
        items.append(_frame_to_candidate_item(row, taste_profile, semantic_lookup))
    return items


def _prepare_universe(
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    candidate_semantic_path: str | Path = DEFAULT_CANDIDATE_SEMANTIC_PATH,
    candidate_embeddings_path: str | Path = DEFAULT_CANDIDATE_EMBEDDINGS_PATH,
    taste_profile_path: str | Path = DEFAULT_TASTE_PROFILE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frame = _load_candidate_frame(Path(run_dir)).copy().reset_index(drop=True)
    candidate_frame["source_id"] = candidate_frame["source_id"].astype(str)

    candidate_embeddings = _load_embedding_cache(candidate_embeddings_path)
    if candidate_embeddings.empty:
        raise RuntimeError("Frozen candidate embeddings are unavailable.")
    candidate_embeddings = candidate_embeddings.loc[:, ["source_id", *_vector_columns(candidate_embeddings)]].copy()
    candidate_embeddings["source_id"] = candidate_embeddings["source_id"].astype(str)

    candidate_semantics = _load_candidate_semantics(candidate_semantic_path)
    candidate_semantics["canonical_id"] = candidate_semantics["canonical_id"].astype(str)
    candidate_semantics = candidate_semantics.rename(columns={"canonical_id": "source_id"})
    candidate_semantics["source_id"] = candidate_semantics["source_id"].astype(str)

    universe = (
        candidate_frame.merge(candidate_embeddings, on="source_id", how="left", validate="one_to_one")
        .merge(candidate_semantics, on="source_id", how="left", validate="one_to_one")
        .reset_index(drop=True)
    )
    taste_profile = _load_taste_profile(taste_profile_path)
    return universe, candidate_semantics, taste_profile


def _rank_universe(
    eligible: pd.DataFrame,
    *,
    intent,
    query: str,
    candidate_context_size: int,
    candidate_universe_count: int,
    embedding_fn: Callable[[str], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, RuntimeV3RetrievalDiagnostics]:
    eligible = eligible.copy().reset_index(drop=True)
    post_constraint_candidate_count = int(len(eligible))
    if post_constraint_candidate_count == 0:
        diagnostics = RuntimeV3RetrievalDiagnostics(
            candidate_universe_count=candidate_universe_count,
            post_constraint_candidate_count=0,
            contextual_retrieval_used=False,
            candidate_context_size=candidate_context_size,
            shortlist_size=0,
            retrieval_mode="no_match",
            context_rank_map={},
            b3_rank_map={},
            rrf_rank_map={},
            final_shortlist_rank_map={},
        )
        return eligible, diagnostics

    retrieval_used = _has_meaningful_context(query, intent)
    b3_sorted = eligible.sort_values(["rank", "predicted_preference", "source_id"], ascending=[True, False, True]).copy()
    b3_rank_map = {str(row.source_id): index + 1 for index, row in enumerate(b3_sorted.itertuples(index=False))}

    if not retrieval_used:
        shortlist = b3_sorted.head(candidate_context_size).copy().reset_index(drop=True)
        diagnostics = RuntimeV3RetrievalDiagnostics(
            candidate_universe_count=candidate_universe_count,
            post_constraint_candidate_count=post_constraint_candidate_count,
            contextual_retrieval_used=False,
            candidate_context_size=candidate_context_size,
            shortlist_size=int(len(shortlist)),
            retrieval_mode="b3_only",
            context_rank_map={str(row.source_id): None for row in shortlist.itertuples(index=False)},
            b3_rank_map=b3_rank_map,
            rrf_rank_map={str(row.source_id): None for row in shortlist.itertuples(index=False)},
            final_shortlist_rank_map={str(row.source_id): index + 1 for index, row in enumerate(shortlist.itertuples(index=False))},
        )
        return shortlist, diagnostics

    if embedding_fn is not None:
        query_vector = embedding_fn(query)
    else:
        keys = get_api_keys()
        if not keys.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for contextual retrieval.")
        query_vector = _embed_text(query, keys.openai_api_key)
    if query_vector.size == 0:
        raise RuntimeError("Query embedding was empty.")

    emb_cols = _vector_columns(eligible)
    if not emb_cols:
        raise RuntimeError("Frozen candidate embeddings are unavailable in the universe frame.")
    emb_matrix = eligible.loc[:, emb_cols].to_numpy(dtype=float)
    normalized_embeddings = _l2_normalize_rows(emb_matrix)
    context_scores = normalized_embeddings @ _l2_normalize_vector(np.asarray(query_vector, dtype=float))
    eligible = eligible.assign(context_relevance=context_scores)

    context_sorted = eligible.sort_values(
        ["context_relevance", "rank", "source_id"],
        ascending=[False, True, True],
    ).copy()
    context_rank_map = {str(row.source_id): index + 1 for index, row in enumerate(context_sorted.itertuples(index=False))}

    fusion = eligible.copy()
    fusion["context_rank"] = fusion["source_id"].astype(str).map(context_rank_map)
    fusion["b3_rank"] = fusion["source_id"].astype(str).map(b3_rank_map)
    fusion["rrf_score"] = (1.0 / (RRF_K + fusion["context_rank"].astype(float))) + (1.0 / (RRF_K + fusion["b3_rank"].astype(float)))
    fusion = fusion.sort_values(
        ["rrf_score", "context_relevance", "rank", "source_id"],
        ascending=[False, False, True, True],
    ).copy()

    shortlist = fusion.head(candidate_context_size).copy().reset_index(drop=True)
    final_shortlist_rank_map = {str(row.source_id): index + 1 for index, row in enumerate(shortlist.itertuples(index=False))}
    rrf_rank_map = {str(row.source_id): index + 1 for index, row in enumerate(fusion.itertuples(index=False))}
    diagnostics = RuntimeV3RetrievalDiagnostics(
        candidate_universe_count=candidate_universe_count,
        post_constraint_candidate_count=post_constraint_candidate_count,
        contextual_retrieval_used=True,
        candidate_context_size=candidate_context_size,
        shortlist_size=int(len(shortlist)),
        retrieval_mode="rrf_fusion",
        context_rank_map=context_rank_map,
        b3_rank_map=b3_rank_map,
        rrf_rank_map=rrf_rank_map,
        final_shortlist_rank_map=final_shortlist_rank_map,
    )
    return shortlist, diagnostics


def build_runtime_v3_recommendation_context(
    query: str,
    target_candidate_id: str | None = None,
    candidate_context_size: int = DEFAULT_CANDIDATE_CONTEXT_SIZE,
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    taste_profile_path: str | Path = "mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv",
    candidate_semantic_path: str | Path = "mvp/artifacts/diagnostics/comprehensive_discovery_audit/candidate_semantic_vectors.csv",
    candidate_embeddings_path: str | Path = DEFAULT_CANDIDATE_EMBEDDINGS_PATH,
    embedding_fn: Callable[[str], np.ndarray] | None = None,
) -> tuple[RecommendationRuntimeContext, RuntimeV3RetrievalDiagnostics]:
    intent = parse_intent(query)
    if intent.intent_type == "EXPLAIN" and target_candidate_id:
        from mvp.src.generative.context_builder import build_recommendation_runtime_context

        context = build_recommendation_runtime_context(
            query=query,
            target_candidate_id=target_candidate_id,
            candidate_context_size=candidate_context_size,
            recommendation_count=recommendation_count,
            run_dir=run_dir,
            taste_profile_path=taste_profile_path,
            candidate_semantic_path=candidate_semantic_path,
        )
        diagnostics = RuntimeV3RetrievalDiagnostics(
            candidate_universe_count=len(context.candidate_frame),
            post_constraint_candidate_count=len(context.candidate_context),
            contextual_retrieval_used=False,
            candidate_context_size=context.candidate_context_size,
            shortlist_size=len(context.candidate_context),
            retrieval_mode="explain_passthrough",
            context_rank_map={candidate.candidate_id: None for candidate in context.candidate_context},
            b3_rank_map={candidate.candidate_id: candidate.raw_rank for candidate in context.candidate_context},
            rrf_rank_map={candidate.candidate_id: None for candidate in context.candidate_context},
            final_shortlist_rank_map={candidate.candidate_id: index + 1 for index, candidate in enumerate(context.candidate_context)},
        )
        return context, diagnostics

    universe, candidate_semantics, taste_profile = _prepare_universe(
        run_dir=run_dir,
        candidate_semantic_path=candidate_semantic_path,
        candidate_embeddings_path=candidate_embeddings_path,
        taste_profile_path=taste_profile_path,
    )
    candidate_universe_count = int(len(universe))

    eligible_rows = universe.loc[universe.apply(lambda row: _candidate_passes_hard_filters(row, intent), axis=1)].copy()
    shortlist_frame, diagnostics = _rank_universe(
        eligible_rows,
        intent=intent,
        query=query,
        candidate_context_size=candidate_context_size,
        candidate_universe_count=candidate_universe_count,
        embedding_fn=embedding_fn,
    )

    semantic_lookup = candidate_semantics.set_index("source_id")
    candidate_context = _build_candidate_items(shortlist_frame, taste_profile, semantic_lookup)
    hard_filters = {
        "requested_genres": intent.requested_genres,
        "requested_languages": intent.requested_languages,
        "requested_countries": intent.requested_countries,
        "requested_decades": intent.requested_decades,
        "exclusions": intent.exclusions,
    }

    context = RecommendationRuntimeContext(
        run_dir=Path(run_dir),
        query=query,
        intent=intent,
        target_candidate_id=str(target_candidate_id) if target_candidate_id is not None else None,
        candidate_context_size=candidate_context_size,
        recommendation_count=recommendation_count,
        candidate_frame=shortlist_frame.reset_index(drop=True),
        candidate_semantics=candidate_semantics,
        taste_profile=taste_profile,
        candidate_context=candidate_context,
        positive_taste_profile=_to_signals(_eligible_taste_profile(taste_profile).loc[lambda frame: frame["preference_association"] > 0]),
        negative_taste_profile=_to_signals(_eligible_taste_profile(taste_profile).loc[lambda frame: frame["preference_association"] < 0]),
        hard_filters=hard_filters,
    )
    return context, diagnostics


def diagnose_runtime_v3_retrieval(
    queries: Sequence[dict[str, Any]],
    *,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    candidate_context_size: int = DEFAULT_CANDIDATE_CONTEXT_SIZE,
    candidate_semantic_path: str | Path = "mvp/artifacts/diagnostics/comprehensive_discovery_audit/candidate_semantic_vectors.csv",
    candidate_embeddings_path: str | Path = DEFAULT_CANDIDATE_EMBEDDINGS_PATH,
    taste_profile_path: str | Path = "mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv",
    embedding_fn: Callable[[str], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    universe, _, _ = _prepare_universe(
        run_dir=run_dir,
        candidate_semantic_path=candidate_semantic_path,
        candidate_embeddings_path=candidate_embeddings_path,
        taste_profile_path=taste_profile_path,
    )
    v2_top50 = universe.sort_values(["rank", "predicted_preference", "source_id"], ascending=[True, False, True]).head(candidate_context_size)
    v2_top50_ids = {str(value) for value in v2_top50["source_id"].astype(str).tolist()}

    results: list[dict[str, Any]] = []
    for query_item in queries:
        query = str(query_item.get("query") or query_item.get("exact_prompt") or query_item.get("prompt") or "")
        hitl_id = str(query_item.get("hitl_id") or query_item.get("case_id") or "")
        if not query:
            raise KeyError("Each diagnostic query must include query, exact_prompt, or prompt.")
        context, diagnostics = build_runtime_v3_recommendation_context(
            query=query,
            target_candidate_id=query_item.get("candidate_id"),
            candidate_context_size=candidate_context_size,
            run_dir=run_dir,
            taste_profile_path=taste_profile_path,
            candidate_semantic_path=candidate_semantic_path,
            candidate_embeddings_path=candidate_embeddings_path,
            embedding_fn=embedding_fn,
        )
        shortlist_ids = [candidate.candidate_id for candidate in context.candidate_context]
        overlap = len(set(shortlist_ids).intersection(v2_top50_ids))
        newly_admitted = len([candidate_id for candidate_id in shortlist_ids if candidate_id not in v2_top50_ids])
        top_10 = []
        shortlist_frame = context.candidate_frame.copy()
        if "context_relevance" not in shortlist_frame.columns:
            shortlist_frame["context_relevance"] = np.nan
        for _, row in shortlist_frame.head(10).iterrows():
            candidate_id = str(row["source_id"])
            top_10.append(
                {
                    "candidate_id": candidate_id,
                    "title": row.get("title"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "b3_rank": diagnostics.b3_rank_map.get(candidate_id),
                    "context_rank": diagnostics.context_rank_map.get(candidate_id),
                    "rrf_rank": diagnostics.rrf_rank_map.get(candidate_id),
                    "final_shortlist_rank": diagnostics.final_shortlist_rank_map.get(candidate_id),
                }
            )
        results.append(
            {
                "hitl_id": hitl_id,
                "exact_prompt": query,
                "parsed_intent": context.intent.intent_type,
                "candidate_universe_count": diagnostics.candidate_universe_count,
                "post_constraint_candidate_count": diagnostics.post_constraint_candidate_count,
                "contextual_retrieval_used": diagnostics.contextual_retrieval_used,
                "runtime_v2_top50_overlap": overlap,
                "newly_admitted_from_outside_b3_top50": newly_admitted,
                "retrieval_mode": diagnostics.retrieval_mode,
                "top_10_shortlist": top_10,
            }
        )
    return results
