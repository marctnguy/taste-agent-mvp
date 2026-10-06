from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import socket
import subprocess
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.watchlist_personalization.run_isolated_experiment import (
    build_consumed_keys,
    build_historical_taste,
    build_request_specific_pool,
    compute_b3_scores,
    compute_direct_taste_scores,
    compute_request_relevance,
    configure_tmdb_client,
    exclude_consumed,
    embed_candidates,
    load_env,
    load_inputs,
    load_watchlist,
    percentile_scores,
    prepare_candidate_frame,
    qualify_candidates,
    select_request_relevance_candidates,
    write_json,
)
from mvp.src.candidates import load_consumption_history
from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.schemas import ReferenceSpec, RequestUnderstanding, SemanticConcept, SemanticRequirementGroup, StructuredConstraints
from mvp.src.mvp_deployment import build_film_documents, load_or_generate_candidate_embeddings, load_training_population
from mvp.src.retrieval.catalog_retrieval import _enrich_movie_candidate, _tmdb_request

PROMPT_MANIFEST_PATH = REPO_ROOT / "evaluation/hitl/runtime_v4_final/05_run_manifest.json"
PROMPT_SOURCE_PATH = REPO_ROOT / "evaluation/hitl/runtime_v3/hitl_runtime_v3_rating_template.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation/watchlist_personalization/reversible_history_watchlist"
WATCHLIST_ZIP = REPO_ROOT.parent / "letterboxd-marctguy-2026-09-17-17-00-utc.zip"
TRAIN_EMBEDDINGS_PATH = REPO_ROOT / "mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv"
CONDITION_A_PATH = REPO_ROOT / "mvp/data/processed/condition_a_enriched.csv"
PRIMARY_SPLIT_PATH = REPO_ROOT / "mvp/data/processed/primary_holdout_split.csv"

SEED = 20261005
SLATE_SIZE = 5
REQUEST_EXTERNAL_LIMIT = 12
HISTORY_EXTERNAL_LIMIT = 12
WATCHLIST_LIMIT = 12
SPARSE_WATCHLIST_LIMIT = 8
HISTORY_SEED_LIMIT = 5
HISTORY_RECOMMENDATION_LIMIT = 5

HITL_IDS = [f"H{index:02d}" for index in range(1, 13)]
EUROPE_COUNTRIES = ["DE", "ES", "FR", "GB", "IT"]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def load_authoritative_prompts() -> pd.DataFrame:
    manifest = json.loads(PROMPT_MANIFEST_PATH.read_text())
    source_path = REPO_ROOT / manifest["source_prompt_path"]
    frame = pd.read_csv(source_path)
    frame = frame.loc[frame["hitl_id"].isin(HITL_IDS)].copy().reset_index(drop=True)
    if list(frame["hitl_id"].astype(str)) != HITL_IDS:
        raise RuntimeError("The authoritative H01-H12 prompt set is incomplete or reordered.")
    h11 = frame.loc[frame["hitl_id"] == "H11", "prompt"].iloc[0]
    if "La Bola Negra" not in h11 or "Penelope Cruz" not in h11:
        raise RuntimeError("H11 wording is not the frozen authoritative prompt.")
    return frame


def probe_service(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> dict[str, Any]:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return {"state": "network_failure", "message": "missing host"}
    try:
        socket.getaddrinfo(host, parsed.port or 443)
    except socket.gaierror as exc:
        return {"state": "dns_failure", "message": str(exc)}
    try:
        request = Request(url, headers=headers or {})
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        json.loads(payload)
        return {"state": "ok", "http_status": 200, "message": "ok"}
    except HTTPError as exc:
        return {"state": "http_error", "http_status": exc.code, "message": str(exc)}
    except json.JSONDecodeError as exc:
        return {"state": "parse_failure", "http_status": None, "message": str(exc)}
    except URLError as exc:
        return {"state": "network_failure", "http_status": None, "message": str(exc)}
    except Exception as exc:
        return {"state": "network_failure", "http_status": None, "message": str(exc)}


def preflight_services() -> dict[str, Any]:
    from mvp.src.config import get_api_keys

    keys = get_api_keys()
    checks = {
        "credentials": {
            "tmdb_api_key": bool(keys.tmdb_api_key),
            "openai_api_key": bool(keys.openai_api_key),
        },
        "services": {},
    }
    if not keys.tmdb_api_key:
        checks["services"]["tmdb"] = {"state": "missing_credentials", "message": "TMDB_API_KEY is missing"}
    else:
        checks["services"]["tmdb"] = probe_service(
            f"https://api.themoviedb.org/3/configuration?api_key={keys.tmdb_api_key}"
        )
    if not keys.openai_api_key:
        checks["services"]["openai"] = {"state": "missing_credentials", "message": "OPENAI_API_KEY is missing"}
    else:
        checks["services"]["openai"] = probe_service(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {keys.openai_api_key}"},
        )
    return checks


def validate_controlled_requests(cases: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    prompt_lookup = {row["hitl_id"]: row["prompt"] for _, row in cases.iterrows()}
    if "slow or alternative" not in prompt_lookup["H04"].lower():
        issues.append("H04 lost the slow OR alternative wording.")
    if "jeanne dielman" not in prompt_lookup["H04"].lower():
        issues.append("H04 lost the Jeanne Dielman reference.")
    if "la bola negra" not in prompt_lookup["H11"].lower() or "penelope cruz" not in prompt_lookup["H11"].lower():
        issues.append("H11 lost the La Bola Negra / Penelope Cruz reference wording.")
    if "might not discover on my own" not in prompt_lookup["H12"].lower():
        issues.append("H12 lost the explicit discovery framing.")
    return issues


def normalize_candidate_payload(frame: pd.DataFrame, route_label: str, route_origin: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    out["candidate_sources"] = [["film"]] * len(out)
    out["candidate_source_ranks"] = [["1"]] * len(out)
    out["route_memberships"] = [[route_label]] * len(out)
    out["route_origin"] = route_origin
    out["route_rank"] = np.arange(1, len(out) + 1)
    return out


def merge_route_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    combined["source_id"] = combined["source_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for source_id, group in combined.groupby("source_id", sort=False):
        row = group.iloc[0].to_dict()
        memberships: list[str] = []
        for value in group.get("route_memberships", []):
            if isinstance(value, list):
                memberships.extend([str(item) for item in value if str(item).strip()])
            elif pd.notna(value):
                memberships.append(str(value))
        row["route_memberships"] = list(dict.fromkeys(memberships))
        row["route_origin"] = "external" if any(member != "watchlist" for member in row["route_memberships"]) else "watchlist"
        for column in ["request_route_rank", "history_route_rank", "watchlist_route_rank"]:
            if column in group.columns:
                numeric = pd.to_numeric(group[column], errors="coerce").dropna()
                row[column] = int(numeric.min()) if not numeric.empty else None
        rows.append(row)
    merged = pd.DataFrame(rows)
    merged["source_id"] = merged["source_id"].astype(str)
    return merged.reset_index(drop=True)


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_") and len(column) == 8]


def _primary_genre(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).strip().lower()
    text = str(value or "").strip()
    if not text or text in {"nan", "None"}:
        return ""
    return text.split("|")[0].strip().lower()


def _decade(value: Any) -> str:
    try:
        year = int(float(value))
        return f"{year // 10 * 10}s"
    except Exception:
        return ""


def select_diverse_history_seeds(frame: pd.DataFrame, limit: int = HISTORY_SEED_LIMIT) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.copy().reset_index(drop=True)
    ordered["request_relevance"] = pd.to_numeric(ordered["request_relevance"], errors="coerce").fillna(-1.0)
    ordered["primary_genre"] = ordered.get("tmdb_genres", pd.Series(dtype=object)).apply(_primary_genre)
    ordered["release_decade"] = ordered.get("release_year", ordered.get("year", pd.Series(dtype=object))).apply(_decade)
    ordered = ordered.sort_values(
        ["request_relevance", "rating", "source_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    chosen: list[pd.Series] = []
    seen_keys: set[tuple[str, str]] = set()
    for _, row in ordered.iterrows():
        key = (str(row.get("release_decade") or ""), str(row.get("primary_genre") or ""))
        if key in seen_keys and len(chosen) < limit:
            continue
        chosen.append(row)
        seen_keys.add(key)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        selected_ids = {str(row["source_id"]) for row in chosen}
        for _, row in ordered.iterrows():
            if str(row["source_id"]) in selected_ids:
                continue
            chosen.append(row)
            if len(chosen) >= limit:
                break
    if not chosen:
        return pd.DataFrame(columns=frame.columns)
    return pd.DataFrame(chosen).reset_index(drop=True)


def build_request_external_route(
    request_id: str,
    request: RequestUnderstanding,
    client,
    consumed_keys: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    request_frame, request_report = build_request_specific_pool(request, client)
    request_frame = exclude_consumed(request_frame, consumed_keys)
    request_frame = request_frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    request_frame = normalize_candidate_payload(request_frame, "request_external", "external")
    request_frame, embed_report = embed_candidates(request_frame, request_id=f"{request_id}_request_external")
    request_frame, relevance_report = compute_request_relevance(request_frame, request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query)
    request_frame = select_request_relevance_candidates(request_frame, limit=REQUEST_EXTERNAL_LIMIT)
    request_frame["request_route_rank"] = np.arange(1, len(request_frame) + 1)
    return request_frame, {"request": request_report, "embedding": embed_report, "relevance": relevance_report}


def build_watchlist_route(
    request_id: str,
    request: RequestUnderstanding,
    watchlist_frame: pd.DataFrame,
    consumed_keys: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if watchlist_frame.empty:
        return watchlist_frame.copy(), {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": 0}
    frame = watchlist_frame.copy().reset_index(drop=True)
    frame = exclude_consumed(frame, consumed_keys)
    frame = frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    frame = normalize_candidate_payload(frame, "watchlist", "watchlist")
    frame, embed_report = embed_candidates(frame, request_id=f"{request_id}_watchlist")
    frame, relevance_report = compute_request_relevance(frame, request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query)
    frame = select_request_relevance_candidates(frame, limit=WATCHLIST_LIMIT)
    frame["watchlist_route_rank"] = np.arange(1, len(frame) + 1)
    return frame, {"embedding": embed_report, "relevance": relevance_report, "resolved_watchlist_candidates": int(len(frame))}


def _movie_recommendations(client, tmdb_id: int) -> list[dict[str, Any]]:
    try:
        payload = _tmdb_request(client, f"/movie/{int(tmdb_id)}/recommendations", {"page": 1})
    except Exception as exc:
        raise RuntimeError(f"TMDB recommendations failed for seed {tmdb_id}: {exc}") from exc
    return payload.get("results", []) or []


def build_history_external_route(
    request_id: str,
    request: RequestUnderstanding,
    training_positive: pd.DataFrame,
    client,
    consumed_keys: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if training_positive.empty:
        return pd.DataFrame(columns=["source_id"]), {"seed_count": 0, "history_candidate_count": 0}
    positive = training_positive.copy().reset_index(drop=True)
    positive = normalize_candidate_payload(positive, "history_seed", "external")
    positive, _ = compute_request_relevance(positive, request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query)
    seeds = select_diverse_history_seeds(positive, limit=HISTORY_SEED_LIMIT)
    rows: list[dict[str, Any]] = []
    seed_trace: list[dict[str, Any]] = []
    for seed_rank, (_, seed) in enumerate(seeds.iterrows(), start=1):
        tmdb_id = seed.get("tmdb_id")
        if pd.isna(tmdb_id):
            continue
        seed_trace.append(
            {
                "seed_source_id": str(seed.get("source_id")),
                "seed_tmdb_id": int(tmdb_id),
                "seed_title": seed.get("title"),
                "seed_request_relevance": float(seed.get("request_relevance")) if pd.notna(seed.get("request_relevance")) else None,
                "seed_rank": seed_rank,
            }
        )
        for rank, movie in enumerate(_movie_recommendations(client, int(tmdb_id))[:HISTORY_RECOMMENDATION_LIMIT], start=1):
            candidate = _enrich_movie_candidate(client, movie)
            if not candidate:
                continue
            candidate["candidate_sources"] = ["film"]
            candidate["candidate_source_ranks"] = ["1"]
            candidate["route_memberships"] = ["history_external"]
            candidate["history_seed_source_id"] = str(seed.get("source_id"))
            candidate["history_seed_tmdb_id"] = int(tmdb_id)
            candidate["history_seed_rank"] = seed_rank
            candidate["history_recommendation_rank"] = rank
            rows.append(candidate)
    if not rows:
        return pd.DataFrame(columns=["source_id"]), {"seed_count": int(len(seed_trace)), "history_candidate_count": 0, "seed_trace": seed_trace}
    frame = pd.DataFrame(rows).drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    frame["source_id"] = frame["source_id"].astype(str)
    frame = exclude_consumed(frame, consumed_keys)
    frame = normalize_candidate_payload(frame, "history_external", "external")
    frame, embed_report = embed_candidates(frame, request_id=f"{request_id}_history_external")
    frame, relevance_report = compute_request_relevance(frame, request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query)
    frame = select_request_relevance_candidates(frame, limit=HISTORY_EXTERNAL_LIMIT)
    frame["history_route_rank"] = np.arange(1, len(frame) + 1)
    return frame, {
        "seed_count": int(len(seed_trace)),
        "history_candidate_count": int(len(frame)),
        "seed_trace": seed_trace,
        "embedding": embed_report,
        "relevance": relevance_report,
    }


def validate_request_correction(hitl_id: str, request: RequestUnderstanding) -> list[str]:
    issues: list[str] = []
    spec = request.spec
    intent = request.intent
    if spec is None:
        return [f"{hitl_id}: missing request spec"]

    required = {
        concept.concept or concept.aspect_id
        for concept in spec.semantic_requirements
        if str(getattr(concept, "importance", "preferred")).lower() == "required"
    }
    preferred = {
        concept.concept or concept.aspect_id
        for concept in spec.semantic_requirements
        if str(getattr(concept, "importance", "preferred")).lower() != "required"
    }
    groups = spec.semantic_requirement_groups
    exclusions = {concept.concept or concept.aspect_id for concept in spec.semantic_exclusions}

    if hitl_id == "H03":
        if "comforting" not in required:
            issues.append("H03 must preserve comforting as required.")
        if "cheesy" not in exclusions:
            issues.append("H03 must preserve cheesy as an exclusion.")
    elif hitl_id == "H04":
        if not any(group.mode == "any_of" and {concept.concept or concept.aspect_id for concept in group.concepts} >= {"slow_burn", "alternative"} for group in groups):
            issues.append("H04 must preserve slow OR alternative semantics as an any_of group.")
        if not any((concept.concept or concept.aspect_id) == "unsettling" for concept in spec.semantic_requirements):
            issues.append("H04 must preserve unsettling/anxiety-inducing semantics.")
        if str(request.reference_title or "").strip().lower() != "jeanne dielman":
            issues.append("H04 must preserve the Jeanne Dielman reference.")
    elif hitl_id == "H06":
        if request.request_mode != "novelty":
            issues.append("H06 must be treated as a novelty request.")
    elif hitl_id == "H08":
        if "fr" not in {str(value).lower() for value in spec.structured_constraints.required_languages}:
            issues.append("H08 must preserve the French language constraint.")
    elif hitl_id == "H09":
        if not set(EUROPE_COUNTRIES).intersection({str(value).upper() for value in spec.structured_constraints.required_countries}):
            issues.append("H09 must preserve the Europe country proxy.")
        if not {"freedom", "youthful"}.issubset(required | preferred):
            issues.append("H09 must preserve freedom and youthful semantics.")
    elif hitl_id == "H10":
        if "drama" not in {str(value).lower() for value in spec.structured_constraints.excluded_genres}:
            issues.append("H10 must preserve drama as an exclusion.")
    elif hitl_id == "H11":
        if str(request.reference_title or "").strip().lower() != "la bola negra":
            issues.append("H11 must preserve La Bola Negra as the reference title.")
        if not any(group.mode == "preferred" and {concept.concept or concept.aspect_id for concept in group.concepts} >= {"period_setting"} for group in groups):
            issues.append("H11 must preserve period_setting as a preference.")
        if not {"performers", "fame", "show_business"}.issubset(required | preferred):
            issues.append("H11 must preserve performers/fame/show business semantics.")
    elif hitl_id == "H12":
        if request.request_mode != "novelty" and not request.novelty_requested:
            issues.append("H12 must be treated as a novelty / long-tail discovery request.")
    return issues


def correct_hitl_request(hitl_id: str, request: RequestUnderstanding) -> tuple[RequestUnderstanding, list[str]]:
    corrections: list[str] = []
    spec = request.spec
    if spec is None:
        return request, corrections

    def concept(aspect_id: str, label: str, importance: str = "preferred") -> SemanticConcept:
        return SemanticConcept(aspect_id=aspect_id, concept=label, importance=importance, source_span=request.query)

    if hitl_id == "H03":
        comforting = concept("comforting", "comforting", "required")
        cheesy = concept("cheesy", "cheesy", "preferred")
        request = request.model_copy(
            update={
                "request_mode": "contextual",
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [comforting],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(group_id="comforting", mode="all_of", concepts=[comforting], source_span=request.query)
                        ],
                        "semantic_exclusions": [cheesy],
                        "request_summary": "Something comforting but not cheesy.",
                        "semantic_query_text": "comforting not cheesy",
                    }
                ),
            }
        )
        corrections.append("H03 normalized to preserve comforting and cheesy exclusion.")
    elif hitl_id == "H04":
        slow = concept("slow_burn", "slow_burn", "preferred")
        alternative = concept("alternative", "alternative", "preferred")
        unsettling = concept("unsettling", "unsettling", "required")
        reference = ReferenceSpec(title="Jeanne Dielman", relation="reference", resolved_tmdb_id=None, resolved_tmdb_title=None, resolution_status="absent")
        request = request.model_copy(
            update={
                "request_mode": "reference",
                "reference_title": "Jeanne Dielman",
                "reference_status": "absent",
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [slow, alternative, unsettling],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(group_id="pace", mode="any_of", concepts=[slow, alternative], source_span="slow or alternative in atmosphere/pace"),
                            SemanticRequirementGroup(group_id="anxiety", mode="all_of", concepts=[unsettling], source_span="extremely unsettling and anxiety-inducing"),
                        ],
                        "semantic_exclusions": [],
                        "reference": reference,
                        "request_summary": "Slow or alternative in atmosphere/pace but still extremely unsettling and anxiety-inducing, like Jeanne Dielman.",
                        "semantic_query_text": "slow or alternative unsettling anxiety inducing Jeanne Dielman",
                    }
                ),
            }
        )
        corrections.append("H04 restored OR semantics and explicit reference context.")
    elif hitl_id == "H06":
        request = request.model_copy(update={"request_mode": "novelty", "novelty_requested": True})
        if spec.novelty_goal.enabled is False:
            request = request.model_copy(update={"spec": spec.model_copy(update={"novelty_goal": spec.novelty_goal.model_copy(update={"enabled": True, "priority": "different from what I normally watch"})})})
        corrections.append("H06 normalized as novelty request.")
    elif hitl_id == "H08":
        french = concept("french", "french", "required")
        request = request.model_copy(
            update={
                "request_mode": "contextual",
                "spec": spec.model_copy(
                    update={
                        "structured_constraints": StructuredConstraints(
                            required_languages=["fr"],
                            excluded_languages=list(spec.structured_constraints.excluded_languages),
                            required_countries=list(spec.structured_constraints.required_countries),
                            excluded_countries=list(spec.structured_constraints.excluded_countries),
                            required_genres=list(spec.structured_constraints.required_genres),
                            excluded_genres=list(spec.structured_constraints.excluded_genres),
                            required_decades=list(spec.structured_constraints.required_decades),
                            excluded_decades=list(spec.structured_constraints.excluded_decades),
                            min_year=spec.structured_constraints.min_year,
                            max_year=spec.structured_constraints.max_year,
                        ),
                        "semantic_requirements": [french],
                        "semantic_requirement_groups": [SemanticRequirementGroup(group_id="language", mode="all_of", concepts=[french], source_span=request.query)],
                        "request_summary": "A French film tonight.",
                        "semantic_query_text": "French film",
                    }
                ),
            }
        )
        corrections.append("H08 normalized to a French language constraint.")
    elif hitl_id == "H09":
        freedom = concept("freedom", "freedom", "required")
        youthful = concept("youthful", "youthful", "required")
        request = request.model_copy(
            update={
                "request_mode": "contextual",
                "spec": spec.model_copy(
                    update={
                        "structured_constraints": StructuredConstraints(
                            required_languages=list(spec.structured_constraints.required_languages),
                            excluded_languages=list(spec.structured_constraints.excluded_languages),
                            required_countries=list(EUROPE_COUNTRIES),
                            excluded_countries=list(spec.structured_constraints.excluded_countries),
                            required_genres=list(spec.structured_constraints.required_genres),
                            excluded_genres=list(spec.structured_constraints.excluded_genres),
                            required_decades=list(spec.structured_constraints.required_decades),
                            excluded_decades=list(spec.structured_constraints.excluded_decades),
                            min_year=spec.structured_constraints.min_year,
                            max_year=spec.structured_constraints.max_year,
                        ),
                        "semantic_requirements": [freedom, youthful],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(group_id="freedom", mode="all_of", concepts=[freedom], source_span="feels like freedom"),
                            SemanticRequirementGroup(group_id="youthful", mode="all_of", concepts=[youthful], source_span="and youthful"),
                        ],
                        "request_summary": "A European movie that feels like freedom and youthful.",
                        "semantic_query_text": "European freedom youthful",
                    }
                ),
            }
        )
        corrections.append("H09 normalized Europe to the frozen any-of country proxy.")
    elif hitl_id == "H10":
        classic = concept("classic", "classic", "required")
        drama_exclusion = concept("drama", "drama", "preferred")
        request = request.model_copy(
            update={
                "request_mode": "contextual",
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [classic],
                        "semantic_requirement_groups": [SemanticRequirementGroup(group_id="classic", mode="all_of", concepts=[classic], source_span="good old classic")],
                        "semantic_exclusions": [drama_exclusion],
                        "request_summary": "A good old classic, but no drama.",
                        "semantic_query_text": "good old classic no drama",
                    }
                ),
            }
        )
        corrections.append("H10 preserved the no-drama exclusion.")
    elif hitl_id == "H11":
        performers = concept("performers", "performers", "required")
        fame = concept("fame", "fame", "required")
        show_business = concept("show_business", "show_business", "required")
        period_setting = concept("period_setting", "period_setting", "preferred")
        request = request.model_copy(
            update={
                "request_mode": "reference",
                "reference_title": "La Bola Negra",
                "reference_status": "unresolved",
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [performers, fame, show_business, period_setting],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(group_id="required", mode="all_of", concepts=[performers, fame, show_business], source_span="performers, fame and/or show business"),
                            SemanticRequirementGroup(group_id="period", mode="preferred", concepts=[period_setting], source_span="preferably with a period setting"),
                        ],
                        "semantic_exclusions": [],
                        "reference": ReferenceSpec(title="La Bola Negra", relation="watched_reference", resolved_tmdb_id=None, resolved_tmdb_title=None, resolution_status="unresolved"),
                        "request_summary": "I just watched La Bola Negra and loved the Penelope Cruz character in it, I want something about performers, fame and/or show business preferably with a period setting.",
                        "semantic_query_text": "La Bola Negra Penelope Cruz performers fame show business period setting",
                    }
                ),
            }
        )
        corrections.append("H11 preserved La Bola Negra, the character reference, and period preference.")
    elif hitl_id == "H12":
        request = request.model_copy(update={"request_mode": "novelty", "novelty_requested": True})
        if not spec.novelty_goal.enabled:
            request = request.model_copy(update={"spec": spec.model_copy(update={"novelty_goal": spec.novelty_goal.model_copy(update={"enabled": True, "priority": "something I might not discover on my own"})})})
        corrections.append("H12 normalized as long-tail novelty discovery.")
    return request, corrections


def rank_variant_candidates(
    frame: pd.DataFrame,
    *,
    variant: str,
    hitl_id: str,
    request_mode: str,
    top_k: int = SLATE_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame.loc[frame["qualification_status"].astype(str).isin({"strong", "partial"})].copy().reset_index(drop=True)
    if eligible.empty:
        return eligible, {"selected_count": 0, "candidate_count": 0, "eligible_count": 0}

    eligible["qualification_rank"] = eligible["qualification_status"].map({"strong": 2, "partial": 1}).fillna(0).astype(int)
    eligible["request_relevance_percentile"] = percentile_scores(eligible, "request_relevance")
    eligible["direct_taste_percentile"] = percentile_scores(eligible, "direct_taste_score")
    eligible["b3_percentile"] = percentile_scores(eligible, "predicted_preference")
    eligible["novelty_percentile"] = percentile_scores(eligible, "novelty_score")

    if variant == "A":
        eligible["policy_score"] = pd.to_numeric(eligible["request_relevance_percentile"], errors="coerce")
        policy_description = "request relevance percentile"
    else:
        if request_mode == "novelty":
            eligible["policy_score"] = 0.5 * pd.to_numeric(eligible["direct_taste_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
                eligible["novelty_percentile"], errors="coerce"
            )
            policy_description = "0.5 direct taste + 0.5 novelty distance"
        elif request_mode == "generic":
            eligible["policy_score"] = pd.to_numeric(eligible["direct_taste_percentile"], errors="coerce")
            policy_description = "direct historical taste percentile"
        else:
            eligible["policy_score"] = 0.5 * pd.to_numeric(eligible["request_relevance_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
                eligible["direct_taste_percentile"], errors="coerce"
            )
            policy_description = "0.5 request relevance + 0.5 direct historical taste"

    eligible["route_priority"] = eligible["route_memberships"].apply(
        lambda memberships: 0
        if hitl_id == "H12" and isinstance(memberships, list) and "request_external" in memberships
        else (1 if hitl_id == "H12" and isinstance(memberships, list) and "history_external" in memberships else 2 if hitl_id == "H12" else 0)
    )
    ranked = eligible.sort_values(
        ["qualification_rank", "policy_score", "route_priority", "source_id"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["selected_rank"] = np.arange(1, len(ranked) + 1)
    selected = ranked.head(top_k).copy().reset_index(drop=True)
    report = {
        "variant": variant,
        "hitl_id": hitl_id,
        "request_mode": request_mode,
        "policy_description": policy_description,
        "candidate_count": int(len(frame)),
        "eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
    }
    return selected, report


def _build_watched_history_embeddings() -> pd.DataFrame:
    history = load_consumption_history()
    if history.empty:
        return pd.DataFrame()
    enriched = pd.read_csv(CONDITION_A_PATH)
    enriched["source_id"] = enriched["source_id"].astype(str)
    merged = history.merge(enriched, on="source_id", how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged = merged.loc[:, [column for column in ["source_id", "title", "year", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"] if column in merged.columns]].copy()
    merged["candidate_sources"] = [["film"]] * len(merged)
    docs = build_film_documents(merged)
    embeddings, _, _ = load_or_generate_candidate_embeddings(docs, cache_path=DEFAULT_OUTPUT_DIR / "artifacts" / "cache" / "watched_history_embeddings.csv")
    return embeddings


def _scenario_candidate_ids(
    variant: str,
    request_ids: pd.Series,
    history_ids: pd.Series,
    watchlist_ids: pd.Series,
) -> set[str]:
    selected = set(request_ids.astype(str).tolist())
    if variant == "B":
        selected.update(history_ids.astype(str).tolist())
    selected.update(watchlist_ids.astype(str).tolist())
    return selected


def _selection_subset(master: pd.DataFrame, allowed_ids: set[str]) -> pd.DataFrame:
    if master.empty:
        return master.copy()
    return master.loc[master["source_id"].astype(str).isin(allowed_ids)].copy().reset_index(drop=True)


def build_human_sheet_rows(prompt_id: str, prompt: str, selected: pd.DataFrame, scenario_label: str, variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    human_rows: list[dict[str, Any]] = []
    unblinding_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        anon_label = f"{prompt_id}-{scenario_label[:1]}{variant}-{int(row['selected_rank']):02d}"
        human_rows.append(
            {
                "hitl_id": prompt_id,
                "prompt": prompt,
                "anon_label": anon_label,
                "title": row.get("title"),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "synopsis": row.get("tmdb_overview"),
                "request_fit_1_5": "",
                "taste_fit_1_5": "",
                "discovery_value_1_5": "",
                "explanation_usefulness_1_5": "",
                "would_actually_watch": "",
                "already_knew_titles": "",
                "heard_of_titles": "",
                "new_to_me_titles": "",
                "best_recommendation": "",
                "worst_recommendation": "",
                "qualitative_feedback": "",
            }
        )
        unblinding_rows.append(
            {
                "hitl_id": prompt_id,
                "prompt": prompt,
                "anon_label": anon_label,
                "title": row.get("title"),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "variant": variant,
                "scenario": scenario_label,
                "selected_rank": int(row["selected_rank"]) if pd.notna(row.get("selected_rank")) else None,
                "route_memberships": row.get("route_memberships", []),
                "request_relevance": float(row["request_relevance"]) if pd.notna(row.get("request_relevance")) else None,
                "direct_taste_score": float(row["direct_taste_score"]) if pd.notna(row.get("direct_taste_score")) else None,
                "predicted_preference": float(row["predicted_preference"]) if pd.notna(row.get("predicted_preference")) else None,
            }
    )
    return human_rows, unblinding_rows


def build_prompt_review_pack(prompt_id: str, prompt: str, selected_a: pd.DataFrame, selected_b: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for variant, selected in [("A", selected_a), ("B", selected_b)]:
        if selected.empty:
            continue
        for _, row in selected.iterrows():
            rows.append(
                {
                    "candidate_id": str(row["source_id"]),
                    "hitl_id": prompt_id,
                    "prompt": prompt,
                    "variant": variant,
                    "selected_rank": int(row["selected_rank"]) if pd.notna(row.get("selected_rank")) else None,
                    "title": row.get("title"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "synopsis": row.get("tmdb_overview"),
                    "request_fit_1_5": "",
                    "taste_fit_1_5": "",
                    "discovery_value_1_5": "",
                    "explanation_usefulness_1_5": "",
                    "would_actually_watch": "",
                    "already_knew_titles": "",
                    "heard_of_titles": "",
                    "new_to_me_titles": "",
                    "best_recommendation": "",
                    "worst_recommendation": "",
                    "qualitative_feedback": "",
                    "route_memberships": row.get("route_memberships", []),
                    "request_relevance": float(row["request_relevance"]) if pd.notna(row.get("request_relevance")) else None,
                    "direct_taste_score": float(row["direct_taste_score"]) if pd.notna(row.get("direct_taste_score")) else None,
                    "predicted_preference": float(row["predicted_preference"]) if pd.notna(row.get("predicted_preference")) else None,
                }
            )
    if not rows:
        return [], []
    frame = pd.DataFrame(rows).drop_duplicates(subset=["hitl_id", "candidate_id"], keep="first").reset_index(drop=True)
    frame["anon_label"] = [f"{prompt_id}-F{index + 1:02d}" for index in range(len(frame))]
    human_rows = frame.loc[
        :,
        [
            "hitl_id",
            "prompt",
            "anon_label",
            "title",
            "year",
            "synopsis",
            "request_fit_1_5",
            "taste_fit_1_5",
            "discovery_value_1_5",
            "explanation_usefulness_1_5",
            "would_actually_watch",
            "already_knew_titles",
            "heard_of_titles",
            "new_to_me_titles",
            "best_recommendation",
            "worst_recommendation",
            "qualitative_feedback",
        ],
    ].to_dict(orient="records")
    unblinding_rows = frame.loc[
        :,
        [
            "hitl_id",
            "prompt",
            "anon_label",
            "candidate_id",
            "title",
            "year",
            "variant",
            "selected_rank",
            "route_memberships",
            "request_relevance",
            "direct_taste_score",
            "predicted_preference",
        ],
    ].to_dict(orient="records")
    return human_rows, unblinding_rows


def run() -> dict[str, Any]:
    load_env()
    preflight = preflight_services()
    failing = {name: status for name, status in preflight.get("services", {}).items() if status.get("state") != "ok"}
    if failing:
        raise RuntimeError(json.dumps({"preflight": preflight, "failing_services": failing}, indent=2, ensure_ascii=False, default=str))

    random.seed(SEED)
    np.random.seed(SEED)

    prompts = load_authoritative_prompts()
    prompt_issues = validate_controlled_requests(prompts)
    if prompt_issues:
        raise RuntimeError(json.dumps({"prompt_validation": prompt_issues}, indent=2, ensure_ascii=False, default=str))

    inputs = load_inputs()
    condition_a = inputs["condition_a"]
    split = inputs["split"]
    train_embeddings = inputs["train_embeddings"]
    watchlist = inputs["watchlist"]
    consumed_keys = build_consumed_keys(condition_a)
    training_population = load_training_population()
    if len(training_population) != 393:
        raise RuntimeError("Frozen training population changed.")
    if len(split.loc[split["split"] == "train"]) != 393 or len(split.loc[split["split"] == "test"]) != 99:
        raise RuntimeError("Frozen 393/99 split changed.")
    if set(training_population["source_id"].astype(str)) != set(train_embeddings["source_id"].astype(str)):
        raise RuntimeError("Training embeddings are not aligned with the frozen training split.")

    tmdb_client = configure_tmdb_client()
    historical = build_historical_taste(training_population, train_embeddings)
    watched_history_embeddings = _build_watched_history_embeddings()

    full_watchlist_resolved, watchlist_resolution_report = prepare_watchlist_resolution(watchlist, tmdb_client, consumed_keys)
    sparse_watchlist_resolved = full_watchlist_resolved.head(SPARSE_WATCHLIST_LIMIT).copy().reset_index(drop=True)
    empty_watchlist_resolved = pd.DataFrame(columns=full_watchlist_resolved.columns if not full_watchlist_resolved.empty else ["source_id"])

    artifacts_dir = DEFAULT_OUTPUT_DIR / "artifacts"
    human_dir = DEFAULT_OUTPUT_DIR / "human_scoring"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    smoke_result = evaluate_single_prompt(
        prompts.iloc[0],
        tmdb_client,
        historical,
        consumed_keys,
        watched_history_embeddings,
        full_watchlist_resolved.head(12).copy().reset_index(drop=True),
        sparse_watchlist_resolved,
        empty_watchlist_resolved,
        scenario_label="full_watchlist",
        perform_artifact_write=False,
    )
    if smoke_result.get("status") != "ok":
        raise RuntimeError(json.dumps({"smoke_test": smoke_result}, indent=2, ensure_ascii=False, default=str))

    comparison_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    unblinding_rows: list[dict[str, Any]] = []
    prompt_results: list[dict[str, Any]] = []
    scenario_summaries: dict[str, list[dict[str, Any]]] = {"full_watchlist": [], "empty_watchlist": [], "sparse_watchlist": []}
    total_llm_calls = 0
    total_tokens = {"input": 0, "output": 0, "total": 0}

    for _, prompt_row in prompts.iterrows():
        prompt_result = evaluate_single_prompt(
            prompt_row,
            tmdb_client,
            historical,
            consumed_keys,
            watched_history_embeddings,
            full_watchlist_resolved,
            sparse_watchlist_resolved,
            empty_watchlist_resolved,
            scenario_label=None,
            perform_artifact_write=False,
        )
        prompt_results.append(prompt_result)
        total_llm_calls += int(prompt_result["llm_usage"]["llm_call_count"])
        total_tokens["input"] += int(prompt_result["llm_usage"]["runtime_input_tokens"] or 0)
        total_tokens["output"] += int(prompt_result["llm_usage"]["runtime_output_tokens"] or 0)
        total_tokens["total"] += int(prompt_result["llm_usage"]["runtime_total_tokens"] or 0)
        for scenario_name, scenario_payload in prompt_result["scenarios"].items():
            scenario_summaries[scenario_name].append(
                {
                    "hitl_id": prompt_result["hitl_id"],
                    "eligible_counts": scenario_payload["eligible_counts"],
                    "A_selected_ids": scenario_payload["A"]["selected_ids"],
                    "B_selected_ids": scenario_payload["B"]["selected_ids"],
                    "A_selected_titles": scenario_payload["A"]["selected_titles"],
                    "B_selected_titles": scenario_payload["B"]["selected_titles"],
                }
            )
        full_A = prompt_result["scenarios"]["full_watchlist"]["A"]["selected"]
        full_B = prompt_result["scenarios"]["full_watchlist"]["B"]["selected"]
        rows_AB, key_AB = build_prompt_review_pack(prompt_result["hitl_id"], prompt_result["prompt"], full_A, full_B)
        human_rows.extend(rows_AB)
        unblinding_rows.extend(key_AB)
        comparison_rows.append(
            {
                "hitl_id": prompt_result["hitl_id"],
                "prompt": prompt_result["prompt"],
                "parsed_request": prompt_result["parsed_request"],
                "controlled_request_spec": prompt_result["controlled_request_spec"],
                "request_corrections": prompt_result["corrections"],
                "route_reports": prompt_result["route_reports"],
                "scenarios": {
                    scenario_name: {
                        "eligible_counts": payload["eligible_counts"],
                        "A": payload["A"]["selected_titles"],
                        "B": payload["B"]["selected_titles"],
                    }
                    for scenario_name, payload in prompt_result["scenarios"].items()
                },
            }
        )

    human_sheet = pd.DataFrame(human_rows).drop_duplicates(subset=["hitl_id", "anon_label"], keep="first").reset_index(drop=True)
    unblinding_key = pd.DataFrame(unblinding_rows).drop_duplicates(subset=["anon_label"], keep="first").reset_index(drop=True)
    human_sheet.to_csv(human_dir / "blinded_human_review_sheet.csv", index=False)
    unblinding_key.to_csv(human_dir / "unblinding_key.csv", index=False)

    manifest = {
        "seed": SEED,
        "input_hashes": {
            "watchlist_zip": sha256_path(WATCHLIST_ZIP),
            "condition_a_enriched": sha256_path(CONDITION_A_PATH),
            "primary_holdout_split": sha256_path(PRIMARY_SPLIT_PATH),
            "train_embeddings": sha256_path(TRAIN_EMBEDDINGS_PATH),
            "prompt_manifest": sha256_path(PROMPT_MANIFEST_PATH),
            "prompt_source_set": sha256_path(REPO_ROOT / json.loads(PROMPT_MANIFEST_PATH.read_text())["source_prompt_path"]),
        },
        "code_hashes": {
            "run_reversible_history_watchlist_experiment": sha256_path(Path(__file__).resolve()),
            "run_isolated_experiment": sha256_path(REPO_ROOT / "evaluation/watchlist_personalization/run_isolated_experiment.py"),
            "generative_v4_runtime": sha256_path(REPO_ROOT / "mvp/src/generative_v4/runtime.py"),
            "generative_v4_intent_chain": sha256_path(REPO_ROOT / "mvp/src/generative_v4/intent_chain.py"),
            "generative_v4_qualification_chain": sha256_path(REPO_ROOT / "mvp/src/generative_v4/qualification_chain.py"),
            "catalog_retrieval": sha256_path(REPO_ROOT / "mvp/src/retrieval/catalog_retrieval.py"),
        },
        "repo_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=REPO_ROOT, text=True),
        "scenario_settings": {
            "slate_size": SLATE_SIZE,
            "request_external_limit": REQUEST_EXTERNAL_LIMIT,
            "history_external_limit": HISTORY_EXTERNAL_LIMIT,
            "watchlist_limit": WATCHLIST_LIMIT,
            "sparse_watchlist_limit": SPARSE_WATCHLIST_LIMIT,
            "history_seed_limit": HISTORY_SEED_LIMIT,
            "history_recommendation_limit": HISTORY_RECOMMENDATION_LIMIT,
        },
        "preflight": preflight,
        "watchlist_resolution": watchlist_resolution_report,
        "training_population": {
            "train_count": 393,
            "test_count": 99,
            "positive_examples": int((pd.to_numeric(training_population["rating"], errors="coerce") >= 4.0).sum()),
            "negative_examples": int((pd.to_numeric(training_population["rating"], errors="coerce") <= 2.5).sum()),
        },
    }

    summary = {
        "status": "completed",
        "llm_calls": int(total_llm_calls),
        "token_usage": total_tokens,
        "prompt_count": int(len(prompt_results)),
        "scenarios": scenario_summaries,
        "full_watchlist_pairs_with_competing_choices": sum(1 for row in prompt_results if row["scenarios"]["full_watchlist"]["A"]["selected_ids"] != row["scenarios"]["full_watchlist"]["B"]["selected_ids"]),
        "empty_watchlist_pairs_with_competing_choices": sum(1 for row in prompt_results if row["scenarios"]["empty_watchlist"]["A"]["selected_ids"] != row["scenarios"]["empty_watchlist"]["B"]["selected_ids"]),
    }

    report_md = build_technical_report(prompt_results, summary, manifest, human_sheet, unblinding_key)

    write_json(artifacts_dir / "reversible_history_watchlist_manifest.json", manifest)
    write_json(artifacts_dir / "reversible_history_watchlist_results.json", comparison_rows)
    write_json(artifacts_dir / "reversible_history_watchlist_summary.json", summary)
    (artifacts_dir / "reversible_history_watchlist_technical_report.md").write_text(report_md)

    return {
        "status": "completed",
        "artifacts": {
            "manifest": str((artifacts_dir / "reversible_history_watchlist_manifest.json").resolve()),
            "results": str((artifacts_dir / "reversible_history_watchlist_results.json").resolve()),
            "summary": str((artifacts_dir / "reversible_history_watchlist_summary.json").resolve()),
            "technical_report": str((artifacts_dir / "reversible_history_watchlist_technical_report.md").resolve()),
            "human_sheet": str((human_dir / "blinded_human_review_sheet.csv").resolve()),
            "unblinding_key": str((human_dir / "unblinding_key.csv").resolve()),
        },
        "summary": summary,
    }


def prepare_watchlist_resolution(watchlist: pd.DataFrame, client, consumed_keys: dict[str, set[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from mvp.src.retrieval.catalog_retrieval import _enrich_movie_candidate

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for _, entry in watchlist.iterrows():
        title = entry["title"]
        year = entry["year"]
        try:
            match = client.search_movie(str(title), year)
        except Exception as exc:
            raise RuntimeError(f"TMDB watchlist search failed for {title!r} / {year!r}: {exc}") from exc
        if not match:
            unresolved += 1
            continue
        candidate = _enrich_movie_candidate(client, match)
        if not candidate:
            unresolved += 1
            continue
        candidate["candidate_sources"] = ["film"]
        candidate["candidate_source_ranks"] = ["1"]
        candidate["route_memberships"] = ["watchlist"]
        candidate["watchlist_rank"] = int(entry["watchlist_rank"])
        candidate["watchlist_uri"] = entry.get("Letterboxd URI")
        rows.append(candidate)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["source_id"] = frame["source_id"].astype(str)
        frame = exclude_consumed(frame, consumed_keys)
        frame = frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return frame, {"resolved_watchlist_candidates": int(len(frame)), "unresolved_watchlist_candidates": int(unresolved)}


def evaluate_single_prompt(
    prompt_row: pd.Series,
    tmdb_client,
    historical: dict[str, Any],
    consumed_keys: dict[str, set[str]],
    watched_history_embeddings: pd.DataFrame,
    full_watchlist_resolved: pd.DataFrame,
    sparse_watchlist_resolved: pd.DataFrame,
    empty_watchlist_resolved: pd.DataFrame,
    scenario_label: str | None,
    perform_artifact_write: bool = False,
) -> dict[str, Any]:
    hitl_id = str(prompt_row["hitl_id"])
    prompt = str(prompt_row["prompt"])
    request = understand_request(prompt)
    request, corrections = correct_hitl_request(hitl_id, request)
    issues = validate_request_correction(hitl_id, request)
    if issues:
        raise RuntimeError(json.dumps({"hitl_id": hitl_id, "issues": issues, "parsed_request": request.model_dump()}, indent=2, ensure_ascii=False, default=str))

    request_route, request_report = build_request_external_route(hitl_id, request, tmdb_client, consumed_keys)
    history_route, history_report = build_history_external_route(hitl_id, request, historical["positive"], tmdb_client, consumed_keys)
    full_watchlist_route, watchlist_report_full = build_watchlist_route(hitl_id, request, full_watchlist_resolved, consumed_keys)
    sparse_watchlist_route, watchlist_report_sparse = build_watchlist_route(hitl_id, request, sparse_watchlist_resolved, consumed_keys)
    empty_watchlist_route, watchlist_report_empty = build_watchlist_route(hitl_id, request, empty_watchlist_resolved, consumed_keys)

    master = merge_route_frames(request_route, history_route, full_watchlist_route)
    if master.empty:
        raise RuntimeError(f"{hitl_id}: no master candidates could be constructed.")
    master_emb_cols = _embedding_columns(master)
    if len(master_emb_cols) != 1536:
        raise RuntimeError(
            f"{hitl_id}: master route merge must preserve exactly 1536 embedding dimensions, found {len(master_emb_cols)}."
        )
    master["candidate_id"] = master["source_id"].astype(str)
    for column in ["tmdb_id", "year"]:
        if column in master.columns:
            master[column] = pd.Series(
                [None if pd.isna(value) else int(value) for value in master[column]],
                index=master.index,
                dtype=object,
            )
    embed_report = {
        "provider": "reused_route_embeddings",
        "model": "route_level_embedding_columns",
        "embedding_dim": 1536,
        "coverage": 1.0,
        "cache_status": "reused",
        "cache_hits": int(len(master)),
        "cache_misses": 0,
    }
    master, relevance_report = compute_request_relevance(master, request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query)
    master = compute_b3_scores(master)
    master = compute_direct_taste_scores(master, historical)
    novelty_history = watched_history_embeddings
    if not novelty_history.empty:
        candidate_cols = _embedding_columns(master)
        history_cols = _embedding_columns(novelty_history)
        if candidate_cols and history_cols:
            candidate_matrix = master.loc[:, candidate_cols].to_numpy(dtype=float)
            history_matrix = novelty_history.loc[:, history_cols].to_numpy(dtype=float)
            candidate_matrix = candidate_matrix / np.clip(np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12, None)
            history_matrix = history_matrix / np.clip(np.linalg.norm(history_matrix, axis=1, keepdims=True), 1e-12, None)
            similarities = candidate_matrix @ history_matrix.T
            master["novelty_score"] = 1.0 - similarities.max(axis=1)
    if "novelty_score" not in master.columns:
        master["novelty_score"] = np.nan

    qualified_master, qualification_records = qualify_candidates(request, master)
    qualification_usage = qualified_master.attrs.get("llm_usage", {}) if hasattr(qualified_master, "attrs") else {}
    request_mode = getattr(request, "request_mode", "generic")
    route_ids = {
        "request": set(request_route["source_id"].astype(str).tolist()) if not request_route.empty else set(),
        "history": set(history_route["source_id"].astype(str).tolist()) if not history_route.empty else set(),
        "full_watchlist": set(full_watchlist_route["source_id"].astype(str).tolist()) if not full_watchlist_route.empty else set(),
        "sparse_watchlist": set(sparse_watchlist_route["source_id"].astype(str).tolist()) if not sparse_watchlist_route.empty else set(),
        "empty_watchlist": set(empty_watchlist_route["source_id"].astype(str).tolist()) if not empty_watchlist_route.empty else set(),
    }

    scenarios: dict[str, Any] = {}
    for scenario_name, watchlist_ids in {
        "full_watchlist": route_ids["full_watchlist"],
        "empty_watchlist": route_ids["empty_watchlist"],
        "sparse_watchlist": route_ids["sparse_watchlist"],
    }.items():
        allowed_a = route_ids["request"].union(watchlist_ids)
        allowed_b = route_ids["request"].union(route_ids["history"]).union(watchlist_ids)
        subset_a = _selection_subset(qualified_master, allowed_a)
        subset_b = _selection_subset(qualified_master, allowed_b)
        selected_a, report_a = rank_variant_candidates(subset_a, variant="A", hitl_id=hitl_id, request_mode=request_mode, top_k=SLATE_SIZE)
        selected_b, report_b = rank_variant_candidates(subset_b, variant="B", hitl_id=hitl_id, request_mode=request_mode, top_k=SLATE_SIZE)
        scenarios[scenario_name] = {
            "eligible_counts": {"A": int(report_a["eligible_count"]), "B": int(report_b["eligible_count"])},
            "A": _selected_payload(selected_a, subset_a),
            "B": _selected_payload(selected_b, subset_b),
        }

    prompt_result = {
        "status": "ok",
        "hitl_id": hitl_id,
        "prompt": prompt,
        "parsed_request": request.model_dump(),
        "controlled_request_spec": request.spec.model_dump() if request.spec else {},
        "corrections": corrections,
        "route_reports": {
            "request_external": request_report,
            "history_external": history_report,
            "watchlist_full": watchlist_report_full,
            "watchlist_sparse": watchlist_report_sparse,
            "watchlist_empty": watchlist_report_empty,
            "embedding_report": embed_report,
            "request_relevance_report": relevance_report,
            "qualification_llm_usage": qualification_usage,
            "qualification_records": [record.model_dump() if hasattr(record, "model_dump") else record for record in qualification_records],
        },
        "scenarios": scenarios,
        "llm_usage": qualification_usage,
    }
    return prompt_result


def _selected_payload(selected: pd.DataFrame, subset: pd.DataFrame) -> dict[str, Any]:
    selected_ids = selected["source_id"].astype(str).tolist() if not selected.empty else []
    selected_titles = selected["title"].astype(str).tolist() if not selected.empty else []
    return {
        "selected": selected,
        "selected_ids": selected_ids,
        "selected_titles": selected_titles,
        "candidate_ids": subset["source_id"].astype(str).tolist() if not subset.empty else [],
        "eligible_count": int(len(subset)),
    }


def build_technical_report(prompt_results: list[dict[str, Any]], summary: dict[str, Any], manifest: dict[str, Any], human_sheet: pd.DataFrame, unblinding_key: pd.DataFrame) -> str:
    lines = [
        "# Reversible History + Watchlist Variant",
        "",
        "## Status",
        f"- `status`: `{summary['status']}`",
        f"- prompts evaluated: `{summary['prompt_count']}`",
        f"- LLM calls: `{summary['llm_calls']}`",
        f"- token usage: `{summary['token_usage']}`",
        "",
        "## Verification",
        "- Exact H01-H12 prompts were loaded from the frozen runtime-v4 manifest source prompt set.",
        "- The H11 wording retained the La Bola Negra / Penelope Cruz reference.",
        "- The frozen 393/99 split and aligned training embeddings were verified before execution.",
        "- Request-driven external discovery, history-driven external discovery, and watchlist retrieval were all evaluated in isolation and in combination.",
        "- Qualification judgments were shared across scenarios for identical film candidates.",
        "",
        "## Competing Choices",
        f"- Full watchlist prompts with competing A/B outputs: `{summary['full_watchlist_pairs_with_competing_choices']}` / `{summary['prompt_count']}`",
        f"- Empty watchlist prompts with competing A/B outputs: `{summary['empty_watchlist_pairs_with_competing_choices']}` / `{summary['prompt_count']}`",
        "",
        "## Human Sheet",
        f"- rows: `{len(human_sheet)}`",
        f"- unblinding rows: `{len(unblinding_key)}`",
        "",
        "## Limitations",
        "- History improves candidate generation only when TMDB returns viable seed expansions.",
        "- Some prompts may still collapse to the same slate after qualification; that is recorded as a result, not tuned away.",
        "- The new variant is isolated from the frozen corrected run and does not change the production default.",
    ]
    return "\n".join(lines)


def main() -> None:
    global DEFAULT_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Run the reversible history-plus-watchlist variant.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    DEFAULT_OUTPUT_DIR = Path(args.output_dir)
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
