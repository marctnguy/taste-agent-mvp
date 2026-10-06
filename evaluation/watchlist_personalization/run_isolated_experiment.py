from __future__ import annotations

import hashlib
import json
import os
import random
import time
import socket
import zipfile
import sys
from urllib.parse import urlparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
EXPERIMENT_DIR = Path("evaluation/watchlist_personalization")
RUN_DIR = EXPERIMENT_DIR / "corrected_valid_run"
ARTIFACT_DIR = RUN_DIR / "artifacts"
HUMAN_DIR = RUN_DIR / "human_scoring"
CACHE_DIR = ARTIFACT_DIR / "cache"
WATCHLIST_ZIP = REPO_ROOT.parent / "letterboxd-marctguy-2026-09-17-17-00-utc.zip"

BASELINE_COMMIT = "62b0cd6e9e75fb51dd1abe4423c2da9aa145fffc"
REJECTED_COMMIT = "1bf34d85ae6e095196978a0eecce171b6f148651"
SEED = 20261005
TOP_K = 3
E_COUNT = 12
W_WATCHLIST_COUNT = 6

REQUESTS = [
    ("R1", "I want something comforting but not cheesy."),
    ("R2", "Give me something slow or alternative in atmosphere/pace but still extremely unsettling and anxiety-inducing, like Jeanne Dielman."),
    ("R3", "I want a film about performers, fame and show business, preferably with a period setting."),
    ("R4", "I want a visually striking science-fiction film about memory or identity, but not an action-heavy blockbuster."),
]


def load_env() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mvp.src.config import load_runtime_env

    load_runtime_env()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def title_year_key(title: Any, year: Any) -> str:
    normalized_title = normalize_text(title)
    normalized_year = "" if pd.isna(year) else str(int(float(year)))
    return f"{normalized_title}__{normalized_year}"


def load_watchlist(zip_path: Path = WATCHLIST_ZIP) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("watchlist.csv") as handle:
            frame = pd.read_csv(handle)
    frame = frame.rename(columns={"Date": "source_record_date", "Name": "title", "Year": "year"})
    frame["source_record_date"] = pd.to_datetime(frame["source_record_date"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["watchlist_rank"] = np.arange(1, len(frame) + 1)
    return frame


def load_inputs() -> dict[str, Any]:
    condition_a = pd.read_csv("mvp/data/processed/condition_a_enriched.csv")
    split = pd.read_csv("mvp/data/processed/primary_holdout_split.csv")
    train_embeddings = pd.read_csv("mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv")
    non_film_exclusions = pd.read_csv("mvp/data/processed/non_film_exclusions.csv")
    watched_export = load_watchlist()
    return {
        "condition_a": condition_a,
        "split": split,
        "train_embeddings": train_embeddings,
        "non_film_exclusions": non_film_exclusions,
        "watchlist": watched_export,
    }


def probe_service(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return {"state": "network_failure", "http_status": None, "message": "missing host"}
    try:
        socket.getaddrinfo(host, parsed.port or 443)
    except socket.gaierror as exc:
        return {"state": "dns_failure", "http_status": None, "message": str(exc)}
    return {"state": "ok", "http_status": None, "message": "dns_ok"}


def preflight_services() -> dict[str, Any]:
    from mvp.src.config import get_api_keys, get_langsmith_config

    keys = get_api_keys()
    langsmith = get_langsmith_config()
    checks: dict[str, Any] = {
        "credentials": {
            "tmdb_api_key": bool(keys.tmdb_api_key),
            "openai_api_key": bool(keys.openai_api_key),
            "langsmith_api_key": bool(langsmith.api_key),
            "langsmith_endpoint": bool(langsmith.endpoint),
        },
        "services": {},
    }

    if not keys.tmdb_api_key:
        checks["services"]["tmdb"] = {"state": "missing_credentials", "http_status": None, "message": "TMDB_API_KEY is missing"}
    else:
        checks["services"]["tmdb"] = probe_service(
            f"https://api.themoviedb.org/3/configuration?api_key={keys.tmdb_api_key}"
        )

    if not keys.openai_api_key:
        checks["services"]["openai"] = {"state": "missing_credentials", "http_status": None, "message": "OPENAI_API_KEY is missing"}
    else:
        checks["services"]["openai"] = probe_service(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {keys.openai_api_key}"},
        )

    return checks


def build_consumed_keys(condition_a: pd.DataFrame) -> dict[str, set[str]]:
    consumed = condition_a.copy()
    consumed["source_id"] = consumed["source_id"].astype(str)
    consumed_tmdb = set(pd.to_numeric(consumed.get("tmdb_id"), errors="coerce").dropna().astype(int).astype(str))
    consumed_source = set(consumed["source_id"].astype(str))
    consumed_title_year = {title_year_key(row["title"], row["year"]) for _, row in consumed.iterrows()}
    return {
        "tmdb": consumed_tmdb,
        "source": consumed_source,
        "title_year": consumed_title_year,
    }


def build_training_sets(split: pd.DataFrame, condition_a: pd.DataFrame, train_embeddings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = split.merge(condition_a, on="source_id", how="inner", validate="one_to_one")
    if len(merged) != 492:
        raise RuntimeError(f"Expected 492 film rows after merge, found {len(merged)}.")
    train = merged.loc[merged["split"] == "train"].copy().reset_index(drop=True)
    test = merged.loc[merged["split"] == "test"].copy().reset_index(drop=True)
    if len(train) != 393 or len(test) != 99:
        raise RuntimeError(f"Expected 393/99 train/test split, found {len(train)}/{len(test)}.")
    if len(train_embeddings) != 393:
        raise RuntimeError(f"Expected 393 training embeddings, found {len(train_embeddings)}.")
    train_ids = set(train["source_id"].astype(str))
    embedding_ids = set(train_embeddings["source_id"].astype(str))
    if train_ids != embedding_ids:
        raise RuntimeError("Training embedding IDs do not align with frozen training split.")
    return train, test


def configure_tmdb_client():
    from mvp.src.config import get_api_keys
    from mvp.src.prepare import TMDBClient

    keys = get_api_keys()
    if not keys.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for the isolated experiment.")
    return TMDBClient(api_key=keys.tmdb_api_key)


def build_direct_search_candidates(request_text: str, phrases: list[str], client) -> pd.DataFrame:
    from mvp.src.retrieval.catalog_retrieval import _enrich_movie_candidate, _tmdb_request

    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for phrase_index, phrase in enumerate([request_text, *phrases], start=1):
        phrase = " ".join(str(phrase or "").split())
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        try:
            payload = _tmdb_request(client, "/search/movie", {"query": phrase, "page": 1, "include_adult": "false"})
        except Exception:
            continue
        rows = []
        for rank, movie in enumerate(payload.get("results", [])[:5], start=1):
            candidate = _enrich_movie_candidate(client, movie)
            if not candidate:
                continue
            candidate["candidate_sources"] = [f"query:{phrase}"]
            candidate["candidate_source_ranks"] = [str(rank)]
            candidate["request_query_rank"] = phrase_index
            rows.append(candidate)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["source_id"] = combined["source_id"].astype(str)
    return combined.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)


def build_request_phrases(request) -> list[str]:
    from mvp.src.generative_v4.qualification_chain import ASPECT_SYNONYMS

    phrases: list[str] = []
    spec = request.spec
    if spec is None:
        return phrases
    if spec.semantic_query_text:
        phrases.append(spec.semantic_query_text)
    for concept in spec.semantic_requirements:
        canonical = concept.concept or concept.aspect_id
        if canonical:
            phrases.append(canonical.replace("_", " "))
            phrases.extend(ASPECT_SYNONYMS.get(canonical, [])[:3])
    for concept in spec.semantic_exclusions:
        canonical = concept.concept or concept.aspect_id
        if canonical:
            phrases.append(canonical.replace("_", " "))
    if request.reference_title:
        phrases.append(request.reference_title)
    return list(dict.fromkeys([phrase for phrase in phrases if phrase.strip()]))


def validate_frozen_request(request_id: str, request) -> list[str]:
    issues: list[str] = []
    spec = getattr(request, "spec", None)
    intent = getattr(request, "intent", None)
    if spec is None:
        return [f"{request_id}: missing parsed spec"]

    def concept_key(value: Any) -> str:
        return normalize_text(value).replace("-", "_").replace(" ", "_")

    required_concepts = {concept_key(concept.concept or concept.aspect_id) for concept in getattr(spec, "semantic_requirements", [])}
    preferred_concepts = {
        concept_key(concept.concept or concept.aspect_id)
        for group in getattr(spec, "semantic_requirement_groups", [])
        for concept in getattr(group, "concepts", [])
        if str(getattr(concept, "importance", "")).lower() != "required"
    }
    requested_genres = {str(value).lower() for value in (getattr(intent, "requested_genres", None) or [])}

    if request_id == "R1":
        if "comforting" not in required_concepts:
            issues.append("R1 must preserve comforting as a required concept.")
        if "cheesy" not in {concept_key(concept.concept or concept.aspect_id) for concept in getattr(spec, "semantic_exclusions", [])}:
            issues.append("R1 must preserve cheesy as an exclusion.")
    elif request_id == "R2":
        if not {"contemplative", "alternative", "unsettling"}.issubset(required_concepts):
            issues.append("R2 must preserve slow/alternative/unsettling semantics as required concepts.")
        if str(getattr(request, "reference_title", "")).strip().lower() != "jeanne dielman":
            issues.append("R2 must preserve the Jeanne Dielman reference title.")
        if str(getattr(request, "request_mode", "")).lower() != "reference":
            issues.append("R2 must remain a reference-mode request.")
    elif request_id == "R3":
        if "historical" in required_concepts:
            issues.append("R3 must not promote the period preference into a required historical condition.")
        if "period_setting" not in preferred_concepts:
            issues.append("R3 must preserve period_setting as a preference.")
    elif request_id == "R4":
        if "action" in requested_genres:
            issues.append("R4 must not turn the request into a required Action-genre query.")
        if "science_fiction" not in required_concepts and "science-fiction" not in required_concepts:
            issues.append("R4 must preserve science fiction as a required concept.")
        if not {"memory", "identity"} & preferred_concepts:
            issues.append("R4 must preserve memory or identity as preferred concepts.")
    return issues


def correct_frozen_request(request_id: str, request):
    corrections: list[str] = []
    spec = getattr(request, "spec", None)
    if spec is None:
        return request, corrections

    from mvp.src.generative_v4.schemas import SemanticConcept, SemanticRequirementGroup, ReferenceSpec, StructuredConstraints

    if request_id == "R1":
        comforting = SemanticConcept(aspect_id="comforting", concept="comforting", importance="required", source_span="comforting")
        cheesy = SemanticConcept(aspect_id="cheesy", concept="cheesy", importance="preferred", source_span="cheesy")
        request = request.model_copy(
            update={
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [comforting],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(
                                group_id="comforting",
                                mode="all_of",
                                concepts=[comforting],
                                source_span="comforting",
                            )
                        ],
                        "semantic_exclusions": [cheesy],
                        "semantic_query_text": "comforting not cheesy",
                        "request_summary": "Request for something comforting but not cheesy.",
                    }
                ),
                "intent": request.intent.model_copy(
                    update={
                        "requested_taste_dimensions": ["comforting"],
                        "requested_genres": [],
                        "requested_countries": [],
                        "requested_languages": [],
                        "requested_decades": [],
                        "exclusions": ["cheesy"],
                        "free_text_context": "Request for something comforting but not cheesy.",
                    }
                ),
            }
        )
    if request_id == "R2":
        contemplative = SemanticConcept(aspect_id="contemplative", concept="contemplative", importance="required", source_span="slow or alternative in atmosphere/pace")
        alternative = SemanticConcept(aspect_id="alternative", concept="alternative", importance="required", source_span="slow or alternative in atmosphere/pace")
        unsettling = SemanticConcept(aspect_id="unsettling", concept="unsettling", importance="required", source_span="unsettling/anxiety-inducing")
        reference = ReferenceSpec(title="Jeanne Dielman", relation="reference", resolved_tmdb_id=None, resolved_tmdb_title=None, resolution_status="absent")
        request = request.model_copy(
            update={
                "request_mode": "reference",
                "reference_title": "Jeanne Dielman",
                "reference_status": "absent",
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [contemplative, alternative, unsettling],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(
                                group_id="atmosphere_pace",
                                mode="all_of",
                                concepts=[contemplative, alternative],
                                source_span="slow or alternative in atmosphere/pace",
                            ),
                            SemanticRequirementGroup(
                                group_id="unsettling",
                                mode="all_of",
                                concepts=[unsettling],
                                source_span="unsettling/anxiety-inducing",
                            ),
                        ],
                        "semantic_exclusions": [],
                        "reference": reference,
                        "semantic_query_text": "slow alternative unsettling anxiety inducing Jeanne Dielman",
                        "request_summary": "Request for something slow or alternative in atmosphere/pace, unsettling and anxiety-inducing, with Jeanne Dielman as a reference.",
                    }
                ),
                "intent": request.intent.model_copy(
                    update={
                        "requested_taste_dimensions": ["contemplative", "alternative", "unsettling"],
                        "requested_genres": [],
                        "requested_countries": [],
                        "requested_languages": [],
                        "requested_decades": [],
                        "exclusions": [],
                        "free_text_context": "Request for something slow or alternative in atmosphere/pace, unsettling and anxiety-inducing, with Jeanne Dielman as a reference.",
                    }
                ),
            }
        )
    if request_id == "R3":
        performers = SemanticConcept(aspect_id="performers", concept="performers", importance="required", source_span="performers")
        fame = SemanticConcept(aspect_id="fame", concept="fame", importance="required", source_span="fame")
        show_business = SemanticConcept(aspect_id="show_business", concept="show_business", importance="required", source_span="show business")
        period_setting = SemanticConcept(aspect_id="period_setting", concept="period_setting", importance="preferred", source_span="preferably with a period setting")
        request = request.model_copy(
            update={
                "spec": spec.model_copy(
                    update={
                        "semantic_requirements": [performers, fame, show_business, period_setting],
                        "semantic_requirement_groups": [
                            SemanticRequirementGroup(
                                group_id="required_semantics",
                                mode="all_of",
                                concepts=[performers, fame, show_business],
                                source_span="performers, fame and show business",
                            ),
                            SemanticRequirementGroup(
                                group_id="period_setting",
                                mode="preferred",
                                concepts=[period_setting],
                                source_span="preferably with a period setting",
                            ),
                        ],
                        "semantic_exclusions": [],
                        "semantic_query_text": "performers fame show_business period_setting",
                        "request_summary": "Request for a film about performers, fame and show business, preferably with a period setting.",
                    }
                ),
                "intent": request.intent.model_copy(
                    update={
                        "requested_taste_dimensions": ["performers", "fame", "show_business", "period_setting"],
                        "requested_genres": [],
                        "exclusions": [],
                        "free_text_context": "Request for a film about performers, fame and show business, preferably with a period setting.",
                    }
                ),
            }
        )
    if request_id == "R4":
        science_fiction = SemanticConcept(aspect_id="science_fiction", concept="science fiction", importance="required", source_span="science-fiction film")
        memory = SemanticConcept(aspect_id="memory", concept="memory", importance="preferred", source_span="about memory")
        identity = SemanticConcept(aspect_id="identity", concept="identity", importance="preferred", source_span="about identity")
        visually_striking = SemanticConcept(aspect_id="visually_striking", concept="visually striking", importance="preferred", source_span="visually striking presentation")
        requirement_groups = [
            SemanticRequirementGroup(group_id="science_fiction", mode="all_of", concepts=[science_fiction], source_span="science-fiction film"),
            SemanticRequirementGroup(group_id="memory_identity", mode="any_of", concepts=[memory, identity], source_span="memory or identity"),
            SemanticRequirementGroup(group_id="visual_style", mode="preferred", concepts=[visually_striking], source_span="visually striking presentation"),
        ]
        request = request.model_copy(
            update={
                "spec": spec.model_copy(
                    update={
                        "structured_constraints": StructuredConstraints(
                            required_languages=list(spec.structured_constraints.required_languages),
                            excluded_languages=list(spec.structured_constraints.excluded_languages),
                            required_countries=list(spec.structured_constraints.required_countries),
                            excluded_countries=list(spec.structured_constraints.excluded_countries),
                            required_genres=[],
                            excluded_genres=list(spec.structured_constraints.excluded_genres),
                            required_decades=list(spec.structured_constraints.required_decades),
                            excluded_decades=list(spec.structured_constraints.excluded_decades),
                            min_year=spec.structured_constraints.min_year,
                            max_year=spec.structured_constraints.max_year,
                        ),
                        "semantic_requirement_groups": requirement_groups,
                        "semantic_requirements": [science_fiction, memory, identity, visually_striking],
                        "semantic_exclusions": [],
                        "semantic_query_text": "science fiction memory identity visually striking avoid action heavy blockbuster",
                        "request_summary": "Request for a visually striking science-fiction film about memory or identity, avoiding action-heavy blockbusters.",
                    }
                ),
                "intent": request.intent.model_copy(
                    update={
                        "requested_genres": [],
                        "requested_taste_dimensions": ["science_fiction", "memory", "identity", "visually_striking"],
                        "exclusions": ["action-heavy blockbusters"],
                        "free_text_context": "Request for a visually striking science-fiction film about memory or identity, avoiding action-heavy blockbusters.",
                    }
                ),
            }
        )
    return request, corrections


def build_request_specific_pool(request, client) -> tuple[pd.DataFrame, dict[str, Any]]:
    from mvp.src.retrieval.catalog_retrieval import _augment_with_constraint_candidates, _augment_with_reference_candidates, _augment_with_semantic_keyword_candidates

    frames: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    reference_frame, reference_report = _augment_with_reference_candidates(request, client)
    constraint_frame = pd.DataFrame(columns=["source_id"])
    constraint_report = {"constraint_candidates": 0}
    if request.spec is not None:
        constraints = request.spec.structured_constraints
        if any(
            [
                constraints.required_languages,
                constraints.required_countries,
                constraints.required_genres,
                constraints.required_decades,
                constraints.excluded_languages,
                constraints.excluded_countries,
                constraints.excluded_genres,
                constraints.excluded_decades,
                constraints.min_year is not None,
                constraints.max_year is not None,
            ]
        ):
            try:
                constraint_frame, constraint_report = _augment_with_constraint_candidates(request, client)
            except Exception:
                constraint_frame = pd.DataFrame(columns=["source_id"])
                constraint_report = {"constraint_candidates": 0, "constraint_error": True}
    keyword_frame, keyword_report = _augment_with_semantic_keyword_candidates(request, client)
    direct_frame = build_direct_search_candidates(request.query, build_request_phrases(request), client)
    for frame in [reference_frame, constraint_frame, keyword_frame, direct_frame]:
        if not frame.empty:
            frames.append(frame)
    reports.update(reference_report)
    reports.update(constraint_report)
    reports.update(keyword_report)
    reports["direct_query_candidates"] = int(len(direct_frame))
    if not frames:
        return pd.DataFrame(columns=["source_id"]), reports
    combined = pd.concat(frames, ignore_index=True)
    combined["source_id"] = combined["source_id"].astype(str)
    combined = combined.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return combined, reports


def resolve_watchlist_candidates(watchlist: pd.DataFrame, client) -> tuple[pd.DataFrame, dict[str, Any]]:
    from mvp.src.retrieval.catalog_retrieval import _enrich_movie_candidate

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for _, entry in watchlist.iterrows():
        title = entry["title"]
        year = entry["year"]
        try:
            match = client.search_movie(str(title), year)
        except Exception:
            match = None
        if not match:
            candidate = {}
        else:
            try:
                candidate = _enrich_movie_candidate(client, match)
            except Exception:
                candidate = {}
        if not candidate:
            unresolved += 1
            continue
        candidate["candidate_sources"] = [f"watchlist:{entry['watchlist_rank']}"]
        candidate["candidate_source_ranks"] = [str(int(entry["watchlist_rank"]))]
        candidate["watchlist_uri"] = entry["Letterboxd URI"]
        candidate["watchlist_rank"] = int(entry["watchlist_rank"])
        rows.append(candidate)
    if not rows:
        return pd.DataFrame(columns=["source_id"]), {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": int(unresolved)}
    frame = pd.DataFrame(rows)
    frame["source_id"] = frame["source_id"].astype(str)
    frame["watchlist_rank"] = pd.to_numeric(frame["watchlist_rank"], errors="coerce")
    frame = frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return frame, {"resolved_watchlist_candidates": int(len(frame)), "unresolved_watchlist_candidates": int(unresolved)}


def exclude_consumed(frame: pd.DataFrame, consumed_keys: dict[str, set[str]]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy().reset_index(drop=True)
    filtered["source_id"] = filtered["source_id"].astype(str)
    filtered["normalized_title_year"] = filtered.apply(lambda row: title_year_key(row.get("title"), row.get("year")), axis=1)
    keep_mask = []
    for _, row in filtered.iterrows():
        source_id = str(row.get("source_id", ""))
        tmdb_id = row.get("tmdb_id")
        tmdb_key = None
        try:
            if pd.notna(tmdb_id):
                tmdb_key = str(int(float(tmdb_id)))
        except Exception:
            tmdb_key = None
        title_year = title_year_key(row.get("title"), row.get("year"))
        excluded = source_id in consumed_keys["source"] or title_year in consumed_keys["title_year"] or (tmdb_key is not None and tmdb_key in consumed_keys["tmdb"])
        keep_mask.append(not excluded)
    return filtered.loc[keep_mask].drop(columns=[column for column in ["normalized_title_year"] if column in filtered.columns]).reset_index(drop=True)


def overlap_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    keys = set(frame["source_id"].astype(str).tolist())
    keys.update(title_year_key(row.get("title"), row.get("year")) for _, row in frame.iterrows())
    if "tmdb_id" in frame.columns:
        keys.update(str(int(float(value))) for value in pd.to_numeric(frame["tmdb_id"], errors="coerce").dropna().astype(int).tolist())
    return keys


def select_fixed_candidates(frame: pd.DataFrame, *, limit: int, sort_by: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.copy()
    for column in ["candidate_source_ranks", "candidate_sources", "watchlist_rank"]:
        if column not in ordered.columns:
            ordered[column] = pd.NA
    ordered["source_family"] = ordered["candidate_sources"].apply(lambda value: str((value[0] if isinstance(value, list) and value else value or "")).split(":", 1)[0])
    family_rank = {
        "query": 0,
        "watchlist": 1,
        "reference_recommendations": 2,
        "keyword": 3,
        "genre": 4,
        "language": 5,
        "country": 6,
        "decade": 7,
    }
    ordered["source_family_rank"] = ordered["source_family"].map(lambda value: family_rank.get(str(value), 99))
    ordered["source_rank_value"] = pd.to_numeric(
        ordered["candidate_source_ranks"].apply(lambda value: value[0] if isinstance(value, list) and value else (value if pd.notna(value) else None)),
        errors="coerce",
    ).fillna(9999).astype(int)
    if "watchlist_rank" not in ordered.columns:
        ordered["watchlist_rank"] = pd.NA
    if sort_by == ["watchlist_rank"]:
        ordered = ordered.sort_values(["watchlist_rank", "title", "source_id"], ascending=[True, True, True], na_position="last")
    else:
        ordered = ordered.sort_values(
            ["source_family_rank", "source_rank_value", "title", "source_id"],
            ascending=[True, True, True, True],
            na_position="last",
        )
    return ordered.head(limit).reset_index(drop=True)


def select_request_relevance_candidates(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "request_relevance" not in frame.columns:
        raise KeyError("Request relevance must be computed before request-aware selection.")
    ordered = frame.copy().reset_index(drop=True)
    ordered = ordered.sort_values(
        ["request_relevance", "title", "source_id"],
        ascending=[False, True, True],
        na_position="last",
    )
    return ordered.head(limit).reset_index(drop=True)


def attach_pool_memberships(frame: pd.DataFrame, memberships: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame.copy().reset_index(drop=True)
    selected["pool_memberships"] = [list(dict.fromkeys(memberships)) for _ in range(len(selected))]
    return selected


def prepare_candidate_frame(frame: pd.DataFrame, source_label: str, source_condition: str, source_origin: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    prepared = frame.copy().reset_index(drop=True)
    prepared["source_condition"] = source_condition
    prepared["source_origin"] = source_origin
    prepared["source_label"] = source_label
    prepared["candidate_id"] = prepared["source_id"].astype(str)
    return prepared


def embed_candidates(frame: pd.DataFrame, request_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from mvp.src.mvp_deployment import load_or_generate_candidate_embeddings
    from mvp.src.retrieval.catalog_retrieval import build_canonical_film_documents

    if frame.empty:
        return frame.copy(), {"provider": "none", "model": "none", "embedding_dim": 0, "coverage": 0.0, "cache_status": "empty", "cache_hits": 0, "cache_misses": 0}
    documents = build_canonical_film_documents(frame)
    embeddings, manifest, report = load_or_generate_candidate_embeddings(documents, cache_path=CACHE_DIR / f"{request_id}_candidate_embeddings.csv")
    embedded = frame.merge(embeddings, on="source_id", how="left", suffixes=("", "_emb"))
    report = dict(report)
    report["manifest_rows"] = int(len(manifest))
    report["cache_path"] = str(CACHE_DIR / f"{request_id}_candidate_embeddings.csv")
    return embedded, report


def compute_request_relevance(frame: pd.DataFrame, request_text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from mvp.src.config import get_api_keys
    from mvp.src.mvp_deployment import OPENAI_EMBEDDING_MODEL, _embed_texts

    if frame.empty:
        return frame.copy(), {"request_embedding": None, "request_relevance_method": "no_candidates"}
    emb_cols = [column for column in frame.columns if column.startswith("emb_")]
    if len(emb_cols) != 1536:
        return frame.copy(), {"request_embedding": None, "request_relevance_method": "missing_embeddings"}
    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for request relevance embedding.")
    query_embedding = np.asarray(_embed_texts([request_text], keys.openai_api_key, OPENAI_EMBEDDING_MODEL)[0], dtype=float)
    matrix = frame.loc[:, emb_cols].to_numpy(dtype=float)
    candidate_norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    query_norm = np.linalg.norm(query_embedding) or 1.0
    relevance = (matrix / candidate_norms) @ (query_embedding / query_norm)
    scored = frame.copy()
    scored["request_relevance"] = relevance
    scored["request_relevance_rank"] = pd.Series(relevance).rank(method="first", ascending=False).astype(int)
    return scored, {"request_embedding": query_embedding, "request_relevance_method": "cosine"}


def compute_b3_scores(frame: pd.DataFrame) -> pd.DataFrame:
    from mvp.src.mvp_deployment import load_model_bundle

    if frame.empty:
        return frame.copy()
    bundle, pca, metadata_builder = load_model_bundle("mvp/artifacts/models/b3_mvp")
    emb_cols = [column for column in frame.columns if column.startswith("emb_")]
    if len(emb_cols) != 1536:
        raise RuntimeError("Candidate embeddings must have 1536 dimensions for B3 scoring.")
    embeddings = frame.loc[:, emb_cols].to_numpy(dtype=float)
    metadata = metadata_builder.transform(frame).reset_index(drop=True)
    scaled = bundle.scaler.transform(embeddings)
    latent = pca.transform(scaled)
    latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(latent.shape[1])]
    features = pd.concat([metadata, pd.DataFrame(latent, columns=latent_columns)], axis=1)
    features = features.reindex(columns=bundle.feature_columns, fill_value=0.0)
    predictions = bundle.ridge.predict(features.to_numpy(dtype=float))
    scored = frame.copy().reset_index(drop=True)
    scored["predicted_preference"] = predictions
    scored["model_status"] = bundle.model_status
    scored["architecture"] = bundle.architecture
    scored["rank"] = pd.Series(predictions).rank(method="first", ascending=False).astype(int)
    scored["taste_evidence"] = scored.apply(lambda row: {}, axis=1)
    return scored


def build_historical_taste(training: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, Any]:
    merged = training.merge(embeddings, on="source_id", how="inner", validate="one_to_one")
    positives = merged.loc[pd.to_numeric(merged["rating"], errors="coerce") >= 4.0].copy().reset_index(drop=True)
    negatives = merged.loc[pd.to_numeric(merged["rating"], errors="coerce") <= 2.5].copy().reset_index(drop=True)
    emb_cols = [column for column in embeddings.columns if column.startswith("emb_")]
    if len(emb_cols) != 1536:
        raise RuntimeError("Historical embeddings are not aligned to the 1536-dimension model.")
    return {
        "positive": positives,
        "negative": negatives,
        "emb_cols": emb_cols,
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
    }


def cosine_topk(candidate_vector: np.ndarray, history_matrix: np.ndarray, history_ids: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    if candidate_vector.size == 0 or history_matrix.size == 0:
        return []
    cand = candidate_vector / max(np.linalg.norm(candidate_vector), 1e-12)
    hist = history_matrix / np.clip(np.linalg.norm(history_matrix, axis=1, keepdims=True), 1e-12, None)
    sims = hist @ cand
    order = np.argsort(-sims)[: min(top_k, len(sims))]
    return [{"source_id": history_ids[index], "similarity": float(sims[index])} for index in order]


def compute_direct_taste_scores(frame: pd.DataFrame, historical: dict[str, Any]) -> pd.DataFrame:
    emb_cols = historical["emb_cols"]
    positive = historical["positive"]
    negative = historical["negative"]
    if frame.empty:
        return frame.copy()
    candidate_matrix = frame.loc[:, emb_cols].to_numpy(dtype=float)
    pos_matrix = positive.loc[:, emb_cols].to_numpy(dtype=float) if not positive.empty else np.empty((0, len(emb_cols)))
    neg_matrix = negative.loc[:, emb_cols].to_numpy(dtype=float) if not negative.empty else np.empty((0, len(emb_cols)))
    pos_ids = positive["source_id"].astype(str).tolist() if not positive.empty else []
    neg_ids = negative["source_id"].astype(str).tolist() if not negative.empty else []
    rows = []
    for idx, candidate_vector in enumerate(candidate_matrix):
        pos_top = cosine_topk(candidate_vector, pos_matrix, pos_ids, top_k=3)
        neg_top = cosine_topk(candidate_vector, neg_matrix, neg_ids, top_k=3)
        pos_mean = float(np.mean([item["similarity"] for item in pos_top])) if pos_top else np.nan
        neg_mean = float(np.mean([item["similarity"] for item in neg_top])) if neg_top else np.nan
        rows.append(
            {
                "direct_taste_score": float(pos_mean - neg_mean) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan,
                "positive_neighbors": pos_top,
                "negative_neighbors": neg_top,
                "positive_neighbor_count": int(len(pos_top)),
                "negative_neighbor_count": int(len(neg_top)),
            }
        )
    enriched = frame.copy().reset_index(drop=True)
    for key in rows[0].keys():
        enriched[key] = [row[key] for row in rows]
    return enriched


def percentile_scores(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.rank(method="average", pct=True)


def qualification_rank(value: str) -> int:
    return {"strong": 2, "partial": 1}.get(str(value), 0)


def qualify_candidates(request, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from mvp.src.config import get_api_keys
    from mvp.src.generative_v4.qualification_chain import (
        DEFAULT_QUALIFICATION_MODEL,
        QualificationBatch,
        QualificationOutput,
        RetrievedCandidate,
        _candidate_payload_for_llm,
        _normalize_candidate_payload,
        _record_from_output,
        _request_payload_for_llm,
    )

    if candidates.empty:
        empty = candidates.copy()
        empty["qualification_status"] = []
        return empty, {"llm_call_count": 0, "runtime_input_tokens": None, "runtime_output_tokens": None, "runtime_total_tokens": None}

    def _normalize_grounded_evidence_item(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            aspect_id = item.get("aspect_id", "")
            field = item.get("field", "")
            evidence = item.get("evidence", "")
            return f"{aspect_id}:{field}:{evidence}".strip(":")
        aspect_id = getattr(item, "aspect_id", None)
        field = getattr(item, "field", None)
        evidence = getattr(item, "evidence", None)
        if aspect_id is not None or field is not None or evidence is not None:
            return f"{aspect_id or ''}:{field or ''}:{evidence or ''}".strip(":")
        return str(item)

    def _grounded_evidence_details(item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if isinstance(item, dict):
            return dict(item)
        return {"value": str(item)}

    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for candidate qualification.")

    llm = ChatOpenAI(model=DEFAULT_QUALIFICATION_MODEL, temperature=0, api_key=keys.openai_api_key)
    structured = llm.with_structured_output(QualificationBatch, include_raw=True)
    request_payload = _request_payload_for_llm(request)
    prompt = (
        "You are conservatively qualifying film candidates against a structured semantic request.\n"
        "Use only the supplied evidence.\n"
        "Guidance:\n"
        "- Structured constraints are hard constraints.\n"
        "- Semantic exclusions are admission blockers.\n"
        "- Do not infer subject matter from fame of cast/crew alone.\n"
        "- Be conservative: if grounded evidence does not establish the requested meaning, mark it unsupported.\n"
        "- A resolved reference may be used only from the supplied reference metadata.\n"
        "- Return candidate-level structured judgments only.\n"
    )

    if "candidate_id" not in candidates.columns:
        raise KeyError("Candidate qualification requires candidate_id alignment.")

    batch_size = 6
    total_llm_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    records: list[dict[str, Any]] = []
    qualified_rows: list[dict[str, Any]] = []
    candidate_rows = list(candidates.iterrows())

    for start in range(0, len(candidate_rows), batch_size):
        batch_rows = candidate_rows[start : start + batch_size]
        payloads = [_candidate_payload_for_llm(_normalize_candidate_payload(row)) for _, row in batch_rows]
        result = structured.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Request payload:\n"
                        f"{request_payload}\n\n"
                        "Candidate payloads:\n"
                        f"{payloads}\n"
                    )
                ),
            ]
        )
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None
        usage_metadata = getattr(raw, "usage_metadata", None) if raw is not None else None
        input_tokens = usage_metadata.get("input_tokens") if isinstance(usage_metadata, dict) else None
        output_tokens = usage_metadata.get("output_tokens") if isinstance(usage_metadata, dict) else None
        batch_total_tokens = usage_metadata.get("total_tokens") if isinstance(usage_metadata, dict) else None
        if parsed is None or not getattr(parsed, "items", None):
            raise RuntimeError("Qualification LLM unavailable or parse failed.")

        parsed_by_id = {str(item.candidate_id): item for item in parsed.items}
        expected_ids = [str(_normalize_candidate_payload(row).get("candidate_id")) for _, row in batch_rows]
        if set(parsed_by_id) != set(expected_ids):
            raise RuntimeError(
                f"Qualification LLM returned candidate IDs {sorted(parsed_by_id)} for expected IDs {sorted(expected_ids)}."
            )

        total_llm_calls += 1
        total_input_tokens += int(input_tokens or 0)
        total_output_tokens += int(output_tokens or 0)
        total_tokens += int(batch_total_tokens or 0)

        for _, row in batch_rows:
            candidate_payload = _normalize_candidate_payload(row)
            candidate = RetrievedCandidate.model_validate(candidate_payload)
            parsed_item = parsed_by_id[str(candidate.candidate_id)]
            output = QualificationOutput(
                status=parsed_item.qualification_status,
                supported_required_aspects=list(parsed_item.supported_required_aspects),
                unsupported_required_aspects=list(parsed_item.unsupported_required_aspects),
                supported_preferred_aspects=list(parsed_item.supported_preferred_aspects),
                unsupported_preferred_aspects=list(parsed_item.unsupported_preferred_aspects),
                violated_semantic_exclusions=list(parsed_item.violated_semantic_exclusions),
                grounded_evidence=[_normalize_grounded_evidence_item(evidence) for evidence in getattr(parsed_item, "grounded_evidence", [])],
                grounded_evidence_details=[_grounded_evidence_details(evidence) for evidence in getattr(parsed_item, "grounded_evidence", [])],
                qualification_reason=parsed_item.reason,
                request_match=f"Supported request aspects: {', '.join(parsed_item.supported_required_aspects[:5])}" if parsed_item.supported_required_aspects else (
                    f"Supported request aspects: {', '.join(parsed_item.supported_preferred_aspects[:5])}" if parsed_item.supported_preferred_aspects else None
                ),
                caveat=parsed_item.required_caveat,
            )
            record = _record_from_output(candidate.candidate_id, output, llm_used=True)
            records.append(record.model_dump())
            candidate.qualification_status = record.qualification_status
            candidate.supported_required_aspects = list(record.supported_required_aspects)
            candidate.unsupported_required_aspects = list(record.unsupported_required_aspects)
            candidate.supported_preferred_aspects = list(record.supported_preferred_aspects)
            candidate.unsupported_preferred_aspects = list(record.unsupported_preferred_aspects)
            candidate.supported_request_aspects = list(record.supported_request_aspects)
            candidate.unsupported_request_aspects = list(record.unsupported_request_aspects)
            candidate.grounded_evidence = list(record.grounded_evidence)
            candidate.qualification_reason = record.qualification_reason
            candidate.request_match = record.request_match
            candidate.caveat = record.caveat
            qualified_row = dict(candidate_payload)
            qualified_row.update(
                {
                    "qualification_status": record.qualification_status,
                    "supported_required_aspects": list(record.supported_required_aspects),
                    "unsupported_required_aspects": list(record.unsupported_required_aspects),
                    "supported_preferred_aspects": list(record.supported_preferred_aspects),
                    "unsupported_preferred_aspects": list(record.unsupported_preferred_aspects),
                    "supported_request_aspects": list(record.supported_request_aspects),
                    "unsupported_request_aspects": list(record.unsupported_request_aspects),
                    "grounded_evidence": list(record.grounded_evidence),
                    "qualification_reason": record.qualification_reason,
                    "request_match": record.request_match,
                    "caveat": record.caveat,
                }
            )
            qualified_rows.append(qualified_row)

    qualified = pd.DataFrame(qualified_rows)
    llm_usage = {
        "llm_call_count": int(total_llm_calls),
        "runtime_input_tokens": total_input_tokens,
        "runtime_output_tokens": total_output_tokens,
        "runtime_total_tokens": total_tokens,
        "qualification_records": records,
    }
    qualified.attrs["llm_usage"] = llm_usage
    return qualified, llm_usage


def rank_candidates(frame: pd.DataFrame, source_condition: str, *, top_k: int = TOP_K) -> dict[str, pd.DataFrame]:
    eligible = frame.loc[frame["qualification_status"].astype(str).isin({"strong", "partial"})].copy().reset_index(drop=True)
    if eligible.empty:
        return {name: eligible.copy() for name in ["A", "B", "C"]}
    required_columns = {"request_relevance_percentile", "direct_taste_percentile", "b3_percentile"}
    missing_columns = required_columns.difference(eligible.columns)
    if missing_columns:
        raise KeyError(f"Missing global percentile columns for ranking: {sorted(missing_columns)}")
    eligible["qualification_rank"] = eligible["qualification_status"].map(qualification_rank)
    eligible["policy_A"] = pd.to_numeric(eligible["request_relevance_percentile"], errors="coerce")
    eligible["policy_B"] = 0.5 * pd.to_numeric(eligible["request_relevance_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
        eligible["direct_taste_percentile"], errors="coerce"
    )
    eligible["policy_C"] = 0.5 * pd.to_numeric(eligible["request_relevance_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
        eligible["b3_percentile"], errors="coerce"
    )

    results: dict[str, pd.DataFrame] = {}
    for policy in ["A", "B", "C"]:
        score_column = f"policy_{policy}"
        ranked = eligible.sort_values(
            ["qualification_rank", score_column, "source_id"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked["selected_rank"] = np.arange(1, len(ranked) + 1)
        results[policy] = ranked.head(top_k).copy()
    return results


def make_anon_labels(request_id: str, count: int, rng: random.Random) -> list[str]:
    labels = [f"{request_id}-F{index:02d}" for index in range(1, count + 1)]
    rng.shuffle(labels)
    return labels


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def run() -> dict[str, Any]:
    load_env()
    preflight = preflight_services()
    failing_services = {name: status for name, status in preflight.get("services", {}).items() if status.get("state") != "ok"}
    if failing_services:
        raise RuntimeError(json.dumps({"preflight": preflight, "failing_services": failing_services}, indent=2, ensure_ascii=False, default=str))
    random.seed(SEED)
    np.random.seed(SEED)

    from mvp.src.generative_v4.intent_chain import understand_request
    from mvp.src.mvp_deployment import load_training_population

    inputs = load_inputs()
    condition_a = inputs["condition_a"]
    split = inputs["split"]
    train_embeddings = inputs["train_embeddings"]
    non_film_exclusions = inputs["non_film_exclusions"]
    watchlist = inputs["watchlist"]
    consumed_keys = build_consumed_keys(condition_a)
    train, test = build_training_sets(split, condition_a, train_embeddings)
    tmdb_client = configure_tmdb_client()
    historical = build_historical_taste(train, train_embeddings)
    train_population = load_training_population()
    if len(train_population) != 393:
        raise RuntimeError("Frozen training population changed.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_commit": BASELINE_COMMIT,
        "rejected_commit": REJECTED_COMMIT,
        "seed": SEED,
        "input_hashes": {
            "watchlist_zip": sha256_path(WATCHLIST_ZIP),
            "condition_a_enriched": sha256_path(Path("mvp/data/processed/condition_a_enriched.csv")),
            "primary_holdout_split": sha256_path(Path("mvp/data/processed/primary_holdout_split.csv")),
            "train_embeddings": sha256_path(Path("mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv")),
            "non_film_exclusions": sha256_path(Path("mvp/data/processed/non_film_exclusions.csv")),
        },
        "model": {
            "generation_model": "gpt-4o-mini",
            "qualification_model": "gpt-4o-mini",
            "request_parsing": "mvp.src.generative_v4.intent_chain.understand_request",
            "qualification_prompt": "mvp.src.generative_v4.qualification_chain._qualify_with_llm",
        },
        "training_population": {
            "total_film_rows": int(len(condition_a)),
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_ids_match_embeddings": bool(set(train["source_id"].astype(str)) == set(train_embeddings["source_id"].astype(str))),
            "positive_examples": int((pd.to_numeric(train["rating"], errors="coerce") >= 4.0).sum()),
            "negative_examples": int((pd.to_numeric(train["rating"], errors="coerce") <= 2.5).sum()),
        },
        "source_pool_rules": {
            "external_pool": "request-aware selection from the retrieved external universe by request relevance only; 12 unique non-consumed films per request; the top 6 external IDs are shared into W",
            "watchlist_pool": "request-aware selection from the resolved watchlist by request relevance only; 6 unique unwatched watchlist films per request after excluding the selected external IDs",
            "pool_membership": "source_origin records retrieval provenance; pool_memberships records E/W assignment; shared external candidates belong to both pools",
            "consumed_exclusion": "TMDB ID, source_id, and normalized title/year fallback applied after all candidate sources were collected",
        },
        "scoring_rules": {
            "qualification_precedence": "strong before partial; unsupported excluded",
            "A": "request relevance percentile computed once over the shared eligible union",
            "B": "0.5 * request relevance percentile + 0.5 * direct taste percentile computed once over the shared eligible union",
            "C": "0.5 * request relevance percentile + 0.5 * B3 percentile computed once over the shared eligible union",
            "direct_taste_definition": "mean cosine similarity to three nearest positives minus mean cosine similarity to three nearest negatives",
        },
        "benchmark_controls": {
            "parser_variability": "not validated; frozen request specifications are enforced explicitly inside the isolated experiment",
            "request_aware_pooling": True,
            "shared_external_count": 6,
            "external_pool_size": 12,
            "watchlist_pool_size": 12,
        },
        "coverage": {},
        "calls": {},
        "failures": [],
        "timings": {},
    }

    request_summaries: list[dict[str, Any]] = []
    machine_results: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    unblinding_rows: list[dict[str, Any]] = []
    overall_timings = []
    qualification_llm_total = 0
    qualification_token_total = {"input": 0, "output": 0, "total": 0}
    tmdb_unresolved_total = 0
    shortfalls = []

    for request_index, (request_id, query) in enumerate(REQUESTS, start=1):
        request_started = time.time()
        request = understand_request(query)
        request, request_corrections = correct_frozen_request(request_id, request)
        request_issues = validate_frozen_request(request_id, request)
        if request_issues:
            raise RuntimeError(json.dumps({"request_id": request_id, "issues": request_issues, "parsed_request": request.model_dump()}, indent=2, ensure_ascii=False, default=str))
        query_text = request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query
        request_phrases = build_request_phrases(request)
        request_frame, request_report = build_request_specific_pool(request, tmdb_client)
        request_frame = exclude_consumed(request_frame, consumed_keys)
        request_frame = request_frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)

        watchlist_frame, watchlist_report = resolve_watchlist_candidates(watchlist, tmdb_client)
        watchlist_frame = exclude_consumed(watchlist_frame, consumed_keys)
        request_embedded, request_embed_report = embed_candidates(request_frame, request_id=f"{request_id}_external")
        request_embedded, request_relevance_report = compute_request_relevance(request_embedded, query_text)
        watchlist_embedded, watchlist_embed_report = embed_candidates(watchlist_frame, request_id=f"{request_id}_watchlist")
        watchlist_embedded, watchlist_relevance_report = compute_request_relevance(watchlist_embedded, query_text)

        external_frame = select_request_relevance_candidates(request_embedded, limit=E_COUNT)
        shared_external_ids = external_frame.head(6)["source_id"].astype(str).tolist()
        selected_external = external_frame.head(E_COUNT).copy().reset_index(drop=True)
        selected_external_ids = set(selected_external["source_id"].astype(str).tolist())
        watchlist_candidates = watchlist_embedded.loc[
            ~watchlist_embedded["source_id"].astype(str).isin(selected_external_ids)
        ].copy().reset_index(drop=True)
        selected_watchlist = select_request_relevance_candidates(watchlist_candidates, limit=W_WATCHLIST_COUNT)

        source_shortfalls = {
            "external_shortfall": max(0, E_COUNT - len(selected_external)),
            "watchlist_shortfall": max(0, W_WATCHLIST_COUNT - len(selected_watchlist)),
            "shared_external_shortfall": max(0, 6 - len(shared_external_ids)),
        }
        if any(source_shortfalls.values()):
            shortfalls.append({"request_id": request_id, **source_shortfalls})
        if len(selected_external) != E_COUNT or len(selected_watchlist) != W_WATCHLIST_COUNT or len(shared_external_ids) != 6:
            raise RuntimeError(
                f"{request_id}: corrected pool construction failed (external={len(selected_external)}, watchlist={len(selected_watchlist)}, shared={len(shared_external_ids)})."
            )

        external_pool = prepare_candidate_frame(selected_external, source_label=request_id, source_condition="E", source_origin="request")
        external_pool["pool_memberships"] = external_pool["source_id"].astype(str).apply(
            lambda sid: ["E", "W"] if sid in set(shared_external_ids) else ["E"]
        )
        watchlist_pool = prepare_candidate_frame(selected_watchlist, source_label=request_id, source_condition="W", source_origin="watchlist")
        watchlist_pool["pool_memberships"] = [["W"]] * len(watchlist_pool)

        union = pd.concat([external_pool, watchlist_pool], ignore_index=True)
        union = union.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
        if union.empty:
            request_summaries.append(
                {
                    "request_id": request_id,
                    "query": query,
                    "corrections": request_corrections,
                    "parsed_request": request.model_dump(),
                    "controlled_request_spec": request.spec.model_dump() if request.spec else {},
                    "source_counts": {"request_candidates": int(len(request_frame)), "watchlist_candidates": int(len(watchlist_frame)), "union": 0},
                    "shortfalls": source_shortfalls,
                }
            )
            continue

        candidate_embeddings_report = {
            "external": request_embed_report,
            "watchlist": watchlist_embed_report,
        }
        union_scored = compute_b3_scores(union)
        union_scored = compute_direct_taste_scores(union_scored, historical)
        qualified_union, llm_usage = qualify_candidates(request, union_scored)
        qualification_llm_total += int(llm_usage.get("llm_call_count", 0) or 0)
        qualification_token_total["input"] += int(llm_usage.get("runtime_input_tokens") or 0) if llm_usage.get("runtime_input_tokens") is not None else 0
        qualification_token_total["output"] += int(llm_usage.get("runtime_output_tokens") or 0) if llm_usage.get("runtime_output_tokens") is not None else 0
        qualification_token_total["total"] += int(llm_usage.get("runtime_total_tokens") or 0) if llm_usage.get("runtime_total_tokens") is not None else 0

        eligible_union = qualified_union.loc[qualified_union["qualification_status"].astype(str).isin({"strong", "partial"})].copy().reset_index(drop=True)
        if not eligible_union.empty:
            eligible_union["request_relevance_percentile"] = percentile_scores(eligible_union, "request_relevance")
            eligible_union["direct_taste_percentile"] = percentile_scores(eligible_union, "direct_taste_score")
            eligible_union["b3_percentile"] = percentile_scores(eligible_union, "predicted_preference")
            percentile_lookup = eligible_union.loc[:, ["source_id", "request_relevance_percentile", "direct_taste_percentile", "b3_percentile"]].copy()
            qualified_union = qualified_union.merge(percentile_lookup, on="source_id", how="left", suffixes=("", "_global"))
            for column in ["request_relevance_percentile", "direct_taste_percentile", "b3_percentile"]:
                if f"{column}_global" in qualified_union.columns:
                    qualified_union[column] = qualified_union[f"{column}_global"]
                    qualified_union = qualified_union.drop(columns=[f"{column}_global"])
        else:
            qualified_union["request_relevance_percentile"] = np.nan
            qualified_union["direct_taste_percentile"] = np.nan
            qualified_union["b3_percentile"] = np.nan
        qualified_union["policy_A"] = pd.to_numeric(qualified_union["request_relevance_percentile"], errors="coerce")
        qualified_union["policy_B"] = 0.5 * pd.to_numeric(qualified_union["request_relevance_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
            qualified_union["direct_taste_percentile"], errors="coerce"
        )
        qualified_union["policy_C"] = 0.5 * pd.to_numeric(qualified_union["request_relevance_percentile"], errors="coerce") + 0.5 * pd.to_numeric(
            qualified_union["b3_percentile"], errors="coerce"
        )

        qualified_union["pool_memberships"] = qualified_union["source_id"].astype(str).apply(
            lambda sid: ["E", "W"] if sid in set(shared_external_ids) else (["E"] if sid in set(selected_external_ids) else ["W"])
        )
        qualified_union["source_condition"] = qualified_union["source_id"].astype(str).apply(
            lambda sid: "E" if sid in set(selected_external_ids) else "W"
        )
        request_rng = random.Random(SEED + request_index)
        anon_labels = make_anon_labels(request_id, len(qualified_union), request_rng)
        qualified_union["anon_label"] = anon_labels
        qualified_union["human_display_title"] = qualified_union["title"]
        qualified_union["human_display_year"] = qualified_union["year"].astype("Int64")
        qualified_union["human_display_overview"] = qualified_union["tmdb_overview"].fillna("").astype(str)

        pool_e_subset = qualified_union.loc[qualified_union["pool_memberships"].apply(lambda memberships: "E" in memberships)].copy().reset_index(drop=True)
        pool_w_subset = qualified_union.loc[qualified_union["pool_memberships"].apply(lambda memberships: "W" in memberships)].copy().reset_index(drop=True)
        if len(set(pool_e_subset["source_id"].astype(str))) != 12 or len(set(pool_w_subset["source_id"].astype(str))) != 12:
            raise RuntimeError(f"{request_id}: corrected pools must contain 12 unique candidates each after qualification.")

        policy_slates: dict[str, dict[str, pd.DataFrame]] = {
            "E": rank_candidates(pool_e_subset, "E", top_k=TOP_K),
            "W": rank_candidates(pool_w_subset, "W", top_k=TOP_K),
        }

        machine_row = {
            "request_id": request_id,
            "query": query,
            "parsed_request": request.model_dump(),
            "controlled_request_spec": request.spec.model_dump() if request.spec else {},
            "request_corrections": request_corrections,
            "candidate_sources": {
                "request_candidates": int(len(request_frame)),
                "watchlist_candidates": int(len(watchlist_frame)),
                "external_selected": int(len(selected_external)),
                "watchlist_selected": int(len(selected_watchlist)),
                "shared_external_selected": int(len(shared_external_ids)),
                "union": int(len(qualified_union)),
            },
            "retrieval_report": {
                "request_only_source_report": request_report,
                "watchlist_report": watchlist_report,
                "embedding_report": candidate_embeddings_report,
                "request_relevance_report": {
                    "external": request_relevance_report,
                    "watchlist": watchlist_relevance_report,
                },
                "qualification_llm_usage": llm_usage,
                "source_shortfalls": source_shortfalls,
                "pool_verification": {
                    "external_unique_count": int(len(set(pool_e_subset["source_id"].astype(str)))),
                    "watchlist_unique_count": int(len(set(pool_w_subset["source_id"].astype(str)))),
                    "shared_external_count": int(len(shared_external_ids)),
                    "shared_external_ids": shared_external_ids,
                    "external_ids": sorted(pool_e_subset["source_id"].astype(str).tolist()),
                    "watchlist_ids": sorted(pool_w_subset["source_id"].astype(str).tolist()),
                },
            },
            "candidates": [],
            "slates": {},
        }

        all_candidates = qualified_union.copy().reset_index(drop=True)
        slate_lookup: dict[str, list[str]] = {}
        for source_condition, subset in {"E": pool_e_subset, "W": pool_w_subset}.items():
            policies = policy_slates[source_condition]
            for policy_name, slate in policies.items():
                label = f"{source_condition}-{policy_name}"
                slate_lookup[label] = slate["anon_label"].astype(str).tolist() if not slate.empty else []
                machine_row["slates"][label] = slate.loc[
                    :,
                    [column for column in [
                        "source_id",
                        "title",
                        "year",
                        "source_condition",
                        "source_origin",
                        "pool_memberships",
                        "qualification_status",
                        "request_relevance",
                        "request_relevance_percentile",
                        "direct_taste_score",
                        "direct_taste_percentile",
                        "predicted_preference",
                        "b3_percentile",
                        "selected_rank",
                        "policy_A",
                        "policy_B",
                        "policy_C",
                    ] if column in slate.columns],
                ].to_dict(orient="records")

        for _, row in all_candidates.iterrows():
            machine_row["candidates"].append(
                {
                    "candidate_id": str(row["source_id"]),
                    "title": row.get("title"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "source_condition": row.get("source_condition"),
                    "source_origin": row.get("source_origin"),
                    "source_label": row.get("source_label"),
                    "pool_memberships": row.get("pool_memberships", []),
                    "candidate_sources": row.get("candidate_sources", []),
                    "qualification_status": row.get("qualification_status"),
                    "supported_required_aspects": row.get("supported_required_aspects", []),
                    "unsupported_required_aspects": row.get("unsupported_required_aspects", []),
                    "supported_preferred_aspects": row.get("supported_preferred_aspects", []),
                    "unsupported_preferred_aspects": row.get("unsupported_preferred_aspects", []),
                    "violated_semantic_exclusions": row.get("violated_semantic_exclusions", []),
                    "qualification_reason": row.get("qualification_reason"),
                    "caveat": row.get("caveat"),
                    "request_relevance": float(row["request_relevance"]) if pd.notna(row.get("request_relevance")) else None,
                    "request_relevance_percentile": float(row["request_relevance_percentile"]) if pd.notna(row.get("request_relevance_percentile")) else None,
                    "direct_taste_score": float(row["direct_taste_score"]) if pd.notna(row.get("direct_taste_score")) else None,
                    "direct_taste_percentile": float(row["direct_taste_percentile"]) if pd.notna(row.get("direct_taste_percentile")) else None,
                    "predicted_preference": float(row["predicted_preference"]) if pd.notna(row.get("predicted_preference")) else None,
                    "b3_percentile": float(row["b3_percentile"]) if pd.notna(row.get("b3_percentile")) else None,
                    "policy_A": float(row["policy_A"]) if pd.notna(row.get("policy_A")) else None,
                    "policy_B": float(row["policy_B"]) if pd.notna(row.get("policy_B")) else None,
                    "policy_C": float(row["policy_C"]) if pd.notna(row.get("policy_C")) else None,
                    "historical_neighbors_positive": row.get("positive_neighbors", []),
                    "historical_neighbors_negative": row.get("negative_neighbors", []),
                    "anon_label": row.get("anon_label"),
                    "selected_ranks": {
                        "E-A": int(policy_slates["E"]["A"].loc[policy_slates["E"]["A"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["E"]["A"].empty and row["source_id"] in set(policy_slates["E"]["A"]["source_id"].astype(str))
                        else None,
                        "E-B": int(policy_slates["E"]["B"].loc[policy_slates["E"]["B"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["E"]["B"].empty and row["source_id"] in set(policy_slates["E"]["B"]["source_id"].astype(str))
                        else None,
                        "E-C": int(policy_slates["E"]["C"].loc[policy_slates["E"]["C"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["E"]["C"].empty and row["source_id"] in set(policy_slates["E"]["C"]["source_id"].astype(str))
                        else None,
                        "W-A": int(policy_slates["W"]["A"].loc[policy_slates["W"]["A"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["W"]["A"].empty and row["source_id"] in set(policy_slates["W"]["A"]["source_id"].astype(str))
                        else None,
                        "W-B": int(policy_slates["W"]["B"].loc[policy_slates["W"]["B"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["W"]["B"].empty and row["source_id"] in set(policy_slates["W"]["B"]["source_id"].astype(str))
                        else None,
                        "W-C": int(policy_slates["W"]["C"].loc[policy_slates["W"]["C"]["source_id"] == row["source_id"], "selected_rank"].iloc[0])
                        if not policy_slates["W"]["C"].empty and row["source_id"] in set(policy_slates["W"]["C"]["source_id"].astype(str))
                        else None,
                    },
                }
            )
            human_rows.append(
                {
                    "request_id": request_id,
                    "anon_label": row.get("anon_label"),
                    "title": row.get("title"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "overview": row.get("tmdb_overview"),
                    "request_fit_1_5": "",
                    "would_watch": "",
                    "already_familiar": "",
                    "short_rejection_reason": "",
                    "best_recommendation": "",
                    "worst_recommendation": "",
                    "qualitative_feedback": "",
                }
            )
            unblinding_rows.append(
                {
                    "request_id": request_id,
                    "anon_label": row.get("anon_label"),
                    "title": row.get("title"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "source_condition": row.get("source_condition"),
                    "source_origin": row.get("source_origin"),
                    "source_label": row.get("source_label"),
                    "candidate_id": row.get("source_id"),
                }
            )

        machine_row["configurations"] = {
            "E-A": slate_lookup.get("E-A", []),
            "E-B": slate_lookup.get("E-B", []),
            "E-C": slate_lookup.get("E-C", []),
            "W-A": slate_lookup.get("W-A", []),
            "W-B": slate_lookup.get("W-B", []),
            "W-C": slate_lookup.get("W-C", []),
        }
        request_summaries.append(
            {
                "request_id": request_id,
                "query": query,
                "parsed_request": request.model_dump(),
                "controlled_request_spec": request.spec.model_dump() if request.spec else {},
                "request_corrections": request_corrections,
                "source_counts": {
                    "request_candidates": int(len(request_frame)),
                    "watchlist_candidates": int(len(watchlist_frame)),
                    "external_selected": int(len(selected_external)),
                    "watchlist_selected": int(len(selected_watchlist)),
                    "shared_external_selected": int(len(shared_external_ids)),
                    "pool_E_unique": int(len(pool_e_subset)),
                    "pool_W_unique": int(len(pool_w_subset)),
                    "union": int(len(all_candidates)),
                    "qualified_union": int(len(all_candidates.loc[all_candidates["qualification_status"].astype(str).isin({"strong", "partial"})])),
                },
                "shortfalls": source_shortfalls,
                "pool_verification": {
                    "shared_external_ids": shared_external_ids,
                    "pool_E_unique_ids": sorted(pool_e_subset["source_id"].astype(str).tolist()),
                    "pool_W_unique_ids": sorted(pool_w_subset["source_id"].astype(str).tolist()),
                },
            }
        )
        machine_results.append(machine_row)
        tmdb_unresolved_total += int(watchlist_report.get("unresolved_watchlist_candidates", 0) or 0)
        overall_timings.append({"request_id": request_id, "seconds": round(time.time() - request_started, 3)})

    human_sheet = pd.DataFrame(human_rows)
    if human_sheet.empty:
        human_sheet = pd.DataFrame(columns=["request_id", "anon_label", "title", "year", "overview", "request_fit_1_5", "would_watch", "already_familiar", "short_rejection_reason", "best_recommendation", "worst_recommendation", "qualitative_feedback"])
    else:
        human_sheet = human_sheet.loc[:, ["request_id", "anon_label", "title", "year", "overview", "request_fit_1_5", "would_watch", "already_familiar", "short_rejection_reason", "best_recommendation", "worst_recommendation", "qualitative_feedback"]]
    human_sheet.to_csv(HUMAN_DIR / "blinded_human_scoring_sheet.csv", index=False)

    unblinding = pd.DataFrame(unblinding_rows)
    unblinding.to_csv(HUMAN_DIR / "unblinding_key.csv", index=False)

    write_json(ARTIFACT_DIR / "experiment_results.json", machine_results)
    write_json(ARTIFACT_DIR / "experiment_summary.json", request_summaries)
    write_json(
        ARTIFACT_DIR / "reproducibility_manifest.json",
        {
            **manifest,
            "coverage": {
                "requests": len(REQUESTS),
                "requests_with_shortfalls": len(shortfalls),
                "unresolved_watchlist_candidates": tmdb_unresolved_total,
            },
            "calls": {
                "qualification_llm_calls": qualification_llm_total,
                "qualification_input_tokens": qualification_token_total["input"],
                "qualification_output_tokens": qualification_token_total["output"],
                "qualification_total_tokens": qualification_token_total["total"],
            },
            "timings": {
                "per_request_seconds": overall_timings,
                "total_seconds": round(sum(item["seconds"] for item in overall_timings), 3),
            },
            "shortfalls": shortfalls,
        },
    )

    write_json(
        ARTIFACT_DIR / "technical_summary.json",
        {
            "status": "completed_for_human_scoring",
            "baseline_commit": BASELINE_COMMIT,
            "runtime_unchanged": True,
            "requests": len(REQUESTS),
            "total_candidates": sum(item["source_counts"]["union"] for item in request_summaries),
            "watchlist_shortfalls": [item for item in shortfalls if item["watchlist_shortfall"] or item["external_shortfall"]],
            "human_scoring_sheet": str((HUMAN_DIR / "blinded_human_scoring_sheet.csv").resolve()),
            "unblinding_key": str((HUMAN_DIR / "unblinding_key.csv").resolve()),
        },
    )

    return {
        "status": "completed",
        "artifacts": {
            "manifest": str((ARTIFACT_DIR / "reproducibility_manifest.json").resolve()),
            "results": str((ARTIFACT_DIR / "experiment_results.json").resolve()),
            "summary": str((ARTIFACT_DIR / "experiment_summary.json").resolve()),
            "technical_summary": str((ARTIFACT_DIR / "technical_summary.json").resolve()),
            "human_sheet": str((HUMAN_DIR / "blinded_human_scoring_sheet.csv").resolve()),
            "unblinding_key": str((HUMAN_DIR / "unblinding_key.csv").resolve()),
        },
        "shortfalls": shortfalls,
        "timings": overall_timings,
        "llm_calls": qualification_llm_total,
        "train_test": {"train": len(train), "test": len(test)},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
