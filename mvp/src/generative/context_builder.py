from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from mvp.src.generative.schemas import (
    CandidateContextItem,
    CandidateSemanticEvidence,
    RecommendationIntent,
    TasteSignal,
)
from mvp.src.semantics import SEMANTIC_COLUMNS


DEFAULT_RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")
DEFAULT_TASTE_PROFILE_PATH = Path("mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv")
DEFAULT_CANDIDATE_SEMANTIC_PATH = Path("mvp/artifacts/diagnostics/comprehensive_discovery_audit/candidate_semantic_vectors.csv")
DEFAULT_CANDIDATE_CONTEXT_SIZE = 50
DEFAULT_RECOMMENDATION_COUNT = 5
TASTE_GUARDRAIL_ASSOCIATION = 0.05
TASTE_GUARDRAIL_EVIDENCE_COUNT = 3
TASTE_GUARDRAIL_CONFIDENCE = 0.4
CANDIDATE_EVIDENCE_THRESHOLD = 0.25


@dataclass
class RecommendationRuntimeContext:
    run_dir: Path
    query: str
    intent: RecommendationIntent
    target_candidate_id: str | None
    candidate_context_size: int
    recommendation_count: int
    candidate_frame: pd.DataFrame
    candidate_semantics: pd.DataFrame
    taste_profile: pd.DataFrame
    candidate_context: list[CandidateContextItem]
    positive_taste_profile: list[TasteSignal]
    negative_taste_profile: list[TasteSignal]
    hard_filters: dict[str, list[str]]


_INTENT_ALIASES = {
    "melancholic": "melancholic",
    "comforting": "comforting",
    "sentimental": "sentimental",
    "character-driven": "character_driven",
    "character driven": "character_driven",
    "grief": "grief",
    "historical": "historical",
    "nostalgic": "nostalgic",
    "surreal": "surreal",
    "abstract": "abstract",
    "bizarre": "bizarre",
    "experimental": "experimental",
    "atmospheric": "atmospheric",
    "slow burn": "slow_burn",
    "slow-burn": "slow_burn",
    "dialogue heavy": "dialogue_heavy",
    "coming of age": "coming_of_age",
    "coming-of-age": "coming_of_age",
    "nonlinear": "nonlinear",
    "ensemble": "ensemble",
    "weird": "bizarre",
    "emotionally engaging": "emotional_intensity",
}

_LANGUAGE_ALIASES = {
    "french": "fr",
    "english": "en",
    "japanese": "ja",
    "italian": "it",
    "spanish": "es",
    "german": "de",
    "korean": "ko",
    "portuguese": "pt",
    "hindi": "hi",
}

_COUNTRY_ALIASES = {
    "france": "FR",
    "japan": "JP",
    "united states": "US",
    "usa": "US",
    "u.s.": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "britain": "GB",
    "spain": "ES",
    "italy": "IT",
    "germany": "DE",
    "korea": "KR",
}

_GENRE_ALIASES = {
    "comedy": "Comedy",
    "drama": "Drama",
    "thriller": "Thriller",
    "documentary": "Documentary",
    "crime": "Crime",
    "mystery": "Mystery",
    "action": "Action",
    "romance": "Romance",
    "history": "History",
    "horror": "Horror",
    "science fiction": "Science Fiction",
    "sci fi": "Science Fiction",
    "animation": "Animation",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def parse_intent(query: str) -> RecommendationIntent:
    text = _normalize_text(query)
    intent_type: str = "GENERAL_DISCOVERY"
    if any(phrase in text for phrase in ("why are you recommending", "why this", "why does this fit", "explain")):
        intent_type = "EXPLAIN"
    elif any(phrase in text for phrase in ("something different", "surprise me", "outside my usual", "completely different", "novel", "different")):
        intent_type = "NOVELTY"
    elif any(token in text for token in (" from ", "french", "japanese", "italian", "spanish", "german", "korean", "2000s", "1990s", "1980s", "1970s", "1960s", "comedy", "drama", "thriller", "documentary", "not too long")):
        intent_type = "CONSTRAINT"
    elif any(alias in text for alias in _INTENT_ALIASES):
        intent_type = "MOOD_THEME"

    requested_taste_dimensions = []
    for alias, dimension in _INTENT_ALIASES.items():
        if alias in text and dimension not in requested_taste_dimensions:
            requested_taste_dimensions.append(dimension)

    requested_genres = []
    for alias, genre in _GENRE_ALIASES.items():
        if alias == "history" and "goodreads history" in text:
            continue
        if alias in text and genre not in requested_genres:
            requested_genres.append(genre)

    requested_languages = []
    for alias, language in _LANGUAGE_ALIASES.items():
        if alias in text and language not in requested_languages:
            requested_languages.append(language)

    requested_countries = []
    for alias, country in _COUNTRY_ALIASES.items():
        if alias in text and country not in requested_countries:
            requested_countries.append(country)

    requested_decades = []
    for decade in ("2020s", "2010s", "2000s", "1990s", "1980s", "1970s", "1960s", "1950s"):
        if decade in text and decade not in requested_decades:
            requested_decades.append(decade)

    exclusions: list[str] = []
    for match in re.finditer(r"\b(?:not|without|excluding|no)\s+([^.,;!?]+)", text):
        tail = match.group(1).strip()
        tail = re.split(r"\bbut\b|\band\b|\bor\b", tail, maxsplit=1)[0].strip()
        if tail and tail != "too long":
            exclusions.append(tail)

    exclusion_lookup = {value.lower() for value in exclusions}
    requested_genres = [genre for genre in requested_genres if genre.lower() not in exclusion_lookup]

    free_text_context = query.strip()
    return RecommendationIntent(
        intent_type=intent_type,  # type: ignore[arg-type]
        requested_taste_dimensions=requested_taste_dimensions,
        requested_genres=requested_genres,
        requested_languages=requested_languages,
        requested_countries=requested_countries,
        requested_decades=requested_decades,
        exclusions=exclusions,
        free_text_context=free_text_context,
    )


def _load_candidate_frame(run_dir: Path) -> pd.DataFrame:
    ranked = _read_csv(run_dir / "ranked_candidates.csv")
    pool = _read_csv(run_dir / "candidate_pool.csv")
    ranked["source_id"] = ranked["source_id"].astype(str)
    pool["source_id"] = pool["source_id"].astype(str)
    merged = ranked.merge(pool, on="source_id", how="left", validate="one_to_one", suffixes=("", "_pool"))
    merged["candidate_sources"] = merged["candidate_sources"].apply(lambda value: value if isinstance(value, list) else _parse_listish(value))
    merged["candidate_source_ranks"] = merged["candidate_source_ranks"].apply(lambda value: value if isinstance(value, list) else _parse_listish(value)) if "candidate_source_ranks" in merged.columns else [[] for _ in range(len(merged))]
    merged["predicted_preference"] = pd.to_numeric(merged["predicted_preference"], errors="coerce")
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce")
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged["tmdb_id"] = pd.to_numeric(merged["tmdb_id"], errors="coerce")
    return merged.sort_values("rank").reset_index(drop=True)


def _parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _load_candidate_semantics(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if "canonical_id" not in frame.columns:
        frame = frame.rename(columns={"source_id": "canonical_id"})
    frame["canonical_id"] = frame["canonical_id"].astype(str)
    return frame


def _load_taste_profile(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path)
    frame["dimension"] = frame["dimension"].astype(str)
    return frame


def _eligible_taste_profile(profile: pd.DataFrame) -> pd.DataFrame:
    numeric = profile.copy()
    numeric["preference_association"] = pd.to_numeric(numeric["pearson_preference_association"], errors="coerce")
    numeric["evidence_count"] = pd.to_numeric(numeric["evidence_count"], errors="coerce")
    numeric["evidence_confidence"] = pd.to_numeric(numeric["evidence_confidence"], errors="coerce")
    eligible = numeric.loc[
        (numeric["preference_association"].abs() >= TASTE_GUARDRAIL_ASSOCIATION)
        & (numeric["evidence_count"] >= TASTE_GUARDRAIL_EVIDENCE_COUNT)
        & (numeric["evidence_confidence"] >= TASTE_GUARDRAIL_CONFIDENCE)
    ].copy()
    return eligible


def _to_signals(frame: pd.DataFrame, max_items: int = 12) -> list[TasteSignal]:
    if frame.empty:
        return []
    cols = ["dimension", "preference_association", "evidence_count", "evidence_confidence"]
    frame = frame.loc[:, cols].copy()
    frame = frame.sort_values(
        ["preference_association", "evidence_confidence", "evidence_count"],
        ascending=[False, False, False],
    )
    records = frame.head(max_items).to_dict(orient="records")
    return [TasteSignal(**record) for record in records]


def _candidate_evidence(candidate_row: pd.Series, taste_profile: pd.DataFrame, semantic_row: pd.Series | None) -> list[CandidateSemanticEvidence]:
    eligible = _eligible_taste_profile(taste_profile)
    if semantic_row is None:
        return []
    evidence = []
    lookup = eligible.set_index("dimension")
    for dimension in SEMANTIC_COLUMNS:
        if dimension not in lookup.index:
            continue
        value = pd.to_numeric(semantic_row.get(dimension), errors="coerce")
        candidate_score = 0.0 if pd.isna(value) else float(value)
        if candidate_score < CANDIDATE_EVIDENCE_THRESHOLD:
            continue
        assoc = float(lookup.loc[dimension, "preference_association"])
        evidence.append(
            CandidateSemanticEvidence(
                dimension=dimension,
                candidate_score=candidate_score,
                user_preference_association=assoc,
                evidence_count=int(lookup.loc[dimension, "evidence_count"]),
                evidence_confidence=float(lookup.loc[dimension, "evidence_confidence"]),
                direction="positive" if assoc >= 0 else "negative",
            )
        )
    evidence.sort(key=lambda item: (abs(item.user_preference_association) * item.candidate_score, item.candidate_score), reverse=True)
    return evidence[:5]


def build_recommendation_runtime_context(
    query: str,
    target_candidate_id: str | None = None,
    candidate_context_size: int = DEFAULT_CANDIDATE_CONTEXT_SIZE,
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    taste_profile_path: str | Path = DEFAULT_TASTE_PROFILE_PATH,
    candidate_semantic_path: str | Path = DEFAULT_CANDIDATE_SEMANTIC_PATH,
) -> RecommendationRuntimeContext:
    run_dir = Path(run_dir)
    intent = parse_intent(query)
    candidate_frame = _load_candidate_frame(run_dir).head(candidate_context_size).copy().reset_index(drop=True)
    candidate_semantics = _load_candidate_semantics(candidate_semantic_path)
    taste_profile = _load_taste_profile(taste_profile_path)

    if intent.intent_type == "EXPLAIN" and target_candidate_id:
        candidate_frame = candidate_frame.loc[candidate_frame["source_id"].astype(str) == str(target_candidate_id)].copy().reset_index(drop=True)

    candidate_semantic_lookup = candidate_semantics.set_index("canonical_id")
    taste_lookup = _eligible_taste_profile(taste_profile)
    positive_taste_profile = _to_signals(taste_lookup.loc[taste_lookup["preference_association"] > 0])
    negative_taste_profile = _to_signals(taste_lookup.loc[taste_lookup["preference_association"] < 0])

    candidate_context: list[CandidateContextItem] = []
    for _, row in candidate_frame.iterrows():
        candidate_id = str(row["source_id"])
        semantic_row = candidate_semantic_lookup.loc[candidate_id] if candidate_id in candidate_semantic_lookup.index else None
        evidence = _candidate_evidence(row, taste_profile, semantic_row)
        candidate_context.append(
            CandidateContextItem(
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
        )

    hard_filters = {
        "requested_genres": intent.requested_genres,
        "requested_languages": intent.requested_languages,
        "requested_countries": intent.requested_countries,
        "requested_decades": intent.requested_decades,
        "exclusions": intent.exclusions,
    }

    return RecommendationRuntimeContext(
        run_dir=run_dir,
        query=query,
        intent=intent,
        target_candidate_id=str(target_candidate_id) if target_candidate_id is not None else None,
        candidate_context_size=candidate_context_size,
        recommendation_count=recommendation_count,
        candidate_frame=candidate_frame,
        candidate_semantics=candidate_semantics,
        taste_profile=taste_profile,
        candidate_context=candidate_context,
        positive_taste_profile=positive_taste_profile,
        negative_taste_profile=negative_taste_profile,
        hard_filters=hard_filters,
    )


def hard_filter_candidates(context: RecommendationRuntimeContext) -> list[CandidateContextItem]:
    filtered = []
    intent = context.intent
    exclusions = {value.lower() for value in intent.exclusions}
    for candidate in context.candidate_context:
        if intent.intent_type == "EXPLAIN" and context.target_candidate_id:
            if candidate.candidate_id != str(context.target_candidate_id):
                continue
        if intent.requested_languages and candidate.original_language and candidate.original_language.lower() not in {value.lower() for value in intent.requested_languages}:
            continue
        if intent.requested_countries:
            countries = {value.lower() for value in candidate.production_countries}
            if not countries.intersection({value.lower() for value in intent.requested_countries}):
                continue
        if intent.requested_decades and candidate.year is not None:
            decade = f"{candidate.year // 10 * 10}s"
            if decade not in intent.requested_decades:
                continue
        if intent.requested_genres:
            genres = {value.lower() for value in candidate.genres}
            if not genres.intersection({value.lower() for value in intent.requested_genres}):
                continue
        if exclusions:
            haystack = " ".join(
                [
                    candidate.title.lower(),
                    " ".join(candidate.genres).lower(),
                    (candidate.original_language or "").lower(),
                    " ".join(candidate.production_countries).lower(),
                    " ".join(candidate.candidate_provenance).lower(),
                ]
            )
            if any(exclusion in haystack for exclusion in exclusions):
                continue
        filtered.append(candidate)
    return filtered
