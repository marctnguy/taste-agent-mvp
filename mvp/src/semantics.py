from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEMANTIC_GROUPS: dict[str, tuple[str, ...]] = {
    "themes": (
        "identity",
        "love_desire",
        "grief",
        "class",
        "family",
        "alienation",
        "coming_of_age",
        "power",
        "morality",
        "friendship",
    ),
    "character": (
        "morally_ambiguous",
        "dysfunctional_relationships",
        "intimate_relationships",
        "outsider_protagonist",
        "ensemble",
        "anti_hero",
    ),
    "narrative": (
        "character_driven",
        "plot_driven",
        "nonlinear",
        "ambiguous",
        "episodic",
        "conventional",
        "complex",
    ),
    "pacing_energy": (
        "slow_burn",
        "contemplative",
        "moderate",
        "fast_paced",
        "high_energy",
    ),
    "tone": (
        "melancholic",
        "darkly_comic",
        "romantic",
        "unsettling",
        "sentimental",
        "absurd",
        "ethereal",
        "bizarre",
        "joyful",
        "tense",
        "comforting",
        "playful",
    ),
    "style": (
        "naturalistic",
        "experimental",
        "minimalist",
        "maximalist",
        "dialogue_heavy",
        "atmospheric",
        "stylized",
        "surreal",
        "visual",
    ),
    "accessibility": (
        "accessible",
        "challenging",
        "abstract",
        "niche",
    ),
    "temporal_cultural": (
        "contemporary",
        "historical",
        "nostalgic",
    ),
    "intensity": (
        "emotional_intensity",
        "violence",
        "suspense",
        "sexuality",
        "psychological_intensity",
        "contained",
    ),
}

SEMANTIC_COLUMNS: tuple[str, ...] = tuple(
    dimension for group in SEMANTIC_GROUPS.values() for dimension in group
)


def empty_semantic_vector(default: float = 0.0) -> dict[str, float]:
    return {dimension: float(default) for dimension in SEMANTIC_COLUMNS}


def coerce_semantic_frame(frame: pd.DataFrame | Mapping[str, Sequence[float]] | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=SEMANTIC_COLUMNS)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    result = frame.copy()
    for column in SEMANTIC_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    result = result.loc[:, list(SEMANTIC_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return result.clip(0.0, 1.0)


def ensure_semantic_columns(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["canonical_id", *SEMANTIC_COLUMNS])
    result = frame.copy()
    for column in SEMANTIC_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    return result


def semantic_vector_frame(records: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    semantic = coerce_semantic_frame(frame)
    if "canonical_id" in frame.columns:
        return pd.concat(
            [frame.loc[:, ["canonical_id"]].astype(str).reset_index(drop=True), semantic.reset_index(drop=True)],
            axis=1,
        )
    return semantic


def resolve_work_key(record: Mapping[str, object]) -> str | None:
    for key in ("canonical_id", "work_id", "tmdb_id", "letterboxd_id", "goodreads_id", "spotify_id", "title"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


@dataclass
class SemanticVectorStore:
    path: str | Path
    filename: str = "semantic_vectors.csv"

    def _path(self) -> Path:
        path = Path(self.path)
        return path / self.filename if path.is_dir() or path.suffix == "" else path

    def load(self) -> pd.DataFrame:
        path = self._path()
        if not path.exists():
            return pd.DataFrame(columns=["canonical_id", *SEMANTIC_COLUMNS])
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported semantic vector format: {path.suffix}")
        if "canonical_id" not in frame.columns:
            raise KeyError("Semantic vector store is missing canonical_id.")
        return pd.concat(
            [frame.loc[:, ["canonical_id"]].astype(str).reset_index(drop=True), coerce_semantic_frame(frame).reset_index(drop=True)],
            axis=1,
        )

    def save(self, frame: pd.DataFrame) -> Path:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        output = frame.copy()
        if "canonical_id" not in output.columns:
            raise KeyError("Semantic vectors must include a canonical_id column.")
        canonical_ids = output["canonical_id"].astype(str).reset_index(drop=True)
        output = coerce_semantic_frame(output).reset_index(drop=True)
        output.insert(0, "canonical_id", canonical_ids)
        output.to_csv(path, index=False)
        return path

    def upsert(self, records: Iterable[Mapping[str, float]]) -> Path:
        rows = list(records)
        frame = pd.DataFrame(rows)
        if "canonical_id" not in frame.columns:
            frame["canonical_id"] = [resolve_work_key(record) or str(index) for index, record in enumerate(rows)]
        frame = pd.concat([frame.loc[:, ["canonical_id"]].astype(str).reset_index(drop=True), coerce_semantic_frame(frame).reset_index(drop=True)], axis=1)
        return self.save(frame)
