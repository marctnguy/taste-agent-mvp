from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MVP_DIR = PROJECT_ROOT / "mvp"
RAW_DIR = MVP_DIR / "data" / "raw"
PROCESSED_DIR = MVP_DIR / "data" / "processed"
ARTIFACTS_DIR = MVP_DIR / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
SEMANTIC_VECTOR_DIR = ARTIFACTS_DIR / "semantic_vectors"
GOODREADS_SEMANTIC_VECTOR_DIR = ARTIFACTS_DIR / "goodreads_semantic_vectors"


_COLUMN_SYNONYMS = {
    "date": "watched_at",
    "created_at": "watched_at",
    "logged_at": "watched_at",
    "watched_on": "watched_at",
    "consumed_date": "watched_at",
    "rating_value": "rating",
    "user_rating": "rating",
    "score": "rating",
    "genre": "genres",
    "genre_names": "genres",
    "releaseyear": "release_year",
    "year": "release_year",
    "original_language": "language",
    "lang": "language",
    "production_country": "country",
    "countries": "country",
    "liked": "like",
    "likes": "like",
    "worktitle": "title",
    "name": "title",
    "work_id": "canonical_id",
    "source_id": "canonical_id",
    "letterboxd_id": "canonical_id",
    "tmdb_id": "canonical_id",
    "goodreads_id": "canonical_id",
    "spotify_id": "canonical_id",
}


RATING_TO_WEIGHT = {
    5.0: 1.00,
    4.5: 0.80,
    4.0: 0.60,
    3.5: 0.25,
    3.0: 0.00,
    2.5: -0.25,
    2.0: -0.60,
    1.5: -0.80,
    1.0: -1.00,
    0.5: -1.00,
}


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_ARTIFACTS_DIR, SEMANTIC_VECTOR_DIR, GOODREADS_SEMANTIC_VECTOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: _COLUMN_SYNONYMS.get(re.sub(r"[\s\-]+", "_", column.strip().lower()), column.strip().lower())
        for column in df.columns
    }
    return df.rename(columns=renamed).copy()


def _extract_first_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def rating_to_preference_weight(rating: object) -> float | None:
    numeric = _extract_first_float(rating)
    if numeric is None:
        return None
    normalized = round(numeric * 2) / 2
    return RATING_TO_WEIGHT.get(normalized)


def add_preference_weight_column(
    df: pd.DataFrame,
    rating_col: str = "rating",
    output_col: str = "preference_weight",
) -> pd.DataFrame:
    frame = df.copy()
    if output_col in frame.columns:
        existing = pd.to_numeric(frame[output_col], errors="coerce")
    else:
        existing = pd.Series(np.nan, index=frame.index)
    if rating_col not in frame.columns:
        frame[output_col] = existing
        return frame
    derived = frame[rating_col].apply(rating_to_preference_weight)
    frame[output_col] = existing.combine_first(derived)
    return frame


def coerce_datetime(df: pd.DataFrame, column: str = "watched_at") -> pd.DataFrame:
    frame = df.copy()
    if column in frame.columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=False)
    return frame


def _split_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[|,/;]+", text)
    return [part.strip() for part in parts if part.strip()]


def parse_multi_value_column(series: pd.Series) -> list[list[str]]:
    return [_split_values(value) for value in series.tolist()]


def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = "watched_at",
    user_col: str = "user_id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = df.copy()
    if len(frame) < 2:
        return frame.copy(), frame.iloc[0:0].copy()

    if time_col in frame.columns:
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")

    if time_col in frame.columns and frame[time_col].notna().any():
        frame = frame.sort_values(time_col)
        if user_col in frame.columns and frame[user_col].nunique(dropna=True) > 1:
            train_parts = []
            test_parts = []
            for _, group in frame.groupby(user_col, dropna=False):
                group = group.sort_values(time_col)
                split_idx = max(1, int(np.floor(len(group) * (1 - test_size))))
                train_parts.append(group.iloc[:split_idx])
                test_parts.append(group.iloc[split_idx:])
            train = pd.concat(train_parts, ignore_index=True) if train_parts else frame.iloc[0:0].copy()
            test = pd.concat(test_parts, ignore_index=True) if test_parts else frame.iloc[0:0].copy()
            return train, test

        split_idx = max(1, int(np.floor(len(frame) * (1 - test_size))))
        train = frame.iloc[:split_idx].copy()
        test = frame.iloc[split_idx:].copy()
        return train, test

    rng = np.random.default_rng(random_state)
    shuffled = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    split_idx = max(1, int(np.floor(len(shuffled) * (1 - test_size))))
    return shuffled.iloc[:split_idx].copy(), shuffled.iloc[split_idx:].copy()


def infer_work_id_column(df: pd.DataFrame, preferred: Sequence[str] | None = None) -> str:
    candidates = preferred or (
        "canonical_id",
        "work_id",
        "tmdb_id",
        "letterboxd_id",
        "goodreads_id",
        "spotify_id",
        "title",
    )
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError("No usable work identifier column found.")


def add_decade_column(df: pd.DataFrame, year_col: str = "release_year", output_col: str = "release_decade") -> pd.DataFrame:
    frame = df.copy()
    if year_col not in frame.columns:
        frame[output_col] = np.nan
        return frame
    year = pd.to_numeric(frame[year_col], errors="coerce")
    frame[output_col] = (np.floor(year / 10) * 10).astype("Int64").astype(str) + "s"
    frame.loc[year.isna(), output_col] = np.nan
    return frame
