from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"


def load_runtime_env() -> None:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


load_runtime_env()


@dataclass(frozen=True)
class ApiKeys:
    tmdb_api_key: str | None
    google_books_api_key: str | None
    openai_api_key: str | None


@dataclass(frozen=True)
class LangSmithConfig:
    api_key: str | None
    endpoint: str | None
    project: str | None
    tracing: str | None
    tracing_v2: str | None


def get_api_keys() -> ApiKeys:
    load_runtime_env()
    return ApiKeys(
        tmdb_api_key=os.getenv("TMDB_API_KEY") or None,
        google_books_api_key=os.getenv("GOOGLE_BOOKS_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
    )


def get_langsmith_config() -> LangSmithConfig:
    load_runtime_env()
    return LangSmithConfig(
        api_key=os.getenv("LANGSMITH_API_KEY") or None,
        endpoint=os.getenv("LANGSMITH_ENDPOINT") or None,
        project=os.getenv("LANGSMITH_PROJECT") or None,
        tracing=os.getenv("LANGSMITH_TRACING") or None,
        tracing_v2=os.getenv("LANGSMITH_TRACING_V2") or None,
    )
