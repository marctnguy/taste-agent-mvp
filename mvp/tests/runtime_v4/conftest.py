from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_trace_factory():
    def factory(events: list[tuple[str, str, object | None]]):
        def trace(name: str, **kwargs):
            @contextmanager
            def cm():
                run = SimpleNamespace(metadata={}, end=lambda outputs=None: events.append(("end", name, outputs)))
                events.append(("enter", name, kwargs))
                try:
                    yield run
                finally:
                    events.append(("exit", name, None))

            return cm()

        return trace

    return factory


@pytest.fixture
def make_embedding_frame():
    def _make(source_ids: list[str], vectors: list[list[float]] | None = None, *, model: str = "local-hashing-1536") -> pd.DataFrame:
        if vectors is None:
            vectors = [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(len(source_ids))]
        rows = []
        for source_id, vector in zip(source_ids, vectors, strict=True):
            row: dict[str, object] = {
                "source_id": str(source_id),
                "title": f"Movie {source_id}",
                "document_hash": f"hash-{source_id}",
                "embedding_model": model,
                "embedding_dim": len(vector),
                "status": "success",
            }
            row.update({f"emb_{index:04d}": float(value) for index, value in enumerate(vector)})
            rows.append(row)
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def runtime_v4_candidate_pool():
    return pd.DataFrame(
        [
            {
                "source_id": "1001",
                "tmdb_id": 1001,
                "title": "Power Broker",
                "year": 1999,
                "release_year": 1999,
                "tmdb_genres": "Crime|Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A crime family story about power and violence.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.90,
                "rank": 1,
                "emb_0000": 1.0,
                "emb_0001": 0.0,
                "b3_applicability_distance": 0.05,
                "request_relevance": 0.10,
            },
            {
                "source_id": "1002",
                "tmdb_id": 1002,
                "title": "Stage Lights",
                "year": 1998,
                "release_year": 1998,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "GB",
                "tmdb_overview": "A celebrated stage actress struggles with celebrity in the theatre world.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.55,
                "rank": 2,
                "emb_0000": 0.0,
                "emb_0001": 1.0,
                "b3_applicability_distance": 0.75,
                "request_relevance": 0.95,
            },
        ]
    )
