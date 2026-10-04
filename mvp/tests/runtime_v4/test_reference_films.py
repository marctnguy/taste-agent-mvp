from __future__ import annotations

from mvp.src.generative_v4.intent_chain import ReferenceResolution, understand_request


def test_unresolved_reference_films_do_not_invent_knowledge(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.generative_v4.intent_chain.get_api_keys",
        lambda: type("Keys", (), {"tmdb_api_key": None})(),
    )

    request = understand_request("Recommend something like an imaginary film title that TMDB cannot resolve.")

    assert request.reference_status == "unresolved"
    assert request.reference_tmdb_id is None
    assert request.reference_title is not None


def test_resolved_reference_does_not_bleed_into_semantic_requirements(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.generative_v4.intent_chain._llm_request_spec",
        lambda query: (None, 0, None, None, None),
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.intent_chain._resolve_reference_title",
        lambda title, client=None: ReferenceResolution(
            title=title,
            tmdb_id=44012,
            status="resolved",
            summary="Jeanne Dielman, 23, quai du Commerce, 1080 Bruxelles",
        ),
    )
    request = understand_request("Recommend something like Jeanne Dielman.")

    assert request.request_mode == "reference"
    assert request.reference_status == "resolved"
    assert request.reference_tmdb_id == 44012
    assert "jeanne_dielman" not in request.intent.requested_taste_dimensions
    assert request.spec is not None
    assert all(
        "jeanne" not in (concept.aspect_id or "").lower()
        and "dielman" not in (concept.aspect_id or "").lower()
        for concept in request.spec.semantic_requirements
    )
    assert all(
        "jeanne" not in (concept.aspect_id or "").lower()
        and "dielman" not in (concept.aspect_id or "").lower()
        for group in request.spec.semantic_requirement_groups
        for concept in group.concepts
    )
