from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mvp.src.semantics import SEMANTIC_GROUPS, SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile, top_associations


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO_ROOT / "mvp/data/processed/condition_a_enriched.csv"
SEMANTIC_PATH = REPO_ROOT / "mvp/artifacts/semantic_vectors/semantic_vectors.csv"
HUMAN_REVIEW_PATH = (
    REPO_ROOT
    / "evaluation/watchlist_personalization/final_evidence/human_review/selection_rating_join.csv"
)

BG = "#14181D"
SURFACE = "#1B2028"
SURFACE_2 = "#22272F"
BORDER = "#2C323B"
TEXT = "#F0F2EF"
MUTED = "#9AA2AB"
MUTED_2 = "#6C757F"
BLUE = "#6E8CF0"
GREEN = "#5CAF5C"
ORANGE = "#E6924A"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --ta-bg: {BG};
            --ta-surface: {SURFACE};
            --ta-surface-2: {SURFACE_2};
            --ta-border: {BORDER};
            --ta-text: {TEXT};
            --ta-muted: {MUTED};
            --ta-muted-2: {MUTED_2};
            --ta-blue: {BLUE};
            --ta-green: {GREEN};
            --ta-orange: {ORANGE};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 88% 8%, rgba(110,140,240,.08), transparent 27rem),
                radial-gradient(circle at 10% 88%, rgba(92,175,92,.05), transparent 25rem),
                var(--ta-bg);
            color: var(--ta-text);
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 5rem;
        }}

        h1, h2, h3, p {{
            color: var(--ta-text);
        }}

        .ta-kicker {{
            color: var(--ta-blue);
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}

        .ta-hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.65fr) minmax(260px, .75fr);
            gap: 2rem;
            align-items: center;
            padding: 2rem 0 1.8rem;
        }}

        .ta-title {{
            font-size: clamp(3.25rem, 7vw, 6.6rem);
            line-height: .92;
            letter-spacing: -.055em;
            font-weight: 500;
            margin: 0;
        }}

        .ta-title-muted {{
            color: var(--ta-muted);
            font-weight: 400;
        }}

        .ta-subtitle {{
            color: var(--ta-muted);
            font-size: 1.08rem;
            max-width: 650px;
            line-height: 1.6;
            margin-top: 1.2rem;
        }}

        .ta-sources {{
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.35rem;
        }}

        .ta-chip {{
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--ta-border);
            background: rgba(27,32,40,.78);
            color: var(--ta-muted);
            border-radius: 999px;
            padding: .38rem .7rem;
            font-size: .76rem;
            line-height: 1;
        }}

        .ta-chip.green {{ color: #DDE9DD; border-color: #3A5A3B; background: #171E1A; }}
        .ta-chip.blue {{ color: #DDE1E5; border-color: #38466F; background: #191F2E; }}
        .ta-chip.orange {{ color: #F1D7BD; border-color: #5C4526; background: #271E14; }}

        .ta-model-wrap {{
            min-height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .ta-model {{
            width: 112px;
            height: 112px;
            border-radius: 50%;
            border: 1px solid var(--ta-blue);
            display: grid;
            place-items: center;
            position: relative;
            color: var(--ta-text);
            font-size: .74rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-align: center;
            box-shadow: 0 0 50px rgba(110,140,240,.08);
        }}

        .ta-model::before, .ta-model::after {{
            content: "";
            position: absolute;
            width: 145px;
            height: 1px;
            right: 98px;
            background: linear-gradient(90deg, transparent, var(--ta-border));
            transform-origin: right center;
        }}
        .ta-model::before {{ transform: rotate(23deg); }}
        .ta-model::after {{ transform: rotate(-20deg); }}

        .ta-dot {{
            position: absolute;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            box-shadow: 0 0 13px currentColor;
        }}
        .ta-dot.d1 {{ color: var(--ta-blue); left: 14%; top: 31%; }}
        .ta-dot.d2 {{ color: var(--ta-green); left: 2%; top: 47%; }}
        .ta-dot.d3 {{ color: var(--ta-orange); left: 19%; top: 69%; }}
        .ta-dot.d4 {{ color: var(--ta-green); left: 27%; top: 16%; }}
        .ta-dot.d5 {{ color: var(--ta-blue); left: 9%; top: 60%; }}

        .ta-section-label {{
            color: var(--ta-muted-2);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }}

        .ta-section-title {{
            color: var(--ta-text);
            font-size: 1.8rem;
            line-height: 1.15;
            font-weight: 650;
            letter-spacing: -.025em;
            margin: 0 0 .45rem;
        }}

        .ta-section-copy {{
            color: var(--ta-muted);
            font-size: .95rem;
            line-height: 1.55;
            max-width: 760px;
            margin-bottom: 1.2rem;
        }}

        .ta-card {{
            background: linear-gradient(180deg, rgba(34,39,47,.92), rgba(27,32,40,.94));
            border: 1px solid var(--ta-border);
            border-radius: 12px;
            padding: 1.05rem 1.1rem;
            min-height: 145px;
        }}

        .ta-card-accent {{
            border-color: #3A5A3B;
            background: linear-gradient(180deg, rgba(23,30,26,.98), rgba(27,32,40,.96));
        }}

        .ta-card-index {{
            color: var(--ta-blue);
            font-size: .7rem;
            font-weight: 750;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin-bottom: .8rem;
        }}

        .ta-card-title {{
            color: var(--ta-text);
            font-size: 1.03rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: .32rem;
        }}

        .ta-card-meta {{
            color: var(--ta-muted);
            font-size: .78rem;
            margin-bottom: .8rem;
        }}

        .ta-card-reason {{
            color: #C3C9CF;
            font-size: .84rem;
            line-height: 1.45;
        }}

        .ta-stat {{
            border-top: 1px solid var(--ta-border);
            padding-top: .85rem;
            margin-top: .25rem;
        }}

        .ta-stat-value {{
            color: var(--ta-text);
            font-size: 1.8rem;
            font-weight: 650;
            letter-spacing: -.035em;
        }}

        .ta-stat-label {{
            color: var(--ta-muted-2);
            font-size: .72rem;
            letter-spacing: .06em;
            text-transform: uppercase;
        }}

        .ta-signal-row {{
            display: grid;
            grid-template-columns: minmax(120px, 170px) 1fr 64px;
            gap: .85rem;
            align-items: center;
            margin: .62rem 0;
        }}

        .ta-signal-label {{
            color: var(--ta-text);
            font-size: .84rem;
        }}

        .ta-track {{
            height: 9px;
            background: var(--ta-surface-2);
            border-radius: 999px;
            overflow: hidden;
        }}

        .ta-fill {{
            height: 100%;
            background: var(--ta-green);
            border-radius: 999px;
        }}

        .ta-signal-value {{
            color: #A8C9A8;
            font-size: .78rem;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .ta-agent-shell {{
            border: 1px solid #38466F;
            background: linear-gradient(135deg, rgba(25,31,46,.96), rgba(27,32,40,.96));
            border-radius: 14px;
            padding: 1.35rem 1.4rem .65rem;
            margin-top: .4rem;
        }}

        .ta-agent-title {{
            display: flex;
            align-items: center;
            gap: .7rem;
            color: var(--ta-text);
            font-size: 1.45rem;
            font-weight: 650;
            margin-bottom: .35rem;
        }}

        .ta-agent-orb {{
            width: 30px;
            height: 30px;
            border: 1px solid var(--ta-blue);
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: var(--ta-blue);
            font-size: .68rem;
            font-weight: 800;
        }}

        .ta-agent-copy {{
            color: var(--ta-muted);
            font-size: .9rem;
            line-height: 1.5;
            margin-bottom: .55rem;
        }}

        div[data-testid="stChatInput"] {{
            background: var(--ta-surface);
            border: 1px solid var(--ta-border);
            border-radius: 12px;
        }}

        div[data-testid="stChatInput"] textarea {{
            color: var(--ta-text) !important;
        }}

        div[data-testid="stExpander"] {{
            border-color: var(--ta-border);
            background: rgba(27,32,40,.45);
        }}

        div.stButton > button {{
            border-radius: 999px;
            border: 1px solid var(--ta-border);
            background: var(--ta-surface);
            color: var(--ta-text);
            font-weight: 600;
        }}

        div.stButton > button:hover {{
            border-color: var(--ta-blue);
            color: var(--ta-blue);
            background: #191F2E;
        }}

        .ta-footnote {{
            color: var(--ta-muted-2);
            font-size: .72rem;
            line-height: 1.45;
        }}

        @media (max-width: 820px) {{
            .ta-hero {{
                grid-template-columns: 1fr;
            }}
            .ta-model-wrap {{
                display: none;
            }}
            .ta-title {{
                font-size: 3.8rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pretty_dimension(name: str) -> str:
    return name.replace("_", " ").replace(" anti hero", " anti-hero").title()


@st.cache_data(show_spinner=False)
def _load_profile() -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(HISTORY_PATH)
    semantic_vectors = SemanticVectorStore(SEMANTIC_PATH).load()
    history = history.copy()
    history["preference_weight"] = pd.to_numeric(history["preference_weight"], errors="coerce")
    rated = history[
        history["preference_weight"].notna()
        & history["canonical_id"].notna()
        & history["media_type"].astype(str).str.lower().eq("film")
    ].copy()
    profile = build_taste_profile(
        rated,
        semantic_vectors,
        id_column="canonical_id",
        target_col="preference_weight",
    )
    return rated, profile


@st.cache_data(show_spinner=False)
def _load_dashboard_picks() -> list[dict[str, Any]]:
    if not HUMAN_REVIEW_PATH.exists():
        return []
    review = pd.read_csv(HUMAN_REVIEW_PATH)
    if review.empty:
        return []

    frame = review.copy()
    frame = frame[
        frame["scenario"].astype(str).eq("full_watchlist")
        & frame["variant"].astype(str).eq("A")
        & frame["would_watch_yes_no_unsure"].astype(str).str.lower().eq("yes")
        & ~frame["already_watched_raw"].astype(str).str.lower().eq("yes")
    ].copy()

    frame["request_fit_numeric"] = pd.to_numeric(frame["request_fit_raw"], errors="coerce")
    frame["taste_fit_numeric"] = pd.to_numeric(frame["taste_fit_raw"], errors="coerce")
    frame["combined_fit"] = frame[["request_fit_numeric", "taste_fit_numeric"]].mean(axis=1)
    frame = frame[
        (frame["request_fit_numeric"] >= 4)
        & (frame["taste_fit_numeric"] >= 4)
    ].sort_values(
        ["combined_fit", "taste_fit_numeric", "request_fit_numeric", "selected_position"],
        ascending=[False, False, False, True],
    )

    picks: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for _, row in frame.iterrows():
        prompt_id = str(row["prompt_id"])
        if prompt_id in used_prompts:
            continue
        used_prompts.add(prompt_id)
        picks.append(
            {
                "title": str(row["title"]),
                "year": None,
                "prompt_id": prompt_id,
                "fit": float(row["combined_fit"]),
            }
        )
        if len(picks) == 3:
            break
    return picks


@st.cache_resource(show_spinner=False)
def _load_agent_service():
    from mvp.src.watchlist_personalization.reversible_history_watchlist_service import (
        load_reversible_history_watchlist_service,
    )

    return load_reversible_history_watchlist_service()


def _signal_bars(profile: pd.DataFrame, n: int = 8) -> str:
    top = top_associations(profile, kind="positive", n=n)
    if top.empty:
        return '<div class="ta-footnote">No semantic profile is available yet.</div>'

    values = pd.to_numeric(top["pearson_preference_association"], errors="coerce")
    max_value = max(float(values.max()), 0.001)
    rows: list[str] = []
    for _, row in top.iterrows():
        value = float(row["pearson_preference_association"])
        width = max(6.0, min(100.0, (value / max_value) * 100.0))
        rows.append(
            f"""
            <div class="ta-signal-row">
                <div class="ta-signal-label">{html.escape(_pretty_dimension(str(row['dimension'])))}</div>
                <div class="ta-track"><div class="ta-fill" style="width:{width:.1f}%"></div></div>
                <div class="ta-signal-value">{value:+.3f}</div>
            </div>
            """
        )
    return "".join(rows)


def _taste_signature(profile: pd.DataFrame) -> list[str]:
    top = top_associations(profile, kind="positive", n=6)
    return [_pretty_dimension(str(value)) for value in top["dimension"].tolist()]


def _render_recommendation_cards(recommendations: list[dict[str, Any]], *, accent: bool = False) -> None:
    if not recommendations:
        st.markdown(
            '<div class="ta-card"><div class="ta-card-title">No grounded match yet.</div>'
            '<div class="ta-card-reason">Try a broader request or remove one constraint.</div></div>',
            unsafe_allow_html=True,
        )
        return

    columns = st.columns(min(3, len(recommendations)))
    for index, recommendation in enumerate(recommendations):
        with columns[index % len(columns)]:
            title = html.escape(str(recommendation.get("title") or "Untitled"))
            year = recommendation.get("year")
            year_text = html.escape(str(year)) if year not in (None, "", "nan") else "Film"

            aspects = recommendation.get("supported_request_aspects") or recommendation.get("grounded_evidence") or []
            if isinstance(aspects, str):
                aspects = [aspects]
            aspects = [str(item) for item in aspects if str(item).strip()]

            reason = recommendation.get("why_it_may_fit") or recommendation.get("request_match")
            if not reason and aspects:
                reason = "Grounded in " + ", ".join(_pretty_dimension(item) for item in aspects[:3]) + "."
            if not reason:
                route = recommendation.get("route_origin") or recommendation.get("variant")
                reason = "Selected by the Taste Agent" + (f" · {route}" if route else "") + "."

            accent_class = " ta-card-accent" if accent else ""
            st.markdown(
                f"""
                <div class="ta-card{accent_class}">
                    <div class="ta-card-index">0{index + 1} · Recommendation</div>
                    <div class="ta-card-title">{title}</div>
                    <div class="ta-card-meta">{year_text}</div>
                    <div class="ta-card-reason">{html.escape(str(reason))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_dashboard_picks(picks: list[dict[str, Any]]) -> None:
    if not picks:
        st.markdown(
            '<div class="ta-card"><div class="ta-card-title">Your next slate lives here.</div>'
            '<div class="ta-card-reason">Ask Taste Agent below and your results will appear as grounded recommendation cards.</div></div>',
            unsafe_allow_html=True,
        )
        return

    columns = st.columns(len(picks))
    for index, pick in enumerate(picks):
        with columns[index]:
            title = html.escape(str(pick["title"]))
            prompt_id = html.escape(str(pick["prompt_id"]))
            st.markdown(
                f"""
                <div class="ta-card">
                    <div class="ta-card-index">0{index + 1} · Validated pick</div>
                    <div class="ta-card-title">{title}</div>
                    <div class="ta-card-meta">Taste × request fit · {pick['fit']:.1f}/5</div>
                    <div class="ta-card-reason">A high-fit unseen title from the latest human-reviewed {prompt_id} slate.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_full_taxonomy(profile: pd.DataFrame) -> None:
    lookup = profile.set_index("dimension").to_dict(orient="index")
    for group_name, dimensions in SEMANTIC_GROUPS.items():
        st.markdown(f"**{_pretty_dimension(group_name)}**")
        group_rows = []
        for dimension in dimensions:
            row = lookup.get(dimension, {})
            group_rows.append(
                {
                    "Signal": _pretty_dimension(dimension),
                    "Preference association": row.get("pearson_preference_association"),
                    "Prevalence": row.get("prevalence"),
                    "Evidence": row.get("evidence_count"),
                    "Confidence": row.get("evidence_confidence"),
                }
            )
        st.dataframe(
            pd.DataFrame(group_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Preference association": st.column_config.NumberColumn(format="%+.3f"),
                "Prevalence": st.column_config.NumberColumn(format="%.2f"),
                "Evidence": st.column_config.NumberColumn(format="%d"),
                "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )


def _quick_prompt(label: str, prompt: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        st.session_state["queued_prompt"] = prompt
        st.rerun()


def _extract_recommendations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = payload.get("response", {}) if isinstance(payload, dict) else {}
    recommendations = response.get("recommendations", []) if isinstance(response, dict) else []
    return recommendations if isinstance(recommendations, list) else []


def main() -> None:
    st.set_page_config(
        page_title="Taste Agent",
        page_icon="◌",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()

    rated, profile = _load_profile()
    dashboard_picks = _load_dashboard_picks()
    signature = _taste_signature(profile)

    st.markdown(
        """
        <div class="ta-hero">
            <div>
                <div class="ta-kicker">Taste intelligence · Personal dashboard</div>
                <div class="ta-title">Taste Agent<br><span class="ta-title-muted">for Letterboxd</span></div>
                <div class="ta-subtitle">
                    Your viewing history becomes a semantic Taste Model; your watchlist and current intent shape what comes next.
                </div>
                <div class="ta-sources">
                    <span class="ta-chip blue">Ratings · long-term taste</span>
                    <span class="ta-chip">Viewing history</span>
                    <span class="ta-chip green">Watchlist · expressed interest</span>
                    <span class="ta-chip orange">Current intent</span>
                </div>
            </div>
            <div class="ta-model-wrap">
                <span class="ta-dot d1"></span><span class="ta-dot d2"></span><span class="ta-dot d3"></span>
                <span class="ta-dot d4"></span><span class="ta-dot d5"></span>
                <div class="ta-model">TASTE<br>MODEL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_1, stat_2, stat_3 = st.columns(3)
    with stat_1:
        st.markdown(
            f'<div class="ta-stat"><div class="ta-stat-value">{len(rated):,}</div>'
            '<div class="ta-stat-label">rated films powering the profile</div></div>',
            unsafe_allow_html=True,
        )
    with stat_2:
        st.markdown(
            '<div class="ta-stat"><div class="ta-stat-value">62</div>'
            '<div class="ta-stat-label">interpretable semantic dimensions</div></div>',
            unsafe_allow_html=True,
        )
    with stat_3:
        st.markdown(
            '<div class="ta-stat"><div class="ta-stat-value">A</div>'
            '<div class="ta-stat-label">accepted watchlist-first policy</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.markdown('<div class="ta-section-label">My Taste</div>', unsafe_allow_html=True)
        st.markdown('<div class="ta-section-title">What consistently resonates</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ta-section-copy">These are descriptive associations between semantic film characteristics and your historical ratings — not personality traits.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_signal_bars(profile, n=8), unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ta-section-label">Taste signature</div>', unsafe_allow_html=True)
        st.markdown('<div class="ta-section-title">Your strongest signals</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ta-section-copy">A compact reading of the full 62-dimensional profile.</div>',
            unsafe_allow_html=True,
        )
        chips = "".join(f'<span class="ta-chip green">{html.escape(signal)}</span>' for signal in signature)
        st.markdown(f'<div class="ta-sources">{chips}</div>', unsafe_allow_html=True)

        negative = top_associations(profile, kind="negative", n=4)
        if not negative.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="ta-section-label">Lower association</div>', unsafe_allow_html=True)
            for _, row in negative.iterrows():
                st.markdown(
                    f'<span class="ta-chip">{html.escape(_pretty_dimension(str(row["dimension"])))} '
                    f'{float(row["pearson_preference_association"]):+.2f}</span> ',
                    unsafe_allow_html=True,
                )

    with st.expander("Explore all 62 semantic dimensions"):
        st.caption(
            "Preference association is Pearson correlation with your rating-derived preference weight. "
            "Confidence is an evidence heuristic, not a probability or confidence interval."
        )
        _render_full_taxonomy(profile)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="ta-section-label">For you</div>', unsafe_allow_html=True)
    st.markdown('<div class="ta-section-title">A few picks your model got right</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ta-section-copy">Instant dashboard cards come from the accepted watchlist-first policy and completed human review; live prompts below use the service directly.</div>',
        unsafe_allow_html=True,
    )
    _render_dashboard_picks(dashboard_picks)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ta-agent-shell">
            <div class="ta-section-label">Taste Agent</div>
            <div class="ta-agent-title"><span class="ta-agent-orb">TA</span> Ask for your next film</div>
            <div class="ta-agent-copy">
                Describe a mood, a constraint, a reference film, or how adventurous you want to be.
                The agent searches first, qualifies against the request, then personalizes from your history and watchlist.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        _quick_prompt("Quiet + contemplative", "I want something quiet, contemplative and character-driven.", "qp1")
    with q2:
        _quick_prompt("French tonight", "Recommend me a French film to watch tonight.", "qp2")
    with q3:
        _quick_prompt("Surprise me", "Show me something surprising that still fits my taste.", "qp3")

    queued_prompt = st.session_state.pop("queued_prompt", None)
    prompt = st.chat_input("What are you in the mood for?")
    if queued_prompt:
        prompt = queued_prompt

    if prompt:
        st.session_state["last_prompt"] = prompt
        try:
            with st.status("Taste Agent is building your slate…", expanded=True) as status:
                st.write("Understanding your request")
                st.write("Searching the watchlist and film catalog")
                service = _load_agent_service()
                st.write("Checking grounded fit and ranking the eligible films")
                result = service.recommend(
                    prompt,
                    scenario_label="full_watchlist",
                    variant="A",
                    requested_count=5,
                )
                status.update(label="Your slate is ready", state="complete", expanded=False)
            st.session_state["last_result"] = result
        except Exception as exc:
            st.session_state["last_result"] = None
            st.error(
                "The live Taste Agent could not complete this request. "
                "The dashboard remains available offline; check the Letterboxd ZIP and TMDB/OpenAI environment before the demo."
            )
            with st.expander("Technical detail"):
                st.code(str(exc))

    last_result = st.session_state.get("last_result")
    last_prompt = st.session_state.get("last_prompt")
    if last_prompt:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ta-section-label">Agent response</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ta-section-title">{html.escape(str(last_prompt))}</div>',
            unsafe_allow_html=True,
        )
        if last_result:
            recommendations = _extract_recommendations(last_result)
            _render_recommendation_cards(recommendations, accent=True)
            metadata = last_result.get("runtime_metadata", {})
            with st.expander("How this slate was built"):
                st.write(
                    {
                        "watchlist-aware": metadata.get("source_watchlist_count"),
                        "eligible candidates": metadata.get("post_constraint_candidate_count"),
                        "recommendations": metadata.get("recommendation_count"),
                        "policy": metadata.get("variant", "A"),
                        "runtime": metadata.get("runtime_version"),
                    }
                )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ta-footnote">Taste Agent models characteristics associated with cultural preference — not personality or identity. '
        'The 62-dimensional profile is descriptive; recommendation policy remains separately evaluated.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
