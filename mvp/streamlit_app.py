from __future__ import annotations



import html

import os

import sys

import textwrap
from datetime import datetime
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Any



import pandas as pd

import streamlit as st





# ============================================================

# REPO IMPORT BOOTSTRAP

# ============================================================



REPO_ROOT = Path(__file__).resolve().parents[1]



if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))





from mvp.src.config import get_api_keys, load_runtime_env

from mvp.src.semantics import SemanticVectorStore

from mvp.src.taste_profile import build_taste_profile





# ============================================================

# PATHS

# ============================================================



HISTORY_PATH = REPO_ROOT / "mvp/data/processed/condition_a_enriched.csv"



SEMANTIC_PATH = (

    REPO_ROOT

    / "mvp/artifacts/semantic_vectors/semantic_vectors.csv"

)



HUMAN_REVIEW_PATH = (

    REPO_ROOT

    / "evaluation/watchlist_personalization/final_evidence/"

      "human_review/selection_rating_join.csv"

)





# ============================================================

# PRODUCT IDENTITY

# ============================================================



USER_NAME = os.getenv(

    "TASTE_AGENT_USER_NAME",

    "MARC",

).upper()





BG = "#14181D"

SURFACE = "#1B2028"

SURFACE_2 = "#22272F"

BORDER = "#303640"



TEXT = "#F2F2EF"

MUTED = "#9CA3AB"

MUTED_2 = "#737C86"



BLUE = "#6E8CF0"

GREEN = "#5CAF5C"

ORANGE = "#E6924A"

ORANGE_SOFT = "#F2B06F"





# ============================================================

# CONSUMER-FRIENDLY TASTE SIGNALS

# ============================================================



DISTINCTIVE_DIMENSIONS = {

    "niche",

    "abstract",

    "atmospheric",

    "surreal",

    "experimental",

    "contemplative",

    "melancholic",

    "historical",

    "nostalgic",

    "naturalistic",

    "stylized",

    "slow_burn",

    "character_driven",

    "intimate_relationships",

    "outsider_protagonist",

    "ambiguous",

    "bizarre",

    "absurd",

    "darkly_comic",

    "challenging",

    "unsettling",

    "visual",

    "identity",

    "alienation",

    "coming_of_age",

}





SIGNAL_COPY = {

    "niche":

        "You do not need the obvious choice to feel at home. "

        "Some of your strongest films sit slightly outside the mainstream.",



    "abstract":

        "You respond to films that leave space for interpretation "

        "rather than explaining everything for you.",



    "atmospheric":

        "Mood matters. You often connect with films where texture, "

        "place and feeling carry as much weight as plot.",



    "surreal":

        "You seem comfortable when cinema bends reality "

        "instead of simply following it.",



    "experimental":

        "You have patience for filmmakers who play with form, "

        "structure and cinematic language.",



    "contemplative":

        "You make room for films that slow down and let ideas "

        "or emotions unfold.",



    "melancholic":

        "There is a recurring pull toward films with emotional "

        "weight and a touch of melancholy.",



    "historical":

        "You return to cinema that looks backward — through "

        "period settings, memory or another cultural moment.",



    "nostalgic":

        "Memory and the emotional texture of the past seem "

        "to resonate with you.",



    "naturalistic":

        "You often respond to films that feel observed "

        "rather than manufactured.",



    "stylized":

        "You notice when a film has a visual point of view "

        "of its own.",



    "slow_burn":

        "You do not need a film to rush. Meaning and tension "

        "can build gradually.",



    "character_driven":

        "Characters often matter more to you than machinery "

        "or spectacle.",



    "intimate_relationships":

        "You frequently connect with films built around emotional "

        "closeness and complicated relationships.",



    "outsider_protagonist":

        "You often follow characters standing slightly outside "

        "the world around them.",



    "ambiguous":

        "You seem comfortable leaving a film with questions "

        "rather than answers.",



    "bizarre":

        "You have room in your taste for films that are strange "

        "on purpose.",



    "absurd":

        "A little absurdity does not scare you off — sometimes "

        "it is exactly the point.",



    "darkly_comic":

        "You respond to films that find humour in places where "

        "it probably should not exist.",



    "challenging":

        "You are willing to meet a film halfway rather than "

        "expecting it to make itself immediately accessible.",



    "unsettling":

        "You seem comfortable with films that leave a slight "

        "knot behind.",



    "visual":

        "Strong visual language is often part of what makes "

        "a film stick with you.",



    "identity":

        "Questions of identity repeatedly appear in films "

        "you respond to.",



    "alienation":

        "Stories about distance, isolation and not quite fitting "

        "in recur in your taste.",



    "coming_of_age":

        "Stories of becoming — awkward, painful or liberating — "

        "appear often in your taste.",

}





# ============================================================

# HTML RENDERER

# ============================================================



def ui(markup: str) -> None:

    """

    Render raw HTML directly.



    Important:

    st.html() avoids Streamlit's Markdown parser, which was

    previously turning indented HTML into visible code blocks.

    """

    st.html(

        textwrap.dedent(markup).strip()

    )





# ============================================================

# CSS

# ============================================================



def inject_css() -> None:



    st.html(

        f"""

        <style>



        :root {{

            --bg: {BG};

            --surface: {SURFACE};

            --surface2: {SURFACE_2};

            --border: {BORDER};

            --text: {TEXT};

            --muted: {MUTED};

            --muted2: {MUTED_2};

            --blue: {BLUE};

            --green: {GREEN};

            --orange: {ORANGE};

            --orange-soft: {ORANGE_SOFT};

        }}



        html,

        body,

        [class*="css"] {{

            font-family:

                Inter,

                -apple-system,

                BlinkMacSystemFont,

                "Segoe UI",

                sans-serif;

        }}

        /* --------------------------------------------
           GLOBAL PRODUCT TYPOGRAPHY
           Native Streamlit controls use the same
           type system as the product UI.
        -------------------------------------------- */

        .stApp,
        .stApp p,
        .stApp label,
        .stApp button,
        .stApp input,
        .stApp textarea,
        .stApp [role="button"],
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stExpander"],
        .stApp [data-testid="stAlert"],
        .stApp [data-baseweb="base-input"],
        .stApp [data-baseweb="input"] {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
        }}

        /* Streamlit uses Google's Material Symbols as ligature icons.
           Never force Inter onto those spans or the icon names become
           visible text such as `keyboard_arrow_right`. */
        .stApp [data-testid="stIconMaterial"],
        .stApp .material-symbols-rounded,
        .stApp .material-symbols-outlined,
        .stApp [class*="material-symbol"] {{
            font-family:
                "Material Symbols Rounded",
                "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
        }}



        .stApp {{

            background:

                radial-gradient(

                    circle at 88% 5%,

                    rgba(230,146,74,.10),

                    transparent 28rem

                ),

                radial-gradient(

                    circle at 8% 75%,

                    rgba(92,175,92,.055),

                    transparent 30rem

                ),

                var(--bg);

            color: var(--text);

        }}



        #MainMenu,

        footer,

        header {{

            visibility: hidden;

        }}



        .block-container {{

            max-width: 1240px;

            padding-top: 2.4rem;

            padding-bottom: 6rem;

        }}



        h1,

        h2,

        h3,

        p {{

            color: var(--text);

        }}





        /* --------------------------------------------

           TYPOGRAPHY

        -------------------------------------------- */



        .section-kicker {{

            color: var(--orange);

            font-size: .69rem;

            font-weight: 800;

            letter-spacing: .13em;

            text-transform: uppercase;

            margin-bottom: .55rem;

        }}



        .section-title {{

            color: var(--text);

            font-size: clamp(1.9rem, 3.3vw, 2.75rem);

            font-weight: 650;

            line-height: 1.08;

            letter-spacing: -.04em;

            margin-bottom: .7rem;

        }}



        .section-copy {{

            color: var(--muted);

            line-height: 1.62;

            font-size: .98rem;

            max-width: 730px;

            margin-bottom: 1.6rem;

        }}





        /* --------------------------------------------

           HERO

        -------------------------------------------- */



        .hero {{

            padding: 3.2rem 0 4.3rem;

            max-width: 1100px;

        }}



        .brand {{

            color: var(--muted2);

            font-size: .76rem;

            font-weight: 800;

            letter-spacing: .14em;

            text-transform: uppercase;

            margin-bottom: 2.5rem;

        }}



        .hero-name {{

            color: var(--orange);

            font-size: .83rem;

            font-weight: 800;

            letter-spacing: .13em;

            text-transform: uppercase;

            margin-bottom: 1rem;

        }}



        .hero-title {{

            color: var(--text);

            font-size: clamp(3.2rem, 7vw, 6.8rem);

            line-height: .91;

            letter-spacing: -.06em;

            font-weight: 500;

            max-width: 1120px;

        }}



        .hero-title .accent {{

            color: var(--orange-soft);

        }}



        .hero-sub {{

            color: var(--muted);

            font-size: 1.08rem;

            max-width: 720px;

            line-height: 1.65;

            margin-top: 1.6rem;

        }}



        .signal-strip {{

            display: flex;

            flex-wrap: wrap;

            gap: .65rem;

            margin-top: 1.7rem;

        }}



        .signal-chip {{

            display: inline-flex;

            align-items: center;

            border: 1px solid #684A2D;

            color: #F4D5B6;

            background: #251C15;

            border-radius: 999px;

            padding: .48rem .78rem;

            font-size: .82rem;

            font-weight: 550;

        }}





        /* --------------------------------------------

           TASTE STORIES

        -------------------------------------------- */



        .taste-story {{

            border-top: 1px solid var(--border);

            padding-top: 1.25rem;

            margin-top: .1rem;

        }}



        .taste-number {{

            color: var(--orange);

            font-size: .68rem;

            font-weight: 800;

            letter-spacing: .12em;

            text-transform: uppercase;

            margin-bottom: .75rem;

        }}



        .taste-name {{

            color: var(--text);

            font-size: 1.5rem;

            font-weight: 650;

            letter-spacing: -.025em;

            margin-bottom: .55rem;

        }}



        .taste-copy {{

            color: var(--muted);

            font-size: .91rem;

            line-height: 1.55;

            min-height: 5rem;

            margin-bottom: 1rem;

        }}



        .example-label {{

            color: var(--muted2);

            font-size: .66rem;

            text-transform: uppercase;

            letter-spacing: .09em;

            font-weight: 750;

            margin: .65rem 0 .45rem;

        }}





        /* --------------------------------------------

           POSTERS

        -------------------------------------------- */



        div[data-testid="stImage"] img {{

            border-radius: 9px;

            border: 1px solid var(--border);

            box-shadow: 0 12px 28px rgba(0,0,0,.28);

        }}



        .poster-placeholder {{

            width: 100%;

            aspect-ratio: 2 / 3;

            border-radius: 9px;

            border: 1px solid var(--border);

            background:

                linear-gradient(

                    145deg,

                    #242932,

                    #191D23

                );

            display: grid;

            place-items: center;

            color: var(--muted2);

            font-size: .7rem;

            letter-spacing: .12em;

            font-weight: 800;

            text-transform: uppercase;

        }}



        .poster-title {{

            color: var(--text);

            font-size: .83rem;

            font-weight: 650;

            line-height: 1.25;

            margin-top: .48rem;

        }}



        .poster-meta {{

            color: var(--muted2);

            font-size: .72rem;

            margin-top: .15rem;

        }}





        /* --------------------------------------------

           RADAR

        -------------------------------------------- */



        .radar-shell {{

            margin-top: 5.5rem;

            padding-top: 2.3rem;

            border-top: 1px solid var(--border);

        }}



        .radar-tag {{

            color: var(--orange);

            font-size: .65rem;

            font-weight: 800;

            letter-spacing: .1em;

            text-transform: uppercase;

            margin-top: .5rem;

        }}





        /* --------------------------------------------

           AGENT

        -------------------------------------------- */



        .agent-shell {{

            margin-top: 6rem;

            border: 1px solid #654726;

            border-radius: 16px;

            padding: 1.65rem 1.7rem 1.1rem;

            background:

                radial-gradient(

                    circle at 100% 0%,

                    rgba(230,146,74,.13),

                    transparent 25rem

                ),

                linear-gradient(

                    145deg,

                    rgba(35,28,21,.98),

                    rgba(27,32,40,.97)

                );

        }}



        .agent-title {{

            color: var(--text);

            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;

            font-size: 1.65rem;

            font-weight: 650;

            letter-spacing: -.025em;

            margin-bottom: .45rem;

            display: flex;

            align-items: center;

        }}

        .agent-title,
        .agent-title * {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
        }}



        .agent-orb {{

            width: 33px;

            height: 33px;

            border: 1px solid var(--orange);

            color: var(--orange);

            border-radius: 50%;

            display: inline-grid;

            place-items: center;

            font-size: .68rem;

            font-weight: 900;

            margin-right: .7rem;

        }}



        .agent-copy {{

            color: var(--muted);

            line-height: 1.55;

            font-size: .93rem;

            max-width: 740px;

        }}



        div[data-testid="stChatInput"] {{

            border: 1px solid #654726;

            background: var(--surface);

            border-radius: 14px;

        }}

        div[data-testid="stChatInput"] textarea {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
            font-size: .94rem !important;
            font-weight: 500 !important;
            letter-spacing: -.01em !important;
            color: var(--text) !important;
        }}

        div[data-testid="stChatInput"] textarea::placeholder {{
            color: var(--muted) !important;
            opacity: 1 !important;
        }}

        div[data-testid="stChatInput"] button {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
        }}





        /* --------------------------------------------

           BUTTONS

        -------------------------------------------- */




        div.stButton > button {{
            border-radius: 4px;
            border: 1px solid #00A82A;
            background: #00C030;
            color: #F7FFF8;
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
            font-size: .82rem;
            font-weight: 750;
            letter-spacing: -.01em;
            padding: .58rem .9rem;
            box-shadow: 0 5px 16px rgba(0,192,48,.14);
        }}

        div.stButton > button p {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
            font-size: inherit !important;
            font-weight: inherit !important;
            letter-spacing: inherit !important;
            color: inherit !important;
        }}



        div.stButton > button:hover {{

            border-color: #16D845;

            background: #16D845;

            color: #FFFFFF;

        }}





        /* --------------------------------------------

           RECOMMENDATION SLATE

        -------------------------------------------- */



        .rec-title {{

            color: var(--text);

            font-size: 1rem;

            line-height: 1.25;

            font-weight: 650;

            margin-top: .65rem;

        }}



        .rec-year {{

            color: var(--muted2);

            font-size: .75rem;

            margin-top: .2rem;

        }}



        .rec-reason {{

            color: var(--muted);

            font-size: .82rem;

            line-height: 1.48;

            margin-top: .7rem;

        }}



        .rec-watchlist {{

            display: inline-block;

            margin-top: .6rem;

            color: #B5D6B5;

            border: 1px solid #37573B;

            background: #19221B;

            border-radius: 999px;

            padding: .25rem .5rem;

            font-size: .65rem;

        }}





        /* --------------------------------------------
           DISCOVERY + AGENT INTERACTION
        -------------------------------------------- */

        div[data-testid="stForm"] {{
            border: 2px solid transparent !important;
            border-radius: 10px !important;
            padding: 1rem 1rem .9rem !important;
            background:
                linear-gradient(
                    135deg,
                    rgba(30,34,40,.99),
                    rgba(24,29,35,.99)
                ) padding-box,
                linear-gradient(
                    90deg,
                    #FF8000 0%,
                    #00C030 50%,
                    #40BCF4 100%
                ) border-box !important;
            box-shadow:
                0 12px 34px rgba(0,0,0,.20),
                0 0 24px rgba(64,188,244,.05),
                0 0 24px rgba(0,192,48,.04);
        }}

        div[data-testid="stForm"] input {{
            color: var(--text) !important;
            background: #171B21 !important;
            border-radius: 4px !important;
        }}

        div[data-testid="stForm"] input::placeholder {{
            color: var(--muted) !important;
            opacity: 1 !important;
        }}

        /* Taste Agent submit: Letterboxd light blue.
           Quick prompt / discovery buttons stay green. */
        div[data-testid="stForm"] button {{
            border: 1px solid #40BCF4 !important;
            border-radius: 4px !important;
            background: #40BCF4 !important;
            color: #FFFFFF !important;
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
            font-size: .82rem !important;
            font-weight: 750 !important;
            letter-spacing: -.01em !important;
            box-shadow: 0 5px 16px rgba(64,188,244,.18) !important;
        }}

        div[data-testid="stForm"] button p,
        div[data-testid="stForm"] button span {{
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif !important;
            font-size: inherit !important;
            font-weight: inherit !important;
            letter-spacing: inherit !important;
            color: inherit !important;
        }}

        div[data-testid="stForm"] button:hover {{
            background: #69CBF7 !important;
            border-color: #69CBF7 !important;
            color: #FFFFFF !important;
        }}

        .result-shell {{
            margin-top: 2.3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #5B4128;
        }}

        .details-title {{
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -.02em;
            margin-bottom: .3rem;
        }}

        .details-meta {{
            color: var(--orange-soft);
            font-size: .75rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }}

        .details-row {{
            display: grid;
            grid-template-columns: 72px 1fr;
            gap: .6rem;
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.45;
            padding: .35rem 0;
            border-top: 1px solid rgba(255,255,255,.05);
        }}

        .details-row span {{
            color: var(--muted2);
        }}

        .details-row strong {{
            color: var(--text);
            font-weight: 600;
        }}

        .details-synopsis {{
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.55;
            margin-top: .9rem;
        }}

        /* --------------------------------------------
           POSTER HOVER DETAILS
        -------------------------------------------- */

        [data-testid="column"],
        .stColumn {{
            overflow: visible !important;
        }}

        .movie-hover-card {{
            position: relative;
            overflow: visible;
            z-index: 1;
        }}

        .movie-poster-shell {{
            position: relative;
            overflow: visible;
            z-index: 1;
        }}

        .movie-hover-card:hover,
        .movie-hover-card:focus-within {{
            z-index: 10000;
        }}

        .movie-poster-image {{
            display: block;
            width: 100%;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            border-radius: 9px;
            border: 1px solid var(--border);
            box-shadow: 0 12px 28px rgba(0,0,0,.28);
            cursor: default;
        }}

        .movie-hover-panel {{
            position: absolute;
            top: 0;
            left: calc(100% + 14px);
            width: min(360px, 38vw);
            max-width: calc(100vw - 36px);
            max-height: 430px;
            overflow-y: auto;
            padding: 1rem 1.05rem 1.05rem;
            border: 1px solid #4A4036;
            border-radius: 12px;
            background: rgba(20,24,29,.985);
            box-shadow: 0 20px 60px rgba(0,0,0,.52);
            opacity: 0;
            visibility: hidden;
            transform: translateX(-8px);
            transition:
                opacity .16s ease,
                transform .16s ease,
                visibility .16s ease;
            pointer-events: none;
            z-index: 99999;
        }}

        .movie-hover-card.open-left .movie-hover-panel {{
            left: auto;
            right: calc(100% + 14px);
            transform: translateX(8px);
        }}

        .movie-hover-card:hover .movie-hover-panel,
        .movie-hover-card:focus-within .movie-hover-panel {{
            opacity: 1;
            visibility: visible;
            transform: translateX(0);
            pointer-events: auto;
        }}

        .movie-hover-eyebrow {{
            color: var(--orange);
            font-size: .64rem;
            font-weight: 800;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }}

        .movie-hover-title {{
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 720;
            letter-spacing: -.02em;
            line-height: 1.25;
            margin-bottom: .35rem;
        }}

        .movie-hover-meta {{
            color: var(--orange-soft);
            font-size: .72rem;
            line-height: 1.45;
            margin-bottom: .8rem;
        }}

        .movie-hover-row {{
            display: grid;
            grid-template-columns: 66px 1fr;
            gap: .55rem;
            padding: .38rem 0;
            border-top: 1px solid rgba(255,255,255,.055);
            color: var(--muted);
            font-size: .75rem;
            line-height: 1.4;
        }}

        .movie-hover-row span {{
            color: var(--muted2);
        }}

        .movie-hover-row strong {{
            color: var(--text);
            font-weight: 600;
        }}

        .movie-hover-synopsis {{
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.52;
            margin-top: .75rem;
        }}

        .movie-hover-hint {{
            color: var(--muted2);
            font-size: .62rem;
            margin-top: .35rem;
        }}

        @media (max-width: 900px) {{
            .movie-hover-panel,
            .movie-hover-card.open-left .movie-hover-panel {{
                left: 0;
                right: auto;
                top: calc(100% + 10px);
                width: min(330px, 80vw);
            }}
        }}


        /* --------------------------------------------

           FOOTER

        -------------------------------------------- */



        .footer-note {{

            color: var(--muted2);

            font-size: .7rem;

            line-height: 1.5;

            margin-top: 5rem;

            border-top: 1px solid var(--border);

            padding-top: 1.2rem;

        }}





        @media (max-width: 820px) {{



            .block-container {{

                padding-left: 1.25rem;

                padding-right: 1.25rem;

            }}



            .hero {{

                padding-top: 2rem;

            }}



            .hero-title {{

                font-size: 3.5rem;

            }}



        }}



        </style>

        """

    )





# ============================================================

# HELPERS

# ============================================================



def pretty_dimension(value: str) -> str:

    return (

        value

        .replace("_", " ")

        .replace("anti hero", "anti-hero")

        .title()

    )





def safe_int(value: Any) -> int | None:



    try:



        if value is None:

            return None



        if pd.isna(value):

            return None



        return int(float(value))



    except Exception:

        return None





def first_non_null(*values: Any) -> Any:



    for value in values:



        try:

            if value is not None and not pd.isna(value):

                return value

        except Exception:

            if value is not None:

                return value



    return None





def details_year(

    details: dict[str, Any] | None,

) -> int | None:



    if not details:

        return None



    release_date = details.get("release_date")



    if not release_date:

        return None



    try:

        return int(str(release_date)[:4])

    except Exception:

        return None





def poster_url(

    details: dict[str, Any] | None,

) -> str | None:



    if not details:

        return None



    path = details.get("poster_path")



    if not path:

        return None



    return (

        "https://image.tmdb.org/t/p/w500"

        + str(path)

    )





def truncate(

    text: str,

    n: int = 155,

) -> str:



    text = " ".join(

        str(text or "").split()

    )



    if len(text) <= n:

        return text



    return (

        text[: n - 1].rstrip()

        + "…"

    )





# ============================================================

# PROFILE DATA

# ============================================================



@st.cache_data(show_spinner=False)

def load_profile_data() -> tuple[

    pd.DataFrame,

    pd.DataFrame,

    pd.DataFrame,

]:



    history = pd.read_csv(

        HISTORY_PATH

    )



    vectors = SemanticVectorStore(

        SEMANTIC_PATH

    ).load()



    history = history.copy()



    history["preference_weight"] = pd.to_numeric(

        history["preference_weight"],

        errors="coerce",

    )



    history["rating"] = pd.to_numeric(

        history["rating"],

        errors="coerce",

    )



    rated = history[

        history["preference_weight"].notna()

        & history["canonical_id"].notna()

        & history["media_type"]

            .astype(str)

            .str.lower()

            .eq("film")

    ].copy()



    profile = build_taste_profile(

        rated,

        vectors,

        id_column="canonical_id",

        target_col="preference_weight",

    )



    merged = rated.merge(

        vectors,

        on="canonical_id",

        how="inner",

    )



    return (

        rated,

        profile,

        merged,

    )





def consumer_signals(

    profile: pd.DataFrame,

    n: int = 3,

) -> list[str]:



    frame = profile.copy()



    frame["association"] = pd.to_numeric(

        frame["pearson_preference_association"],

        errors="coerce",

    )



    frame = frame[

        frame["dimension"].isin(

            DISTINCTIVE_DIMENSIONS

        )

        & frame["association"].notna()

        & (frame["association"] > 0)

    ].copy()



    frame = frame.sort_values(

        [

            "association",

            "evidence_count",

        ],

        ascending=[

            False,

            False,

        ],

    )



    signals = (

        frame["dimension"]

        .head(n)

        .astype(str)

        .tolist()

    )



    if len(signals) < n:



        fallback = (

            profile

            .assign(

                association=pd.to_numeric(

                    profile[

                        "pearson_preference_association"

                    ],

                    errors="coerce",

                )

            )

            .dropna(

                subset=["association"]

            )

            .sort_values(

                "association",

                ascending=False,

            )

        )



        for dimension in (

            fallback["dimension"]

            .astype(str)

            .tolist()

        ):



            if dimension not in signals:

                signals.append(

                    dimension

                )



            if len(signals) >= n:

                break



    return signals[:n]





def representative_films(

    merged_history: pd.DataFrame,

    signals: list[str],

    per_signal: int = 2,

) -> dict[

    str,

    list[dict[str, Any]],

]:



    result: dict[

        str,

        list[dict[str, Any]],

    ] = {}



    used_tmdb_ids: set[int] = set()



    for signal in signals:



        if signal not in merged_history.columns:



            result[signal] = []



            continue



        frame = merged_history.copy()



        frame[signal] = pd.to_numeric(

            frame[signal],

            errors="coerce",

        )



        frame = frame[

            frame["rating"].notna()

            & (frame["rating"] >= 4.0)

            & frame[signal].notna()

            & (frame[signal] >= 0.45)

            & frame["tmdb_id"].notna()

        ].copy()



        frame = frame.sort_values(

            [

                signal,

                "rating",

                "preference_weight",

            ],

            ascending=[

                False,

                False,

                False,

            ],

        )



        selected: list[

            dict[str, Any]

        ] = []



        for _, row in frame.iterrows():



            tmdb_id = safe_int(

                row.get("tmdb_id")

            )



            if not tmdb_id:

                continue



            if tmdb_id in used_tmdb_ids:

                continue



            raw_year = first_non_null(

                row.get("year"),

                row.get("release_year"),

            )



            selected.append(

                {

                    "tmdb_id": tmdb_id,

                    "title": str(

                        row.get("title")

                        or ""

                    ),

                    "year": safe_int(

                        raw_year

                    ),

                }

            )



            used_tmdb_ids.add(

                tmdb_id

            )



            if (

                len(selected)

                >= per_signal

            ):

                break



        result[signal] = selected



    return result





# ============================================================

# DASHBOARD PICKS

# ============================================================



@st.cache_data(show_spinner=False)

def load_radar_picks(

    limit: int = 5,

) -> list[dict[str, Any]]:



    if not HUMAN_REVIEW_PATH.exists():

        return []



    frame = pd.read_csv(

        HUMAN_REVIEW_PATH

    )



    if frame.empty:

        return []



    frame = frame[

        frame["scenario"]

            .astype(str)

            .eq("full_watchlist")

        & frame["variant"]

            .astype(str)

            .eq("A")

        & frame["would_watch_yes_no_unsure"]

            .astype(str)

            .str.lower()

            .eq("yes")

        & frame["already_watched_raw"]

            .astype(str)

            .str.lower()

            .str.strip()

            .eq("no")

    ].copy()



    frame["request_fit"] = pd.to_numeric(

        frame["request_fit_raw"],

        errors="coerce",

    )



    frame["taste_fit"] = pd.to_numeric(

        frame["taste_fit_raw"],

        errors="coerce",

    )



    frame = frame[

        (frame["request_fit"] >= 4)

        & (frame["taste_fit"] >= 4)

    ].copy()



    frame["combined"] = (

        frame["request_fit"]

        + frame["taste_fit"]

    )



    frame = frame.sort_values(

        [

            "combined",

            "taste_fit",

            "request_fit",

            "selected_position",

        ],

        ascending=[

            False,

            False,

            False,

            True,

        ],

    )



    picks: list[

        dict[str, Any]

    ] = []



    used_titles: set[str] = set()

    used_prompts: set[str] = set()



    for _, row in frame.iterrows():



        title = str(

            row.get("title")

            or ""

        ).strip()



        prompt_id = str(

            row.get("prompt_id")

            or ""

        )



        tmdb_id = safe_int(

            row.get("candidate_id")

        )



        if not title:

            continue



        if not tmdb_id:

            continue



        if (

            prompt_id in used_prompts

            and len(used_prompts) < limit

        ):

            continue



        if title.lower() in used_titles:

            continue



        picks.append(

            {

                "title": title,

                "tmdb_id": tmdb_id,

            }

        )



        used_titles.add(

            title.lower()

        )



        used_prompts.add(

            prompt_id

        )



        if len(picks) >= limit:

            break



    return picks





# ============================================================

# TMDB DISPLAY DETAILS

# ============================================================


@st.cache_data(
    show_spinner=False,
    ttl=60 * 60 * 24,
)
def load_tmdb_details(
    tmdb_ids: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    """
    UI-only TMDB enrichment.

    Fetches the selected films only, with credits appended, so the
    recommendation layer remains untouched while the UI can show
    poster, director, cast, synopsis, runtime and TMDB rating.
    """

    load_runtime_env()
    keys = get_api_keys()

    if not keys.tmdb_api_key:
        return {}

    unique_ids = sorted(
        {
            int(tmdb_id)
            for tmdb_id in tmdb_ids
            if tmdb_id
        }
    )

    result: dict[int, dict[str, Any]] = {}

    def fetch_one(
        tmdb_id: int,
    ) -> tuple[int, dict[str, Any]]:
        params = urlencode(
            {
                "api_key": keys.tmdb_api_key,
                "language": "en-US",
                "append_to_response": "credits",
            }
        )
        url = (
            f"https://api.themoviedb.org/3/movie/"
            f"{tmdb_id}?{params}"
        )

        try:
            request = Request(
                url,
                headers={"Accept": "application/json"},
            )
            with urlopen(
                request,
                timeout=12,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
            return tmdb_id, payload

        except Exception:
            return tmdb_id, {}

    with ThreadPoolExecutor(
        max_workers=6
    ) as executor:
        futures = {
            executor.submit(
                fetch_one,
                tmdb_id,
            ): tmdb_id
            for tmdb_id in unique_ids
        }

        for future in as_completed(futures):
            tmdb_id, details = future.result()
            result[tmdb_id] = details or {}

    return result


def director_from_details(
    details: dict[str, Any] | None,
) -> str | None:
    if not details:
        return None

    crew = (
        details.get("credits", {})
        .get("crew", [])
        or []
    )

    for person in crew:
        if str(person.get("job")).lower() == "director":
            name = str(person.get("name") or "").strip()
            if name:
                return name

    return None


def cast_from_details(
    details: dict[str, Any] | None,
    limit: int = 4,
) -> list[str]:
    if not details:
        return []

    cast = (
        details.get("credits", {})
        .get("cast", [])
        or []
    )

    names: list[str] = []

    for person in cast:
        name = str(person.get("name") or "").strip()
        if not name:
            continue

        names.append(name)

        if len(names) >= limit:
            break

    return names


def runtime_text(
    details: dict[str, Any] | None,
) -> str | None:
    if not details:
        return None

    minutes = safe_int(
        details.get("runtime")
    )

    if not minutes:
        return None

    hours, remainder = divmod(
        minutes,
        60,
    )

    if hours and remainder:
        return f"{hours}h {remainder}m"

    if hours:
        return f"{hours}h"

    return f"{remainder}m"


def tmdb_rating_text(
    details: dict[str, Any] | None,
) -> str | None:
    if not details:
        return None

    try:
        rating = float(
            details.get("vote_average")
            or 0
        )
    except Exception:
        return None

    if rating <= 0:
        return None

    return f"{rating:.1f}/10"


def synopsis_sentence(
    details: dict[str, Any] | None,
    max_length: int = 150,
) -> str | None:
    if not details:
        return None

    overview = " ".join(
        str(
            details.get("overview")
            or ""
        ).split()
    )

    if not overview:
        return None

    first = overview.split(". ")[0].strip()

    if first and not first.endswith("."):
        first += "."

    return truncate(
        first,
        max_length,
    )


# ============================================================

# POSTER COMPONENT

# ============================================================



def movie_detail_panel_markup(
    details: dict[str, Any],
    *,
    fallback_title: str,
    fallback_year: int | None = None,
) -> str:
    """Return compact TMDB details for the poster hover panel."""

    title = str(
        details.get("title")
        or fallback_title
        or "Film"
    )

    year = (
        details_year(details)
        or fallback_year
    )

    director = director_from_details(
        details
    )

    cast = cast_from_details(
        details,
        limit=4,
    )

    runtime = runtime_text(
        details
    )

    rating = tmdb_rating_text(
        details
    )

    genres = [
        str(item.get("name"))
        for item in (
            details.get("genres", [])
            or []
        )
        if item.get("name")
    ]

    metadata_parts = [
        value
        for value in (
            str(year) if year else None,
            runtime,
            " · ".join(genres[:3]) if genres else None,
            f"TMDB {rating}" if rating else None,
        )
        if value
    ]

    overview = " ".join(
        str(
            details.get("overview")
            or ""
        ).split()
    )

    if overview:
        overview = truncate(
            overview,
            520,
        )

    director_row = (
        f'''
        <div class="movie-hover-row">
            <span>Director</span>
            <strong>{html.escape(director)}</strong>
        </div>
        '''
        if director
        else ""
    )

    cast_row = (
        f'''
        <div class="movie-hover-row">
            <span>Main cast</span>
            <strong>{html.escape(", ".join(cast))}</strong>
        </div>
        '''
        if cast
        else ""
    )

    synopsis = (
        f'''
        <div class="movie-hover-synopsis">
            {html.escape(overview)}
        </div>
        '''
        if overview
        else ""
    )

    return textwrap.dedent(
        f'''
        <div class="movie-hover-eyebrow">Film details</div>
        <div class="movie-hover-title">{html.escape(title)}</div>
        <div class="movie-hover-meta">
            {html.escape(" · ".join(metadata_parts))}
        </div>
        {director_row}
        {cast_row}
        {synopsis}
        '''
    ).strip()


def render_hover_movie_card(
    *,
    title: str,
    tmdb_id: int | None,
    details_lookup: dict[int, dict[str, Any]],
    year: int | None = None,
    visible_tag: str | None = None,
    reason: str | None = None,
    watchlist: bool = False,
    open_left: bool = False,
) -> None:
    """Poster stays visible; richer details appear beside it on hover."""

    details = (
        details_lookup.get(
            tmdb_id,
            {},
        )
        if tmdb_id
        else {}
    )

    image = poster_url(
        details
    )

    display_title = str(
        details.get("title")
        or title
        or "Film"
    )

    display_year = (
        year
        or details_year(details)
    )

    if image:
        image_markup = (
            f'<img class="movie-poster-image" '
            f'src="{html.escape(image, quote=True)}" '
            f'alt="Poster for {html.escape(display_title, quote=True)}">'
        )
    else:
        image_markup = (
            '<div class="poster-placeholder">Taste Agent</div>'
        )

    details_markup = movie_detail_panel_markup(
        details,
        fallback_title=display_title,
        fallback_year=display_year,
    )

    tag_markup = (
        f'<div class="radar-tag">{html.escape(visible_tag)}</div>'
        if visible_tag
        else ""
    )

    reason_markup = (
        f'<div class="rec-reason">{html.escape(reason)}</div>'
        if reason
        else ""
    )

    watchlist_markup = (
        '<div class="rec-watchlist">Already on your radar</div>'
        if watchlist
        else ""
    )

    side_class = (
        " open-left"
        if open_left
        else ""
    )

    ui(
        f'''
        <div class="movie-hover-card{side_class}" tabindex="0">
            <div class="movie-poster-shell">
                {image_markup}
                <div class="movie-hover-panel">
                    {details_markup}
                </div>
            </div>

            <div class="poster-title">
                {html.escape(display_title)}
            </div>

            <div class="poster-meta">
                {html.escape(str(display_year)) if display_year else ""}
            </div>

            {tag_markup}
            {reason_markup}
            {watchlist_markup}
        </div>
        '''
    )


def render_poster(
    *,
    title: str,
    tmdb_id: int | None,
    details_lookup: dict[int, dict[str, Any]],
    year: int | None = None,
    open_left: bool = False,
) -> None:
    render_hover_movie_card(
        title=title,
        tmdb_id=tmdb_id,
        details_lookup=details_lookup,
        year=year,
        open_left=open_left,
    )


# ============================================================

# RECOMMENDATION COPY

# ============================================================


LANGUAGE_LABELS = {
    "de": "German-language",
    "fr": "French-language",
    "es": "Spanish-language",
    "it": "Italian-language",
    "en": "English-language",
    "ja": "Japanese-language",
    "ko": "Korean-language",
    "pt": "Portuguese-language",
}

COUNTRY_LABELS = {
    "de": "German production",
    "fr": "French production",
    "es": "Spanish production",
    "it": "Italian production",
    "gb": "British production",
    "uk": "British production",
    "us": "US production",
    "jp": "Japanese production",
    "kr": "South Korean production",
}


def humanize_request_aspect(
    value: Any,
) -> str | None:
    raw = str(value or "").strip()

    if not raw:
        return None

    normalized = (
        raw.lower()
        .replace("required ", "")
        .replace("preferred ", "")
        .strip()
    )

    if normalized.startswith("languages:"):
        code = normalized.split(":", 1)[1].strip()
        return LANGUAGE_LABELS.get(
            code,
            f"{code.upper()}-language",
        )

    if normalized.startswith("production_countries:"):
        code = normalized.split(":", 1)[1].strip()
        return COUNTRY_LABELS.get(
            code,
            f"{code.upper()} production",
        )

    if normalized.startswith("genres:"):
        genre = normalized.split(":", 1)[1].strip()
        return genre.replace("_", " ")

    if normalized.startswith("release_year"):
        return "the period you asked for"

    # Keep semantic dimensions consumer-readable.
    return pretty_dimension(
        normalized
    ).lower()


def recommendation_fit_labels(
    recommendation: dict[str, Any],
) -> list[str]:
    values: list[Any] = []

    for field in (
        "supported_required_aspects",
        "supported_preferred_aspects",
        "taste_signals",
    ):
        current = recommendation.get(
            field,
            [],
        ) or []

        if isinstance(current, str):
            current = [current]

        values.extend(current)

    labels: list[str] = []

    for value in values:
        label = humanize_request_aspect(
            value
        )

        if (
            label
            and label not in labels
        ):
            labels.append(label)

    return labels


def is_watchlist_recommendation(
    recommendation: dict[str, Any],
) -> bool:
    memberships = (
        recommendation.get(
            "route_memberships",
            [],
        )
        or []
    )

    if isinstance(
        memberships,
        str,
    ):
        memberships = [
            memberships
        ]

    return any(
        "watchlist"
        in str(item).lower()
        for item
        in memberships
    )


def _production_adjective(
    details: dict[str, Any],
) -> str | None:
    mapping = {
        "US": "American",
        "GB": "British",
        "FR": "French",
        "DE": "German",
        "IT": "Italian",
        "ES": "Spanish",
        "AU": "Australian",
        "JP": "Japanese",
        "KR": "South Korean",
        "HK": "Hong Kong",
        "SE": "Swedish",
        "DK": "Danish",
        "NO": "Norwegian",
        "BR": "Brazilian",
        "AR": "Argentinian",
        "MX": "Mexican",
        "CA": "Canadian",
    }

    for country in (
        details.get("production_countries", [])
        or []
    ):
        code = str(
            country.get("iso_3166_1")
            or ""
        ).upper()
        if code in mapping:
            return mapping[code]

    return None


def _decade_label(
    year: int | None,
) -> str | None:
    if not year:
        return None

    decade = (year // 10) * 10

    if 1900 <= decade < 2000:
        return f"{str(decade)[2:]}s"

    return f"{decade}s"


def personalized_reason(
    recommendation: dict[str, Any],
    details: dict[str, Any],
) -> str:
    """
    Short consumer-facing hook.

    This deliberately avoids explaining model mechanics or repeating the
    synopsis. It uses only visible TMDB metadata and broad premise keywords
    to make the film sound worth clicking into.
    """

    year = (
        details_year(details)
        or safe_int(recommendation.get("year"))
    )

    decade = _decade_label(year)
    country = _production_adjective(details)

    genres = [
        str(item.get("name") or "").strip()
        for item in (
            details.get("genres", [])
            or []
        )
        if item.get("name")
    ]
    genres_lower = {
        genre.lower()
        for genre in genres
    }

    overview = " ".join(
        str(
            details.get("overview")
            or ""
        ).split()
    ).lower()

    context_bits = [
        value
        for value in (
            decade,
            country,
        )
        if value
    ]

    context = " ".join(context_bits)

    # Specific high-signal combinations make the copy feel editorial rather
    # than like an explanation of the ranking system.
    if "horror" in genres_lower:
        if "hotel" in overview:
            hook = (
                f"{context + ' ' if context else ''}horror with iconic "
                "haunted-hotel dread and an atmosphere that only gets colder."
            )
        elif "vampire" in overview:
            hook = (
                f"{context + ' ' if context else ''}horror with fangs, "
                "night-time menace and plenty of gothic pull."
            )
        else:
            hook = (
                f"{context + ' ' if context else ''}horror with a strong "
                "atmosphere and enough unease to stay with you afterwards."
            )

        if datetime.now().month == 10:
            hook += " Perfect for spooky season."

        return hook[0].upper() + hook[1:]

    if (
        "comedy" in genres_lower
        and (
            "music" in genres_lower
            or "band" in overview
            or "concert" in overview
            or "metal" in overview
            or "rock" in overview
        )
    ):
        if "metal" in overview:
            lead = "Metal."
        elif "rock" in overview:
            lead = "Rock."
        else:
            lead = "Music."

        era = f" The {decade}." if decade else ""

        return (
            f"{lead}{era} Comedy with backstage chaos and exactly the "
            "right amount of self-serious nonsense. Few combinations are "
            "this reliably fun."
        )

    if "music" in genres_lower:
        if "concert" in overview:
            return (
                f"{context + ' ' if context else ''}concert-film energy: "
                "performance first, filler nowhere. The kind of film that "
                "makes staying seated feel impossible."
            )[0].upper() + (
                f"{context + ' ' if context else ''}concert-film energy: "
                "performance first, filler nowhere. The kind of film that "
                "makes staying seated feel impossible."
            )[1:]

        return (
            f"{context + ' ' if context else ''}music, personality and "
            "performance energy in one very watchable package."
        )[0].upper() + (
            f"{context + ' ' if context else ''}music, personality and "
            "performance energy in one very watchable package."
        )[1:]

    if (
        "comedy" in genres_lower
        and "drama" in genres_lower
        and (
            "student" in overview
            or "school" in overview
            or "detention" in overview
            or "teen" in overview
        )
    ):
        return (
            f"{context + ' ' if context else ''}teen ensemble energy, "
            "sharp comedy and just enough emotional bite to make it stick."
        )[0].upper() + (
            f"{context + ' ' if context else ''}teen ensemble energy, "
            "sharp comedy and just enough emotional bite to make it stick."
        )[1:]

    if "documentary" in genres_lower:
        return (
            f"{context + ' ' if context else ''}documentary with a clear "
            "point of view — curious, distinctive and a long way from filler."
        )[0].upper() + (
            f"{context + ' ' if context else ''}documentary with a clear "
            "point of view — curious, distinctive and a long way from filler."
        )[1:]

    if (
        "romance" in genres_lower
        and "drama" in genres_lower
    ):
        return (
            f"{context + ' ' if context else ''}romance with real dramatic "
            "bite — more complicated feelings than soft-focus sweetness."
        )[0].upper() + (
            f"{context + ' ' if context else ''}romance with real dramatic "
            "bite — more complicated feelings than soft-focus sweetness."
        )[1:]

    if (
        "fantasy" in genres_lower
        and "drama" in genres_lower
    ):
        return (
            f"{context + ' ' if context else ''}drama with a fable-like, "
            "slightly unreal pull. Strange in the way that makes you want "
            "to know where it is going."
        )[0].upper() + (
            f"{context + ' ' if context else ''}drama with a fable-like, "
            "slightly unreal pull. Strange in the way that makes you want "
            "to know where it is going."
        )[1:]

    if "thriller" in genres_lower:
        return (
            f"{context + ' ' if context else ''}thriller with enough tension "
            "to pull you in quickly and enough style to keep it memorable."
        )[0].upper() + (
            f"{context + ' ' if context else ''}thriller with enough tension "
            "to pull you in quickly and enough style to keep it memorable."
        )[1:]

    if "crime" in genres_lower:
        return (
            f"{context + ' ' if context else ''}crime film with a rebellious "
            "edge and more personality than a standard genre exercise."
        )[0].upper() + (
            f"{context + ' ' if context else ''}crime film with a rebellious "
            "edge and more personality than a standard genre exercise."
        )[1:]

    if "animation" in genres_lower:
        return (
            f"{context + ' ' if context else ''}animation with a strong visual "
            "identity and enough personality to stand well outside kids-only territory."
        )[0].upper() + (
            f"{context + ' ' if context else ''}animation with a strong visual "
            "identity and enough personality to stand well outside kids-only territory."
        )[1:]

    primary_genre = genres[0].lower() if genres else "film"

    hook = (
        f"{context + ' ' if context else ''}{primary_genre} with a strong "
        "point of view. The kind of pick that feels distinctive before you "
        "even know exactly where it is going."
    )

    return hook[0].upper() + hook[1:]


# ============================================================

# AGENT RECOMMENDATION CARDS

# ============================================================


def render_movie_details(
    details: dict[str, Any],
) -> None:
    director = director_from_details(
        details
    )

    cast = cast_from_details(
        details,
        limit=4,
    )

    runtime = runtime_text(
        details
    )

    rating = tmdb_rating_text(
        details
    )

    genres = [
        str(item.get("name"))
        for item in (
            details.get("genres", [])
            or []
        )
        if item.get("name")
    ]

    overview = " ".join(
        str(
            details.get("overview")
            or ""
        ).split()
    )

    metadata_parts = [
        value
        for value in (
            runtime,
            " · ".join(genres[:3])
            if genres
            else None,
            f"TMDB {rating}"
            if rating
            else None,
        )
        if value
    ]

    ui(
        f"""
        <div class="details-title">
            {html.escape(str(details.get("title") or "Film details"))}
        </div>

        <div class="details-meta">
            {html.escape(" · ".join(metadata_parts))}
        </div>
        """
    )

    if director:
        ui(
            f"""
            <div class="details-row">
                <span>Director</span>
                <strong>{html.escape(director)}</strong>
            </div>
            """
        )

    if cast:
        ui(
            f"""
            <div class="details-row">
                <span>Main cast</span>
                <strong>{html.escape(", ".join(cast))}</strong>
            </div>
            """
        )

    if overview:
        ui(
            f"""
            <div class="details-synopsis">
                {html.escape(overview)}
            </div>
            """
        )


def render_agent_recommendations(
    recommendations: list[
        dict[str, Any]
    ],
) -> None:

    if not recommendations:
        ui(
            """
            <div class="section-copy">
                I couldn't find a film I was confident enough
                to recommend for that exact request.
                Try opening the brief slightly.
            </div>
            """
        )
        return

    ids = tuple(
        tmdb_id
        for tmdb_id in (
            safe_int(
                recommendation.get(
                    "tmdb_id"
                )
            )
            for recommendation in recommendations
        )
        if tmdb_id
    )

    details_lookup = load_tmdb_details(
        ids
    )

    # Always render recommendations in the same five-slot grid.
    # This keeps poster/card width identical whether the runtime returns
    # 1, 2, 3, 4 or 5 films instead of stretching a single result full-width.
    card_slots = 5

    columns = st.columns(
        card_slots,
        gap="medium",
    )

    for index, recommendation in enumerate(
        recommendations[:card_slots]
    ):

        with columns[index]:
            title = str(
                recommendation.get(
                    "title"
                )
                or "Untitled"
            )

            tmdb_id = safe_int(
                recommendation.get(
                    "tmdb_id"
                )
            )

            year = safe_int(
                recommendation.get(
                    "year"
                )
            )

            details = (
                details_lookup.get(
                    tmdb_id,
                    {},
                )
                if tmdb_id
                else {}
            )

            reason = personalized_reason(
                recommendation,
                details,
            )

            # Right-most cards open their detail panel to the left so the
            # side panel stays inside the viewport.
            # The fixed five-slot grid means viewport position no longer
            # changes with result count. Only the two right-most slots open
            # their hover panel to the left.
            open_left = index >= 3

            render_hover_movie_card(
                title=title,
                tmdb_id=tmdb_id,
                details_lookup=details_lookup,
                year=year,
                reason=reason,
                watchlist=is_watchlist_recommendation(
                    recommendation
                ),
                open_left=open_left,
            )


# ============================================================

# AGENT SERVICE

# ============================================================



@st.cache_resource(

    show_spinner=False

)

def load_agent_service():



    from mvp.src.watchlist_personalization.reversible_history_watchlist_service import (

        load_reversible_history_watchlist_service,

    )



    return (

        load_reversible_history_watchlist_service()

    )





def extract_recommendations(

    payload: dict[str, Any] | None,

) -> list[dict[str, Any]]:



    if not isinstance(

        payload,

        dict,

    ):

        return []



    response = payload.get(

        "response",

        {},

    )



    if not isinstance(

        response,

        dict,

    ):

        return []



    recommendations = (

        response.get(

            "recommendations",

            [],

        )

    )



    if not isinstance(

        recommendations,

        list,

    ):

        return []



    return recommendations





# ============================================================

# REQUEST EXECUTION

# ============================================================


def run_recommendation(
    prompt: str,
    *,
    status_label: str = "Looking for the right films…",
) -> dict[str, Any] | None:
    try:
        # A spinner disappears when the request completes. This keeps the
        # interface clean and avoids leaving Streamlit's status widget behind.
        with st.spinner(status_label):
            service = load_agent_service()

            result = service.recommend(
                prompt,
                scenario_label="full_watchlist",
                variant="A",
                requested_count=5,
            )

        return result

    except Exception as exc:
        st.error(
            "Taste Agent couldn't complete that request."
        )

        with st.expander(
            "Developer detail"
        ):
            st.code(
                str(exc)
            )

        return None


# ============================================================

# MAIN APP

# ============================================================




# ============================================================
# FRAGMENT-SCOPED INTERACTIONS
# ============================================================

@st.fragment
def render_discovery_actions(
    signals: list[str],
) -> None:
    """
    Run only the discovery controls/results when a discovery action is used.
    This prevents the rest of the Streamlit page from going stale/grey.
    """
    discovery_signal: str | None = None

    button_columns = st.columns(
        len(signals),
        gap="large",
    )

    for column, signal in zip(
        button_columns,
        signals,
    ):
        with column:
            if st.button(
                "Discover more like this →",
                key=f"discover_{signal}",
                use_container_width=True,
            ):
                discovery_signal = signal

    if discovery_signal:
        discovery_prompt = (
            "Recommend me something I haven't seen "
            f"that strongly leans {discovery_signal.replace('_', ' ')}, "
            "while still fitting my broader taste."
        )

        result = run_recommendation(
            discovery_prompt,
            status_label=(
                f"Finding more "
                f"{pretty_dimension(discovery_signal).lower()} films…"
            ),
        )

        if result:
            st.session_state["discovery_result"] = result
            st.session_state["discovery_title"] = (
                f"More {pretty_dimension(discovery_signal)} films for you"
            )

    discovery_result = st.session_state.get("discovery_result")
    discovery_title = st.session_state.get("discovery_title")

    if discovery_result and discovery_title:
        ui(
            f"""
            <div class="result-shell">

                <div class="section-kicker">
                    Discover more
                </div>

                <div class="section-title">
                    {html.escape(str(discovery_title))}
                </div>

                <div class="section-copy">
                    A fresh slate from this part of your taste.
                </div>

            </div>
            """
        )

        render_agent_recommendations(
            extract_recommendations(
                discovery_result
            )
        )


@st.fragment
def render_taste_agent_actions() -> None:
    """
    Run only Taste Agent controls/results on interaction.
    The rest of the dashboard stays fully rendered while the request runs.
    """
    quick_prompt: str | None = None
    quick_title: str | None = None

    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button(
            "Quiet & contemplative",
            key="qp1",
            use_container_width=True,
        ):
            quick_prompt = (
                "I want something quiet, contemplative "
                "and character-driven."
            )
            quick_title = (
                "Quiet & contemplative films for you"
            )

    with q2:
        if st.button(
            "French tonight",
            key="qp2",
            use_container_width=True,
        ):
            quick_prompt = (
                "Recommend me a French film "
                "to watch tonight."
            )
            quick_title = (
                "French films for tonight"
            )

    with q3:
        if st.button(
            "Surprise me",
            key="qp3",
            use_container_width=True,
        ):
            quick_prompt = (
                "Show me something surprising "
                "that still fits my taste."
            )
            quick_title = (
                "A surprise slate for you"
            )

    with st.form(
        "taste_agent_form",
        clear_on_submit=False,
    ):
        typed_prompt = st.text_input(
            "Ask Taste Agent",
            placeholder=(
                "e.g. I want something German tonight, "
                "but not a war film"
            ),
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Ask Taste Agent →",
            use_container_width=True,
        )

    agent_prompt: str | None = None
    agent_title: str | None = None

    if quick_prompt:
        agent_prompt = quick_prompt
        agent_title = quick_title

    elif submitted and typed_prompt.strip():
        agent_prompt = typed_prompt.strip()
        agent_title = typed_prompt.strip()

    if agent_prompt:
        result = run_recommendation(
            agent_prompt,
            status_label="Taste Agent is building your slate…",
        )

        if result:
            st.session_state["agent_result"] = result
            st.session_state["agent_title"] = agent_title

    agent_result = st.session_state.get("agent_result")
    saved_agent_title = st.session_state.get("agent_title")

    if agent_result and saved_agent_title:
        ui(
            f"""
            <div class="result-shell">

                <div class="section-kicker">
                    Your slate
                </div>

                <div class="section-title">
                    {html.escape(str(saved_agent_title))}
                </div>

            </div>
            """
        )

        render_agent_recommendations(
            extract_recommendations(
                agent_result
            )
        )


def main() -> None:



    st.set_page_config(

        page_title="Taste Agent",

        page_icon="◌",

        layout="wide",

        initial_sidebar_state="collapsed",

    )



    load_runtime_env()



    inject_css()



    (

        rated,

        profile,

        merged_history,

    ) = load_profile_data()



    signals = consumer_signals(

        profile,

        n=3,

    )



    examples = representative_films(

        merged_history,

        signals,

        per_signal=2,

    )



    radar = load_radar_picks(

        limit=5,

    )



    dashboard_ids: list[int] = []



    for signal in signals:



        for film in examples.get(

            signal,

            [],

        ):



            if film.get("tmdb_id"):



                dashboard_ids.append(

                    film["tmdb_id"]

                )



    for film in radar:



        if film.get("tmdb_id"):



            dashboard_ids.append(

                film["tmdb_id"]

            )



    dashboard_details = load_tmdb_details(

        tuple(

            dashboard_ids

        )

    )



    pretty_signals = [

        pretty_dimension(

            signal

        )

        for signal

        in signals

    ]



    if len(pretty_signals) >= 3:



        profile_sentence = (

            f"{pretty_signals[0]}, "

            f"{pretty_signals[1]} "

            f"and {pretty_signals[2]}"

        )



    else:



        profile_sentence = (

            ", ".join(

                pretty_signals

            )

        )





    # ========================================================

    # HERO

    # ========================================================



    chips = "".join(

        f"""

        <span class="signal-chip">

            {html.escape(

                pretty_dimension(signal)

            )}

        </span>

        """

        for signal

        in signals

    )



    ui(

        f"""

        <div class="hero">



            <div class="brand">

                Taste Agent · for Letterboxd

            </div>



            <div class="hero-name">

                {html.escape(USER_NAME)},

                this is your taste

            </div>



            <div class="hero-title">

                You keep finding your way toward

                <span class="accent">

                    {html.escape(profile_sentence)}.

                </span>

            </div>



            <div class="hero-sub">

                Not a personality test.

                Not a genre list.

                Just patterns in the films that have

                stayed with you — and a way to turn

                them into your next discovery.

            </div>



            <div class="signal-strip">

                {chips}

            </div>



        </div>

        """

    )





    # ========================================================

    # CINEMATIC FINGERPRINT

    # ========================================================

    ui(
        """
        <div class="section-kicker">
            Your cinematic fingerprint
        </div>

        <div class="section-title">
            The patterns you come back to
        </div>

        <div class="section-copy">
            These aren't boxes to put your taste into.
            They're recurring qualities in films you've
            responded to strongly.
        </div>
        """
    )

    taste_columns = st.columns(
        len(signals),
        gap="large",
    )

    for (
        index,
        (
            column,
            signal,
        ),
    ) in enumerate(
        zip(
            taste_columns,
            signals,
        )
    ):

        with column:
            copy = SIGNAL_COPY.get(
                signal,
                (
                    "This quality repeatedly "
                    "appears in films you've "
                    "responded to strongly."
                ),
            )

            ui(
                f"""
                <div class="taste-story">

                    <div class="taste-number">
                        0{index + 1} · Your taste
                    </div>

                    <div class="taste-name">
                        {
                            html.escape(
                                pretty_dimension(
                                    signal
                                )
                            )
                        }
                    </div>

                    <div class="taste-copy">
                        {html.escape(copy)}
                    </div>

                    <div class="example-label">
                        Films from your history
                    </div>

                </div>
                """
            )

            films = examples.get(
                signal,
                [],
            )

            if films:
                poster_columns = st.columns(
                    len(films)
                )

                for poster_index, (
                    poster_column,
                    film,
                ) in enumerate(
                    zip(
                        poster_columns,
                        films,
                    )
                ):
                    with poster_column:
                        render_poster(
                            title=film["title"],
                            tmdb_id=film["tmdb_id"],
                            year=film.get("year"),
                            details_lookup=dashboard_details,
                            open_left=(
                                True
                                if index == len(signals) - 1
                                else (
                                    False
                                    if index == 0
                                    else poster_index == len(films) - 1
                                )
                            ),
                        )

            else:
                st.caption(
                    "More examples will appear "
                    "as your profile grows."
                )

    # Only this action/result block reruns while a discovery request runs.
    render_discovery_actions(signals)


    # ========================================================

    # ON YOUR RADAR

    # ========================================================

    ui(
        """
        <div class="radar-shell">

            <div class="section-kicker">
                On your radar
            </div>

            <div class="section-title">
                Five films worth a closer look
            </div>

            <div class="section-copy">
                A small starting point before you ask
                Taste Agent for something more specific.
            </div>

        </div>
        """
    )

    if radar:
        radar_columns = st.columns(
            len(radar),
            gap="medium",
        )

        for radar_index, (
            column,
            film,
        ) in enumerate(
            zip(
                radar_columns,
                radar,
            )
        ):
            with column:
                details = dashboard_details.get(
                    film["tmdb_id"],
                    {},
                )

                title = (
                    details.get("title")
                    or film["title"]
                )

                year = details_year(
                    details
                )

                genres = [
                    str(
                        item.get("name")
                    )
                    for item in details.get(
                        "genres",
                        [],
                    )[:2]
                    if item.get("name")
                ]

                genre_text = (
                    " · ".join(
                        genres
                    )
                    if genres
                    else "Worth a look"
                )

                render_hover_movie_card(
                    title=str(title),
                    tmdb_id=film["tmdb_id"],
                    details_lookup=dashboard_details,
                    year=year,
                    visible_tag=genre_text,
                    open_left=(
                        radar_index
                        >= max(1, len(radar) - 2)
                    ),
                )


    # ========================================================

    # TASTE AGENT

    # ========================================================

    ui(
        """
        <div class="agent-shell">

            <div class="section-kicker">
                Ask Taste Agent
            </div>

            <div class="agent-title">
                <span class="agent-orb">
                    TA
                </span>

                What are you in the mood for?
            </div>

            <div class="agent-copy">
                Give me a mood, a language, a reference film,
                something you want to avoid — or just tell me
                what kind of night you're having.
            </div>

        </div>
        """
    )

    # Only this control/result block reruns while Taste Agent works.
    render_taste_agent_actions()


    # ========================================================

    # FOOTER

    # ========================================================



    ui(

        """

        <div class="footer-note">



            Taste Agent looks for recurring characteristics

            in the culture you respond to.

            It does not infer personality or identity.



            <br><br>



            This product uses the TMDB API but is not

            endorsed or certified by TMDB.



        </div>

        """

    )





if __name__ == "__main__":

    main()


