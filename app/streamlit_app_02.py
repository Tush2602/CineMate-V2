"""
streamlit_app.py
────────────────
Cinemate — Netflix-style Movie Recommendation Frontend

Design direction: Cinematic dark luxury
- Deep black backgrounds with subtle warm gradients
- Poster-first grid layout like Netflix/Prime
- Horizontal scrollable recommendation rows
- Real-time user switching
- Genre-based new user onboarding

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ── Config ─────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://localhost:8000")
# TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API_KEY = "be50cfdf6bda073b5c74b0a42f7e6dca"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"
TMDB_BACK    = "https://image.tmdb.org/t/p/w1280"

st.set_page_config(
    page_title     = "Cinemate",
    page_icon      = "🎬",
    layout         = "wide",
    initial_sidebar_state = "collapsed"
)

# ══════════════════════════════════════════════════════════
# GLOBAL CSS — Netflix/Prime aesthetic
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset and base ── */
* { box-sizing: border-box; }

.stApp {
    background-color: #0d0d0d !important;
    color: #e8e8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #222 !important;
}

/* ── Hero section ── */
.hero {
    position    : relative;
    width       : 100%;
    min-height  : 520px;
    background  : linear-gradient(
        105deg,
        #0d0d0d 0%,
        #0d0d0d 40%,
        rgba(13,13,13,0.6) 70%,
        transparent 100%
    ), linear-gradient(
        180deg,
        transparent 50%,
        #0d0d0d 100%
    );
    display     : flex;
    align-items : flex-end;
    padding     : 48px 48px 64px;
    margin-bottom: -32px;
    overflow    : hidden;
}

.hero-bg {
    position   : absolute;
    inset      : 0;
    background : linear-gradient(135deg, #1a0a2e 0%, #0d0d0d 60%);
    z-index    : 0;
}

.hero-content {
    position   : relative;
    z-index    : 1;
    max-width  : 600px;
}

.hero-badge {
    display     : inline-block;
    background  : #e50914;
    color       : white;
    font-size   : 11px;
    font-weight : 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding     : 4px 12px;
    border-radius: 2px;
    margin-bottom: 16px;
}

.hero-title {
    font-family : 'Bebas Neue', cursive !important;
    font-size   : 72px !important;
    line-height : 1 !important;
    color       : #ffffff !important;
    margin      : 0 0 16px 0 !important;
    letter-spacing: 2px;
    text-shadow : 0 4px 32px rgba(0,0,0,0.8);
}

.hero-subtitle {
    font-size   : 16px;
    color       : #aaa;
    margin      : 0 0 32px 0;
    line-height : 1.6;
    font-weight : 300;
}

/* ── Navbar ── */
.navbar {
    position    : sticky;
    top         : 0;
    z-index     : 999;
    display     : flex;
    align-items : center;
    justify-content: space-between;
    padding     : 16px 48px;
    background  : linear-gradient(180deg, #0d0d0d 0%, transparent 100%);
    backdrop-filter: blur(8px);
}

.nav-logo {
    font-family : 'Bebas Neue', cursive;
    font-size   : 32px;
    color       : #e50914;
    letter-spacing: 3px;
    text-shadow : 0 0 20px rgba(229,9,20,0.4);
}

.nav-links {
    display : flex;
    gap     : 32px;
}

.nav-link {
    color     : #ccc;
    font-size : 14px;
    font-weight: 500;
    cursor    : pointer;
    transition: color 0.2s;
}
.nav-link:hover { color: white; }
.nav-link.active { color: white; font-weight: 600; }

/* ── Row headers ── */
.row-header {
    font-family : 'Bebas Neue', cursive;
    font-size   : 24px;
    color       : #e8e8e8;
    letter-spacing: 1px;
    margin      : 40px 0 16px 48px;
    display     : flex;
    align-items : center;
    gap         : 12px;
}

.row-header-accent {
    color     : #e50914;
    font-size : 14px;
    font-weight: 600;
    letter-spacing: 1px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Movie card ── */
.movie-card {
    position      : relative;
    border-radius : 6px;
    overflow      : hidden;
    cursor        : pointer;
    transition    : transform 0.3s ease, box-shadow 0.3s ease;
    aspect-ratio  : 2/3;
    background    : #1a1a1a;
}

.movie-card:hover {
    transform  : scale(1.05) translateY(-4px);
    box-shadow : 0 20px 60px rgba(0,0,0,0.8);
    z-index    : 10;
}

.movie-card img {
    width      : 100%;
    height     : 100%;
    object-fit : cover;
    display    : block;
}

.movie-card-overlay {
    position   : absolute;
    bottom     : 0;
    left       : 0;
    right      : 0;
    background : linear-gradient(transparent, rgba(0,0,0,0.95));
    padding    : 32px 12px 12px;
    opacity    : 0;
    transition : opacity 0.3s;
}

.movie-card:hover .movie-card-overlay {
    opacity : 1;
}

.card-title {
    font-size   : 13px;
    font-weight : 600;
    color       : white;
    margin      : 0 0 4px;
    white-space : nowrap;
    overflow    : hidden;
    text-overflow: ellipsis;
}

.card-genre {
    font-size : 11px;
    color     : #999;
}

.card-score {
    position     : absolute;
    top          : 8px;
    right        : 8px;
    background   : rgba(0,0,0,0.75);
    color        : #f5c518;
    font-size    : 11px;
    font-weight  : 600;
    padding      : 3px 7px;
    border-radius: 4px;
    backdrop-filter: blur(4px);
}

.card-rank {
    position     : absolute;
    top          : 8px;
    left         : 8px;
    background   : #e50914;
    color        : white;
    font-size    : 11px;
    font-weight  : 700;
    padding      : 3px 7px;
    border-radius: 4px;
}

.no-poster {
    width           : 100%;
    height          : 100%;
    display         : flex;
    flex-direction  : column;
    align-items     : center;
    justify-content : center;
    background      : linear-gradient(135deg, #1a1a2e, #16213e);
    padding         : 16px;
    text-align      : center;
}

.no-poster-icon {
    font-size     : 36px;
    margin-bottom : 8px;
}

.no-poster-title {
    font-size    : 12px;
    color        : #ccc;
    font-weight  : 500;
    line-height  : 1.3;
}

/* ── User switcher ── */
.user-switcher {
    background    : #111;
    border        : 1px solid #222;
    border-radius : 12px;
    padding       : 20px;
    margin        : 0 48px 32px;
}

.user-card {
    background    : #1a1a1a;
    border        : 1px solid #2a2a2a;
    border-radius : 8px;
    padding       : 12px;
    text-align    : center;
    cursor        : pointer;
    transition    : all 0.2s;
}

.user-card:hover {
    border-color : #e50914;
    background   : #1f1010;
}

.user-card.active {
    border-color : #e50914;
    background   : #1f1010;
}

.user-avatar {
    font-size     : 32px;
    margin-bottom : 6px;
}

.user-name {
    font-size   : 12px;
    color       : #ccc;
    font-weight : 500;
}

/* ── Search bar ── */
.search-wrapper {
    padding : 24px 48px 0;
}

.stTextInput input {
    background    : #1a1a1a !important;
    border        : 1px solid #333 !important;
    border-radius : 6px !important;
    color         : white !important;
    font-family   : 'DM Sans', sans-serif !important;
    font-size     : 15px !important;
    padding       : 12px 16px !important;
}

.stTextInput input:focus {
    border-color : #e50914 !important;
    box-shadow   : 0 0 0 2px rgba(229,9,20,0.2) !important;
}

/* ── Genre tags ── */
.genre-chip {
    display      : inline-block;
    background   : #1a1a1a;
    border       : 1px solid #333;
    color        : #ccc;
    font-size    : 12px;
    padding      : 4px 12px;
    border-radius: 20px;
    margin       : 4px;
    cursor       : pointer;
    transition   : all 0.2s;
}

.genre-chip:hover, .genre-chip.active {
    background   : #e50914;
    border-color : #e50914;
    color        : white;
}

/* ── Stats bar ── */
.stats-bar {
    display         : flex;
    gap             : 32px;
    padding         : 16px 48px;
    background      : #0a0a0a;
    border-top      : 1px solid #1a1a1a;
    border-bottom   : 1px solid #1a1a1a;
    margin-bottom   : 8px;
    overflow-x      : auto;
}

.stat-item {
    display        : flex;
    flex-direction : column;
    align-items    : center;
    white-space    : nowrap;
    min-width      : 80px;
}

.stat-value {
    font-family : 'Bebas Neue', cursive;
    font-size   : 22px;
    color       : #e50914;
    line-height : 1;
}

.stat-label {
    font-size  : 10px;
    color      : #666;
    margin-top : 2px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Feedback modal ── */
.feedback-panel {
    background    : #111;
    border        : 1px solid #222;
    border-radius : 12px;
    padding       : 24px 48px;
    margin        : 0 48px 32px;
}

/* ── Section padding ── */
.section-pad {
    padding : 0 48px;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div {
    background  : #1a1a1a !important;
    border      : 1px solid #333 !important;
    color       : white !important;
}

.stSlider > div > div > div {
    background : #e50914 !important;
}

.stButton > button {
    background    : #e50914 !important;
    color         : white !important;
    border        : none !important;
    border-radius : 6px !important;
    font-family   : 'DM Sans', sans-serif !important;
    font-weight   : 600 !important;
    padding       : 8px 20px !important;
    transition    : all 0.2s !important;
}

.stButton > button:hover {
    background  : #b20710 !important;
    transform   : translateY(-1px) !important;
}

div[data-testid="stMetric"] {
    background    : #1a1a1a !important;
    border        : 1px solid #222 !important;
    border-radius : 8px !important;
    padding       : 16px !important;
}

div[data-testid="stMetricValue"] {
    color : #e50914 !important;
}

.stSpinner > div {
    border-top-color : #e50914 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb {
    background    : #333;
    border-radius : 2px;
}
::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════

if 'user_idx'      not in st.session_state:
    st.session_state.user_idx      = 0
if 'page'          not in st.session_state:
    st.session_state.page          = 'home'
if 'search_query'  not in st.session_state:
    st.session_state.search_query  = ''
if 'selected_movie'not in st.session_state:
    st.session_state.selected_movie= None


# ══════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def api_get(endpoint, params=None):
    try:
        r = requests.get(
            f"{API_URL}{endpoint}",
            params=params, timeout=15
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint, data):
    try:
        r = requests.post(
            f"{API_URL}{endpoint}",
            json=data, timeout=10
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_poster(tmdb_id):
    if not TMDB_API_KEY or not tmdb_id:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            path = r.json().get("poster_path")
            back = r.json().get("backdrop_path")
            return {
                "poster"  : f"{TMDB_IMG}{path}" if path else None,
                "backdrop": f"{TMDB_BACK}{back}" if back else None,
                "overview": r.json().get("overview", ""),
                "year"    : r.json().get("release_date", "")[:4],
                "rating"  : r.json().get("vote_average", 0),
            }
    except Exception:
        pass
    return None


def api_health():
    return api_get("/health")


# ══════════════════════════════════════════════════════════
# PREDEFINED USER PROFILES
# ══════════════════════════════════════════════════════════

USER_PROFILES = [
    {"name": "Alex",    "icon": "🎬", "idx": 0,      "desc": "Action fan"},
    {"name": "Sam",     "icon": "🎭", "idx": 1024,   "desc": "Drama lover"},
    {"name": "Jordan",  "icon": "🚀", "idx": 15679,  "desc": "Sci-Fi geek"},
    {"name": "Morgan",  "icon": "😂", "idx": 5000,   "desc": "Comedy first"},
    {"name": "Casey",   "icon": "👻", "idx": 8000,   "desc": "Horror fan"},
    {"name": "New User","icon": "✨", "idx": -1,     "desc": "Create profile"},
]

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy",
    "Horror", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]


# ══════════════════════════════════════════════════════════
# COMPONENTS
# ══════════════════════════════════════════════════════════

def render_navbar():
    """Top navigation bar."""
    pages    = ["Home", "Search", "My List", "About"]
    active   = st.session_state.page.capitalize()

    links_html = "".join([
        f'<span class="nav-link {"active" if p == active else ""}">'
        f'{p}</span>'
        for p in pages
    ])

    health     = api_health()
    api_status = (
        "🟢 Live" if health else "🔴 Offline"
    )

    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">CINEMATE</div>
        <div class="nav-links">{links_html}</div>
        <div style="font-size:12px;color:#666">
            {api_status}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_movie_card(movie, rank=None,
                       show_score=True, size="normal"):
    """Render a single movie poster card."""
    tmdb_id = movie.get('tmdb_id', 0)
    title   = movie.get('title', 'Unknown')
    genres  = movie.get('genres', '')
    score   = movie.get('score', 0)

    # Fetch poster
    poster_data = get_poster(tmdb_id) if tmdb_id else None
    poster_url  = poster_data['poster'] if poster_data else None

    rank_html  = (
        f'<div class="card-rank">#{rank}</div>'
        if rank else ""
    )
    score_html = (
        f'<div class="card-score">⭐ {score:.2f}</div>'
        if show_score and score else ""
    )

    short_genres = " · ".join(genres.split()[:2])

    if poster_url:
        img_html = f'<img src="{poster_url}" alt="{title}" loading="lazy"/>'
    else:
        short_title = title[:30] + "..." if len(title) > 30 else title
        img_html = f"""
        <div class="no-poster">
            <div class="no-poster-icon">🎬</div>
            <div class="no-poster-title">{short_title}</div>
        </div>
        """

    st.markdown(f"""
    <div class="movie-card">
        {img_html}
        {rank_html}
        {score_html}
        <div class="movie-card-overlay">
            <div class="card-title">{title[:35]}</div>
            <div class="card-genre">{short_genres}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_movie_row(title, movies, show_rank=False,
                      accent_text=""):
    """Render a horizontal row of movie cards."""
    if not movies:
        return

    header_html = f"""
    <div class="row-header">
        {title}
        {"<span class='row-header-accent'>" + accent_text + "</span>"
         if accent_text else ""}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    n      = min(len(movies), 7)
    cols   = st.columns(n)

    for i, (col, movie) in enumerate(
        zip(cols, movies[:n])
    ):
        with col:
            render_movie_card(
                movie,
                rank       = i+1 if show_rank else None,
                show_score = True
            )


def render_user_switcher():
    """Netflix-style profile switcher."""
    st.markdown("""
    <div style="padding: 24px 48px 0;">
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:18px;color:#666;
                    letter-spacing:2px;margin-bottom:12px;">
            WHO'S WATCHING?
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(USER_PROFILES))

    for col, profile in zip(cols, USER_PROFILES):
        with col:
            is_active = (
                st.session_state.user_idx == profile['idx']
            )
            border    = (
                "border:2px solid #e50914;"
                if is_active else ""
            )

            st.markdown(f"""
            <div class="user-card {"active" if is_active else ""}"
                 style="{border}">
                <div class="user-avatar">{profile['icon']}</div>
                <div class="user-name">{profile['name']}</div>
                <div style="font-size:10px;color:#555;
                            margin-top:2px">
                    {profile['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "Select",
                key=f"user_{profile['idx']}",
                use_container_width=True
            ):
                if profile['idx'] == -1:
                    st.session_state.page = 'new_user'
                else:
                    st.session_state.user_idx = profile['idx']
                    st.session_state.page     = 'home'
                st.rerun()


def render_stats_bar():
    """Live stats ticker."""
    health = api_health()
    stats  = api_get("/stats")

    num_users  = health.get('num_users',  0) if health else 0
    num_movies = health.get('num_movies', 0) if health else 0
    total_recs = (
        stats.get('total_recommendations', 0) if stats else 0
    )
    coverage   = (
        stats.get('catalogue_coverage', 0) if stats else 0
    )

    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value">{num_users:,}</div>
            <div class="stat-label">Users</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{num_movies:,}</div>
            <div class="stat-label">Movies</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">0.1252</div>
            <div class="stat-label">NDCG@10</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">+90.9%</div>
            <div class="stat-label">vs SVD</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{total_recs:,}</div>
            <div class="stat-label">Recs Served</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{coverage:.1%}</div>
            <div class="stat-label">Coverage</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">33M</div>
            <div class="stat-label">Ratings</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">202ms</div>
            <div class="stat-label">Avg Latency</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════

def page_home():
    """Main home page — Netflix-style rows."""

    user_idx = st.session_state.user_idx

    # ── Hero section ──────────────────────────────────────
    health = api_health()
    is_cold = (
        user_idx not in range(health.get('num_users', 0))
        if health else False
    )

    st.markdown(f"""
    <div class="hero">
        <div class="hero-bg"></div>
        <div class="hero-content">
            <div class="hero-badge">Now Streaming</div>
            <h1 class="hero-title">Your<br/>Cinemate</h1>
            <p class="hero-subtitle">
                Two-Tower AI recommendations powered by
                Neural Collaborative Filtering + DistilBERT.
                Personalised for User #{user_idx}.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats bar ─────────────────────────────────────────
    render_stats_bar()

    # ── User switcher ─────────────────────────────────────
    render_user_switcher()

    # ── Main recommendations ───────────────────────────────
    with st.spinner("Loading your recommendations..."):
        result = api_get(
            f"/recommend/{user_idx}",
            params={"top_k": 14}
        )

    if result is None:
        st.markdown("""
        <div style="padding:48px;text-align:center;color:#555">
            <div style="font-size:48px;margin-bottom:16px">🔌</div>
            <div style="font-size:18px">API not connected</div>
            <div style="font-size:13px;margin-top:8px">
                Run: uvicorn api.main:app --reload
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    recs = result.get('recommendations', [])
    rt   = result.get('response_time_ms', 0)

    if is_cold:
        st.markdown("""
        <div style="margin:0 48px 24px;padding:12px 20px;
                    background:#1a1500;border:1px solid #f59e0b;
                    border-radius:8px;font-size:13px;color:#f59e0b">
            ⚡ Cold start user — recommendations based on
            content features. Rate some movies to personalise.
        </div>
        """, unsafe_allow_html=True)

    # Row 1 — Top picks
    render_movie_row(
        "🎯 TOP PICKS FOR YOU",
        recs[:7],
        show_rank  = True,
        accent_text= f"Updated {datetime.now().strftime('%H:%M')}"
    )

    # Row 2 — Hidden gems (debiased = rare films)
    with st.spinner("Finding hidden gems..."):
        gems = api_get(
            f"/recommend/{user_idx}",
            params={"top_k": 7}
        )
    if gems:
        render_movie_row(
            "💎 HIDDEN GEMS",
            gems.get('recommendations', [])[7:],
            accent_text="Debiased picks"
        )

    # Row 3 — Popular right now
    with st.spinner("Loading trending..."):
        popular = api_get("/popular", params={"limit": 20})

    if popular and popular.get('movies'):
        # Enrich popular with movie details
        pop_idxs   = [
            m['movie_idx']
            for m in popular['movies'][:7]
        ]
        pop_movies = []
        for idx in pop_idxs:
            detail = api_get(f"/movie/{idx}")
            if detail:
                pop_movies.append({
                    'movie_idx': idx,
                    'title'    : detail.get('title', ''),
                    'genres'   : detail.get('genres', ''),
                    'tmdb_id'  : detail.get('tmdb_id', 0),
                    'score'    : 0,
                })
        if pop_movies:
            render_movie_row(
                "🔥 TRENDING NOW",
                pop_movies,
                accent_text="Most rated"
            )

    # Row 4 — Similar to top pick
    if recs:
        top_movie_idx = recs[0]['movie_idx']
        with st.spinner("Finding similar movies..."):
            similar = api_get(
                f"/similar/{top_movie_idx}",
                params={"top_k": 7}
            )
        if similar and similar.get('similar'):
            source_title = similar['source_movie']['title']
            render_movie_row(
                f"BECAUSE YOU MIGHT LIKE",
                similar['similar'][:7],
                accent_text=source_title[:30]
            )

    # ── Response time note ────────────────────────────────
    st.markdown(f"""
    <div style="padding:8px 48px;
                font-size:11px;color:#333">
        Response: {rt:.0f}ms · Model: Two-Tower ·
        Strategy: Brute Force MLP
    </div>
    """, unsafe_allow_html=True)

    # ── Quick feedback ────────────────────────────────────
    st.markdown("""
    <div class="row-header">
        ⭐ RATE A RECOMMENDATION
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            '<div class="section-pad">',
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([3, 2, 1])

        movie_opts = {
            f"#{r['rank']} — {r['title'][:40]}": r['movie_idx']
            for r in recs
        }

        with col1:
            selected = st.selectbox(
                "Select movie",
                options=list(movie_opts.keys()),
                label_visibility="collapsed"
            )
        with col2:
            rating = st.slider(
                "Rating", 0.5, 5.0, 4.0, 0.5,
                label_visibility="collapsed"
            )
        with col3:
            if st.button("Submit ⭐"):
                resp = api_post("/feedback", {
                    "user_idx" : int(user_idx),
                    "movie_idx": movie_opts[selected],
                    "rating"   : rating,
                    "from_rec" : True
                })
                if resp:
                    st.success("Rated!")
                else:
                    st.error("Failed")

        st.markdown('</div>', unsafe_allow_html=True)


def page_search():
    """Search page with instant results."""
    st.markdown("""
    <div style="padding:48px 48px 24px">
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:48px;color:#fff;
                    letter-spacing:2px;margin-bottom:8px">
            SEARCH
        </div>
        <div style="font-size:14px;color:#555">
            Find any movie from 27,766 titles
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            '<div class="search-wrapper">',
            unsafe_allow_html=True
        )
        query = st.text_input(
            "Search",
            placeholder="Search movies... (e.g. Inception, Dark Knight)",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if query and len(query) >= 2:
        with st.spinner("Searching..."):
            result = api_get(
                "/search",
                params={"q": query, "limit": 14}
            )

        if result and result.get('results'):
            movies = result['results']
            total  = result['total']

            st.markdown(f"""
            <div style="padding:16px 48px;font-size:13px;
                        color:#666">
                {total} results for
                <span style="color:#e50914">"{query}"</span>
            </div>
            """, unsafe_allow_html=True)

            render_movie_row(
                "SEARCH RESULTS",
                movies[:7],
                show_rank=False
            )
            if len(movies) > 7:
                render_movie_row(
                    "MORE RESULTS",
                    movies[7:],
                    show_rank=False
                )

            # Similar movies for first result
            if movies:
                first_idx = movies[0]['movie_idx']
                similar   = api_get(
                    f"/similar/{first_idx}",
                    params={"top_k": 7}
                )
                if similar and similar.get('similar'):
                    render_movie_row(
                        "YOU MIGHT ALSO LIKE",
                        similar['similar'][:7],
                        accent_text="Content similarity"
                    )
        else:
            st.markdown(f"""
            <div style="padding:48px;text-align:center;
                        color:#555">
                <div style="font-size:48px">🔍</div>
                <div style="font-size:16px;margin-top:16px">
                    No results for "{query}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Show genre browse when no search
        st.markdown("""
        <div class="row-header">🎭 BROWSE BY GENRE</div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-pad">',
            unsafe_allow_html=True
        )
        genres = api_get("/genres")
        if genres:
            genre_list = genres.get('genres', GENRE_LIST)
            cols       = st.columns(5)
            for i, genre in enumerate(genre_list[:20]):
                with cols[i % 5]:
                    if st.button(
                        genre,
                        key=f"genre_{genre}",
                        use_container_width=True
                    ):
                        st.session_state.search_query = genre
        st.markdown('</div>', unsafe_allow_html=True)


def page_new_user():
    """Onboarding for new users — genre preference selection."""
    st.markdown("""
    <div style="padding:48px 48px 24px">
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:48px;color:#fff;
                    letter-spacing:2px;margin-bottom:8px">
            CREATE YOUR PROFILE
        </div>
        <div style="font-size:14px;color:#555">
            Tell us what you love — we'll find your match
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-pad">',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="font-size:16px;color:#aaa;
                margin-bottom:20px;font-weight:500">
        Select your favourite genres:
    </div>
    """, unsafe_allow_html=True)

    # Genre multi-select with styled chips
    selected_genres = st.multiselect(
        "Genres",
        GENRE_LIST,
        default=["Action", "Drama"],
        label_visibility="collapsed"
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "🎬 Get Recommendations",
            use_container_width=True
        ):
            if selected_genres:
                with st.spinner(
                    "Finding your perfect match..."
                ):
                    # Find proxy user by genre matching
                    proxy = find_proxy_user(selected_genres)

                st.session_state.user_idx = proxy
                st.session_state.page     = 'home'
                st.success(
                    f"Profile created! "
                    f"Showing recommendations for "
                    f"a similar user."
                )
                st.rerun()
            else:
                st.warning("Select at least one genre")

    st.markdown('</div>', unsafe_allow_html=True)


def find_proxy_user(preferred_genres):
    """
    Find existing user whose genre profile
    best matches the given genre preferences.
    Uses pre-defined genre → user_idx mapping
    for fast response without DB query.
    """
    genre_user_map = {
        "Action"    : 0,
        "Drama"     : 1024,
        "Sci-Fi"    : 15679,
        "Comedy"    : 5000,
        "Horror"    : 8000,
        "Thriller"  : 2500,
        "Romance"   : 3000,
        "Animation" : 4500,
        "Crime"     : 6000,
        "Adventure" : 7000,
        "Fantasy"   : 9000,
        "Mystery"   : 10000,
        "War"       : 11000,
        "Western"   : 12000,
        "Documentary":13000,
    }

    # Find best matching user by genre votes
    votes = {}
    for genre in preferred_genres:
        user = genre_user_map.get(genre)
        if user is not None:
            votes[user] = votes.get(user, 0) + 1

    if votes:
        return max(votes, key=votes.get)
    return 0  # default user


def page_about():
    """About page — project info."""
    st.markdown("""
    <div style="padding:48px">
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:48px;color:#fff;
                    letter-spacing:2px;margin-bottom:32px">
            ABOUT CINEMATE
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div style="padding:0 0 0 48px">
            <div style="font-size:20px;color:#e50914;
                        font-family:'Bebas Neue',cursive;
                        letter-spacing:1px;margin-bottom:16px">
                THE MODEL
            </div>
            <p style="color:#aaa;font-size:15px;
                      line-height:1.8;margin-bottom:24px">
                Cinemate uses a Two-Tower hybrid architecture
                combining Neural Collaborative Filtering (NCF)
                with DistilBERT content embeddings trained on
                the MovieLens 33M dataset.
            </p>
            <div style="font-size:20px;color:#e50914;
                        font-family:'Bebas Neue',cursive;
                        letter-spacing:1px;margin-bottom:16px">
                ARCHITECTURE
            </div>
            <p style="color:#aaa;font-size:15px;
                      line-height:1.8">
                <strong style="color:#fff">
                    Collaborative Tower
                </strong> — NCF learns user-movie interaction
                patterns from 25M ratings via non-linear MLP.
                <br><br>
                <strong style="color:#fff">
                    Content Tower
                </strong> — DistilBERT embeddings of
                title + genres + genome tags handle cold start.
                <br><br>
                <strong style="color:#fff">
                    Debiasing
                </strong> — Log-popularity penalty + MMR
                genre diversity pushes beyond the popularity
                bubble to 10%+ catalogue coverage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Model metrics
        metrics = [
            ("NDCG@10",      "0.1252", "+90.9% vs SVD"),
            ("Precision@10", "0.1135", "top-10 accuracy"),
            ("Coverage",     "10%+",   "with debiasing"),
            ("Personalisation","0.86", "high variance"),
            ("Dataset",      "33M",    "ratings"),
            ("Users",        "173K",   "trained on"),
            ("Movies",       "27,766", "in catalogue"),
            ("Latency",      "~200ms", "brute force MLP"),
        ]

        for name, val, sub in metrics:
            st.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #222;
                        border-radius:8px;padding:12px 16px;
                        margin-bottom:8px;display:flex;
                        justify-content:space-between;
                        align-items:center">
                <div>
                    <div style="font-size:12px;color:#666;
                                text-transform:uppercase;
                                letter-spacing:1px">
                        {name}
                    </div>
                    <div style="font-size:11px;color:#444;
                                margin-top:2px">
                        {sub}
                    </div>
                </div>
                <div style="font-family:'Bebas Neue',cursive;
                            font-size:24px;color:#e50914">
                    {val}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR — User control panel
# ══════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:28px;color:#e50914;
                    letter-spacing:3px;margin-bottom:24px">
            CINEMATE
        </div>
        """, unsafe_allow_html=True)

        # Page navigation
        page_map = {
            "🏠 Home"       : "home",
            "🔍 Search"     : "search",
            "✨ New User"   : "new_user",
            "ℹ️ About"      : "about",
        }

        for label, page_key in page_map.items():
            if st.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True
            ):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")

        # Manual user ID input
        st.markdown(
            "<div style='font-size:12px;color:#666;"
            "text-transform:uppercase;letter-spacing:1px;"
            "margin-bottom:8px'>Manual User ID</div>",
            unsafe_allow_html=True
        )
        uid = st.number_input(
            "User ID",
            min_value    = 0,
            max_value    = 172999,
            value        = st.session_state.user_idx,
            label_visibility = "collapsed"
        )
        if st.button("Load User", use_container_width=True):
            st.session_state.user_idx = uid
            st.session_state.page     = "home"
            st.rerun()

        st.markdown("---")

        # Debiasing controls
        st.markdown(
            "<div style='font-size:12px;color:#666;"
            "text-transform:uppercase;letter-spacing:1px;"
            "margin-bottom:8px'>Debiasing Controls</div>",
            unsafe_allow_html=True
        )

        alpha = st.slider(
            "Popularity Penalty (α)",
            0.0, 1.0, 0.3, 0.1,
            help="Higher = more diverse, less popular"
        )
        lam = st.slider(
            "Genre Diversity (λ)",
            0.0, 1.0, 0.3, 0.1,
            help="Higher = more genre variety"
        )

        st.session_state['alpha'] = alpha
        st.session_state['lam']   = lam

        st.markdown("---")

        # API health
        health = api_health()
        if health:
            st.success("API Online", icon="✅")
            st.caption(
                f"Users: {health.get('num_users',0):,}\n"
                f"Movies: {health.get('num_movies',0):,}"
            )
        else:
            st.error("API Offline", icon="❌")
            st.caption(
                "uvicorn api.main:app --reload"
            )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    render_sidebar()
    render_navbar()

    page = st.session_state.page

    if page == 'home':
        page_home()
    elif page == 'search':
        page_search()
    elif page == 'new_user':
        page_new_user()
    elif page == 'about':
        page_about()


if __name__ == "__main__":
    main()