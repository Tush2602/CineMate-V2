"""
streamlit_app.py — Cinemate V2
════════════════════════════════════════════════════════════
A Netflix/Prime-style movie recommendation frontend
powered by a Two-Tower hybrid deep learning model.

Dataset  : MovieLens 33M
Model    : NCF + DistilBERT Two-Tower (MLP fusion)
Backend  : FastAPI  → http://localhost:8000
Run      : streamlit run app/streamlit_app.py
════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Path setup ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
PLOTS_DIR  = ROOT_DIR / "data" / "processed" / "plots"
MODELS_DIR = ROOT_DIR / "models"

sys.path.append(str(ROOT_DIR))

# ── API / TMDB config ──────────────────────────────────────
API_URL      = os.getenv("API_URL",      "http://localhost:8000")
TMDB_API_KEY = "be50cfdf6bda073b5c74b0a42f7e6dca"
TMDB_IMG_W   = "https://image.tmdb.org/t/p/w342"
TMDB_IMG_H   = "https://image.tmdb.org/t/p/w1280"

# ══════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cinemate V2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?
  family=Bebas+Neue
  &family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300
  &display=swap');

/* ── Base ── */
html, body, .stApp {
    background : #0c0c0f !important;
    color      : #e2e2e2 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding    : 0 !important;
    max-width  : 100% !important;
}

/* ── Navbar ── */
.cine-nav {
    position        : sticky;
    top             : 0;
    z-index         : 9999;
    display         : flex;
    align-items     : center;
    justify-content : space-between;
    padding         : 14px 48px;
    background      : linear-gradient(180deg,
                      rgba(12,12,15,0.98) 0%,
                      rgba(12,12,15,0.0) 100%);
    backdrop-filter : blur(12px);
    border-bottom   : 1px solid rgba(255,255,255,0.04);
}
.cine-logo {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 28px;
    letter-spacing: 4px;
    background   : linear-gradient(135deg,#e50914,#ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    cursor       : default;
    user-select  : none;
}
.cine-badge {
    font-size    : 9px;
    letter-spacing: 2px;
    color        : #e50914;
    font-weight  : 700;
    vertical-align: super;
    font-family  : 'DM Sans', sans-serif;
}
.nav-pill-row {
    display : flex;
    gap     : 6px;
}
.nav-pill {
    padding       : 7px 18px;
    border-radius : 24px;
    border        : 1px solid rgba(255,255,255,0.1);
    background    : rgba(255,255,255,0.04);
    color         : #aaa;
    font-size     : 13px;
    font-weight   : 500;
    cursor        : pointer;
    transition    : all .2s;
    white-space   : nowrap;
}
.nav-pill:hover  { background:rgba(229,9,20,.15); border-color:#e50914; color:#fff; }
.nav-pill.active { background:#e50914; border-color:#e50914; color:#fff; }
.nav-api-dot     { display:flex; align-items:center; gap:6px;
                   font-size:12px; color:#555; }
.dot-green { width:7px; height:7px; border-radius:50%;
             background:#22c55e; display:inline-block;
             box-shadow:0 0 6px #22c55e; }
.dot-red   { width:7px; height:7px; border-radius:50%;
             background:#ef4444; display:inline-block; }

/* ── Hero ── */
.hero-wrap {
    position      : relative;
    min-height    : 480px;
    display       : flex;
    align-items   : flex-end;
    padding       : 0 48px 60px;
    overflow      : hidden;
    margin-top    : -60px;
}
.hero-bg {
    position   : absolute; inset:0;
    background : linear-gradient(
        120deg,
        #0c0c0f 0%,
        rgba(30,10,10,0.85) 50%,
        transparent 100%
    ), linear-gradient(180deg, transparent 40%, #0c0c0f 100%);
    z-index    : 1;
}
.hero-particles {
    position   : absolute; inset:0;
    background : radial-gradient(ellipse at 70% 50%,
                 rgba(229,9,20,0.08) 0%, transparent 60%);
    z-index    : 0;
}
.hero-content { position:relative; z-index:2; max-width:560px; }
.hero-eyebrow {
    font-size      : 11px;
    letter-spacing : 3px;
    color          : #e50914;
    font-weight    : 700;
    text-transform : uppercase;
    margin-bottom  : 12px;
}
.hero-title {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 76px;
    line-height  : .95;
    color        : #fff;
    margin       : 0 0 20px;
    letter-spacing: 2px;
    text-shadow  : 0 4px 40px rgba(0,0,0,.6);
}
.hero-title span { color:#e50914; }
.hero-sub {
    font-size   : 15px;
    color       : #888;
    line-height : 1.7;
    margin      : 0 0 32px;
    font-weight : 300;
}
.hero-chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip {
    padding      : 5px 14px;
    border-radius: 20px;
    border       : 1px solid rgba(255,255,255,0.12);
    font-size    : 12px;
    color        : #ccc;
    background   : rgba(255,255,255,0.04);
}
.chip.red { border-color:#e50914; color:#e50914;
             background:rgba(229,9,20,0.08); }

/* ── Stats ticker ── */
.stats-ticker {
    display        : flex;
    gap            : 0;
    background     : #0a0a0d;
    border-top     : 1px solid #1a1a1f;
    border-bottom  : 1px solid #1a1a1f;
    overflow-x     : auto;
    margin-bottom  : 8px;
    padding        : 0 48px;
}
.stat-cell {
    display        : flex;
    flex-direction : column;
    align-items    : center;
    padding        : 14px 28px;
    border-right   : 1px solid #1a1a1f;
    min-width      : 100px;
}
.stat-cell:last-child { border-right:none; }
.stat-v {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 24px;
    color        : #e50914;
    line-height  : 1;
}
.stat-l {
    font-size      : 10px;
    color          : #444;
    margin-top     : 3px;
    text-transform : uppercase;
    letter-spacing : .5px;
    white-space    : nowrap;
}

/* ── Section header ── */
.sec-hdr {
    display     : flex;
    align-items : baseline;
    gap         : 12px;
    padding     : 32px 48px 14px;
}
.sec-title {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 22px;
    color        : #e8e8e8;
    letter-spacing: 1px;
}
.sec-sub {
    font-size  : 12px;
    color      : #e50914;
    font-weight: 600;
    letter-spacing: 1px;
}

/* ── Movie card ── */
.movie-card-wrap { position:relative; }
.mc {
    position      : relative;
    border-radius : 8px;
    overflow      : hidden;
    background    : #161619;
    border        : 1px solid rgba(255,255,255,0.04);
    transition    : transform .25s ease, box-shadow .25s ease,
                    border-color .25s ease;
    cursor        : pointer;
    aspect-ratio  : 2/3;
}
.mc:hover {
    transform    : scale(1.04) translateY(-6px);
    box-shadow   : 0 24px 64px rgba(0,0,0,.8);
    border-color : rgba(229,9,20,0.4);
    z-index      : 20;
}
.mc img {
    width:100%; height:100%;
    object-fit:cover; display:block;
}
.mc-overlay {
    position   : absolute; inset:0;
    background : linear-gradient(transparent 45%, rgba(0,0,0,.95) 100%);
    opacity    : 0;
    transition : opacity .25s;
    padding    : 12px;
    display    : flex;
    flex-direction: column;
    justify-content: flex-end;
}
.mc:hover .mc-overlay { opacity:1; }
.mc-title {
    font-size    : 13px;
    font-weight  : 600;
    color        : #fff;
    white-space  : nowrap;
    overflow     : hidden;
    text-overflow: ellipsis;
}
.mc-genre {
    font-size : 11px;
    color     : #999;
    margin-top: 3px;
}
.mc-rank {
    position     : absolute;
    top          : 8px; left:8px;
    background   : #e50914;
    color        : #fff;
    font-size    : 11px;
    font-weight  : 700;
    padding      : 3px 8px;
    border-radius: 4px;
    font-family  : 'Bebas Neue', cursive;
    letter-spacing: 1px;
}
.mc-score {
    position     : absolute;
    top          : 8px; right:8px;
    background   : rgba(0,0,0,0.75);
    color        : #f5c518;
    font-size    : 11px;
    font-weight  : 700;
    padding      : 3px 7px;
    border-radius: 4px;
    backdrop-filter: blur(4px);
}
.mc-tail {
    position     : absolute;
    bottom       : 8px; right:8px;
    background   : rgba(34,197,94,0.2);
    border       : 1px solid rgba(34,197,94,0.5);
    color        : #22c55e;
    font-size    : 9px;
    font-weight  : 700;
    padding      : 2px 6px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.mc-noposter {
    width           : 100%;
    height          : 100%;
    display         : flex;
    flex-direction  : column;
    align-items     : center;
    justify-content : center;
    background      : linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding         : 16px;
    text-align      : center;
}
.mc-noposter-icon  { font-size:40px; margin-bottom:10px; }
.mc-noposter-title { font-size:12px; color:#ccc; font-weight:500;
                     line-height:1.4; }

/* ── Profile switcher ── */
.who-label {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 16px;
    letter-spacing: 3px;
    color        : #444;
    padding      : 20px 48px 12px;
}
.profile-card {
    background    : #161619;
    border        : 2px solid transparent;
    border-radius : 10px;
    padding       : 16px 8px;
    text-align    : center;
    cursor        : pointer;
    transition    : all .2s;
}
.profile-card:hover  { border-color:#e50914; background:#1c1015; }
.profile-card.active { border-color:#e50914; background:#1c1015; }
.profile-icon { font-size:36px; }
.profile-name { font-size:13px; color:#ddd; font-weight:600;
                margin-top:6px; }
.profile-desc { font-size:10px; color:#555; margin-top:2px;
                text-transform:uppercase; letter-spacing:.5px; }

/* ── Buttons ── */
.stButton > button {
    background    : #e50914 !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : 6px !important;
    font-weight   : 600 !important;
    font-family   : 'DM Sans', sans-serif !important;
    transition    : all .2s !important;
    padding       : 8px 20px !important;
}
.stButton > button:hover {
    background : #b20710 !important;
    transform  : translateY(-1px) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background    : #161619 !important;
    border        : 1px solid #2a2a30 !important;
    border-radius : 8px !important;
    color         : #e2e2e2 !important;
    font-family   : 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color  : #e50914 !important;
    box-shadow    : 0 0 0 2px rgba(229,9,20,.2) !important;
}
.stSelectbox > div > div {
    background  : #161619 !important;
    border      : 1px solid #2a2a30 !important;
    color       : #e2e2e2 !important;
}
.stMultiSelect > div {
    background  : #161619 !important;
    border      : 1px solid #2a2a30 !important;
}
.stSlider > div > div > div { background:#e50914 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background  : transparent !important;
    border-bottom: 1px solid #1a1a1f !important;
    gap         : 4px;
    padding     : 0 48px;
}
.stTabs [data-baseweb="tab"] {
    background   : transparent !important;
    color        : #555 !important;
    font-family  : 'DM Sans', sans-serif !important;
    font-weight  : 500 !important;
    border-bottom: 2px solid transparent !important;
    padding      : 12px 20px !important;
}
.stTabs [aria-selected="true"] {
    color        : #fff !important;
    border-bottom: 2px solid #e50914 !important;
}

/* ── Expander ── */
.stExpander {
    background    : #161619 !important;
    border        : 1px solid #2a2a30 !important;
    border-radius : 8px !important;
}

/* ── Metric ── */
div[data-testid="stMetric"] {
    background    : #161619 !important;
    border        : 1px solid #2a2a30 !important;
    border-radius : 8px !important;
    padding       : 16px !important;
}
div[data-testid="stMetricValue"] { color:#e50914 !important; }
div[data-testid="stMetricLabel"] { color:#555 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color:#e50914 !important; }

/* ── Info / warning ── */
.stAlert { border-radius:8px !important; }

/* ── Divider ── */
hr { border-color:#1a1a1f !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:#0c0c0f; }
::-webkit-scrollbar-thumb { background:#2a2a30; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#444; }

/* ── Plot image ── */
.plot-card {
    background    : #161619;
    border        : 1px solid #2a2a30;
    border-radius : 10px;
    padding       : 16px;
    margin-bottom : 16px;
}
.plot-title {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 16px;
    color        : #e50914;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.plot-desc {
    font-size   : 13px;
    color       : #777;
    line-height : 1.6;
    margin-bottom: 12px;
}
.insight-box {
    background    : rgba(229,9,20,0.06);
    border-left   : 3px solid #e50914;
    border-radius : 0 6px 6px 0;
    padding       : 10px 14px;
    font-size     : 13px;
    color         : #ccc;
    line-height   : 1.6;
    margin-top    : 8px;
}

/* ── About card ── */
.about-card {
    background    : #161619;
    border        : 1px solid #2a2a30;
    border-radius : 12px;
    padding       : 24px;
    margin-bottom : 16px;
}
.about-section-title {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 18px;
    color        : #e50914;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.tech-badge {
    display      : inline-block;
    background   : rgba(229,9,20,0.1);
    border       : 1px solid rgba(229,9,20,0.3);
    color        : #e50914;
    font-size    : 12px;
    font-weight  : 600;
    padding      : 4px 12px;
    border-radius: 20px;
    margin       : 3px;
}
.metric-row {
    display         : flex;
    justify-content : space-between;
    align-items     : center;
    padding         : 10px 0;
    border-bottom   : 1px solid #1a1a1f;
}
.metric-row:last-child { border-bottom:none; }
.metric-key { font-size:13px; color:#666; }
.metric-val {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 20px;
    color        : #e50914;
}

/* ── New user onboarding ── */
.onboard-wrap {
    max-width  : 720px;
    margin     : 0 auto;
    padding    : 48px;
}
.onboard-title {
    font-family  : 'Bebas Neue', cursive;
    font-size    : 56px;
    color        : #fff;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.onboard-sub {
    font-size   : 15px;
    color       : #666;
    margin-bottom: 32px;
    font-weight : 300;
}
.genre-grid {
    display              : grid;
    grid-template-columns: repeat(5, 1fr);
    gap                  : 10px;
    margin-bottom        : 32px;
}
.genre-btn {
    padding       : 12px 8px;
    border-radius : 8px;
    border        : 1px solid #2a2a30;
    background    : #161619;
    color         : #aaa;
    font-size     : 13px;
    font-weight   : 500;
    text-align    : center;
    cursor        : pointer;
    transition    : all .2s;
}
.genre-btn:hover  { border-color:#e50914; background:#1c1015; color:#fff; }
.genre-btn.active { border-color:#e50914; background:#e50914; color:#fff; }

/* ── Cold start banner ── */
.cold-banner {
    margin        : 0 48px 16px;
    padding       : 12px 20px;
    background    : rgba(245,158,11,0.08);
    border        : 1px solid rgba(245,158,11,0.3);
    border-radius : 8px;
    font-size     : 13px;
    color         : #f59e0b;
}

/* ── Footer ── */
.cine-footer {
    margin-top  : 60px;
    padding     : 24px 48px;
    border-top  : 1px solid #1a1a1f;
    display     : flex;
    justify-content: space-between;
    align-items : center;
    font-size   : 12px;
    color       : #333;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════
defaults = {
    "page"          : "home",
    "user_idx"      : 0,
    "user_label"    : "Alex",
    "user_icon"     : "🎬",
    "selected_genres": [],
    "my_list"       : [],
    "search_query"  : "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# PROFILES & CONSTANTS
# ══════════════════════════════════════════════════════════
PROFILES = [
    {"name":"Alex",   "icon":"🎬","idx":0,      "desc":"Action fan",    "genres":["Action","Thriller"]},
    {"name":"Sam",    "icon":"🎭","idx":1024,    "desc":"Drama lover",   "genres":["Drama","Romance"]},
    {"name":"Jordan", "icon":"🚀","idx":15679,   "desc":"Sci-Fi geek",   "genres":["Sci-Fi","Fantasy"]},
    {"name":"Morgan", "icon":"😂","idx":5000,    "desc":"Comedy first",  "genres":["Comedy","Animation"]},
    {"name":"Casey",  "icon":"👻","idx":8000,    "desc":"Horror fan",    "genres":["Horror","Mystery"]},
    {"name":"Riley",  "icon":"🎵","idx":12000,   "desc":"Indie lover",   "genres":["Documentary","Drama"]},
]

GENRE_PROXY = {
    "Action":9000,"Adventure":7000,"Animation":4500,"Comedy":5000,
    "Crime":6000,"Documentary":13000,"Drama":1024,"Fantasy":15679,
    "Horror":8000,"Mystery":10000,"Romance":3000,"Sci-Fi":15679,
    "Thriller":2500,"War":11000,"Western":12000,
}

ALL_GENRES = sorted(GENRE_PROXY.keys())

# ══════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def api_get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_URL}{endpoint}",
                         params=params, timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(endpoint: str, data: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}",
                          json=data, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_tmdb(tmdb_id: int) -> dict:
    if not TMDB_API_KEY or not tmdb_id:
        return {}
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY}, timeout=6)
        if r.status_code == 200:
            d = r.json()
            return {
                "poster"  : f"{TMDB_IMG_W}{d['poster_path']}"   if d.get("poster_path")   else "",
                "backdrop": f"{TMDB_IMG_H}{d['backdrop_path']}" if d.get("backdrop_path") else "",
                "overview": d.get("overview",""),
                "year"    : d.get("release_date","")[:4],
                "tmdb_rating": d.get("vote_average",0),
                "tagline" : d.get("tagline",""),
            }
    except Exception:
        pass
    return {}

@st.cache_data(ttl=30)
def health_check():
    return api_get("/health")

# ══════════════════════════════════════════════════════════
# NAVIGATION HELPER
# ══════════════════════════════════════════════════════════
def go(page: str):
    st.session_state.page = page
    st.rerun()

# ══════════════════════════════════════════════════════════
# NAVBAR (rendered on every page)
# ══════════════════════════════════════════════════════════
def render_navbar():
    health = health_check()
    online = health is not None
    dot    = '<span class="dot-green"></span>' if online \
             else '<span class="dot-red"></span>'
    api_txt = "API Online" if online else "API Offline"

    current = st.session_state.page

    st.markdown(f"""
    <div class="cine-nav">
      <div class="cine-logo">
        CINEMATE<span class="cine-badge"> V2</span>
      </div>
      <div class="nav-pill-row" id="navpills">
        <!-- pills injected via Streamlit buttons below -->
      </div>
      <div class="nav-api-dot">{dot} {api_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Actual working nav buttons
    # nav_col = st.columns([1,1,1,1,1,6])
    nav_col = st.columns(5)
    pages = [("🏠 Home","home"),("🔍 Search","search"),
             ("✨ New User","new_user"),("📋 My List","mylist"),
             ("ℹ️ About","about")]
    for i, (label, pg) in enumerate(pages):
        with nav_col[i]:
            if st.button(label, key=f"nav_{pg}",
                         use_container_width=True):
                go(pg)

# ══════════════════════════════════════════════════════════
# STATS TICKER
# ══════════════════════════════════════════════════════════
def render_stats():
    health = health_check()
    stats  = api_get("/stats")
    nu     = health.get("num_users",  173134) if health else 173134
    nm     = health.get("num_movies", 27766)  if health else 27766
    tr     = stats.get("total_recommendations", 0) if stats else 0
    cv     = stats.get("catalogue_coverage",    0) if stats else 0

    st.markdown(f"""
    <div class="stats-ticker">
      <div class="stat-cell">
        <div class="stat-v">{nu:,}</div>
        <div class="stat-l">Users Trained</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">{nm:,}</div>
        <div class="stat-l">Movies</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">33M</div>
        <div class="stat-l">Ratings</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">0.1252</div>
        <div class="stat-l">NCF NDCG@10</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">+90.9%</div>
        <div class="stat-l">vs SVD Baseline</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">5.4%</div>
        <div class="stat-l">Catalogue Coverage</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">{tr:,}</div>
        <div class="stat-l">Recs Served</div>
      </div>
      <div class="stat-cell">
        <div class="stat-v">~200ms</div>
        <div class="stat-l">Avg Latency</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MOVIE CARD RENDERER
# ══════════════════════════════════════════════════════════
def render_card(movie: dict, rank: int = None,
                is_tail: bool = False):
    tmdb_id  = movie.get("tmdb_id", 0) or 0
    title    = movie.get("title",   "Unknown")
    genres   = movie.get("genres",  "")
    score    = movie.get("score",   0)

    tmdb     = fetch_tmdb(int(tmdb_id)) if tmdb_id else {}
    poster   = tmdb.get("poster", "")
    year     = tmdb.get("year",   "")

    rank_html  = f'<div class="mc-rank">#{rank}</div>' if rank else ""
    score_html = f'<div class="mc-score">⭐ {score:.2f}</div>' \
                 if score else ""
    tail_html  = '<div class="mc-tail">DISCOVERY</div>' if is_tail else ""
    short_g    = " · ".join(genres.split()[:2]) if genres else ""
    yr_txt     = f" ({year})" if year else ""

    if poster:
        img_html = f'<img src="{poster}" alt="{title}" loading="lazy"/>'
    else:
        short_t = (title[:28] + "…") if len(title) > 28 else title
        # img_html = f"""
        # <div class="mc-noposter">
        #   <div class="mc-noposter-icon">🎬</div>
        #   <div class="mc-noposter-title">{short_t}</div>
        # </div>"""
        img_html = f"""
        <img src="https://via.placeholder.com/300x450?text=No+Image" />
        """


    st.markdown(f"""
    <div class="movie-card-wrap">
    <div class="mc">
        {img_html}
        {rank_html}{score_html}{tail_html}
        <div class="mc-overlay">
        <div class="mc-title">{title[:36]}{yr_txt}</div>
        <div class="mc-genre">{short_g}</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def render_row(section_title: str, movies: list,
               show_rank: bool = False,
               accent: str = "",
               tail_start: int = 999):
    if not movies:
        return
    st.markdown(f"""
    <div class="sec-hdr">
      <span class="sec-title">{section_title}</span>
      {"<span class='sec-sub'>" + accent + "</span>" if accent else ""}
    </div>
    """, unsafe_allow_html=True)

    n    = min(len(movies), 7)
    cols = st.columns(n, gap="small")
    for i, (col, mv) in enumerate(zip(cols, movies[:n])):
        with col:
            render_card(
                mv,
                rank    = i+1 if show_rank else None,
                is_tail = (i >= tail_start),
            )

# ══════════════════════════════════════════════════════════
# PROFILE SWITCHER
# ══════════════════════════════════════════════════════════
def render_profiles():
    st.markdown('<div class="who-label">WHO\'S WATCHING?</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(PROFILES) + 1)

    for i, p in enumerate(PROFILES):
        with cols[i]:
            active = st.session_state.user_idx == p["idx"]
            border = "border:2px solid #e50914;" if active else ""
            st.markdown(f"""
            <div class="profile-card {"active" if active else ""}"
                 style="{border}">
              <div class="profile-icon">{p["icon"]}</div>
              <div class="profile-name">{p["name"]}</div>
              <div class="profile-desc">{p["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Switch", key=f"prof_{p['idx']}",
                         use_container_width=True):
                st.session_state.user_idx   = p["idx"]
                st.session_state.user_label = p["name"]
                st.session_state.user_icon  = p["icon"]
                st.rerun()

    with cols[len(PROFILES)]:
        st.markdown("""
        <div class="profile-card">
          <div class="profile-icon">✨</div>
          <div class="profile-name">New User</div>
          <div class="profile-desc">Create profile</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Create", key="prof_new",
                     use_container_width=True):
            go("new_user")

# ══════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════
def page_home():
    uid   = st.session_state.user_idx
    uname = st.session_state.user_label
    uicon = st.session_state.user_icon

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-particles"></div>
      <div class="hero-bg"></div>
      <div class="hero-content">
        <div class="hero-eyebrow">Now Personalising</div>
        <h1 class="hero-title">
          {uicon} Hello,<br><span>{uname}</span>
        </h1>
        <p class="hero-sub">
          Your Two-Tower AI engine has scanned 33 million ratings
          to surface films curated exactly for you.<br>
          Top picks use deep collaborative signals. Discovery
          picks surface hidden gems from the long tail.
        </p>
        <div class="hero-chips">
          <span class="chip red">Two-Tower Model</span>
          <span class="chip">NDCG@10: 0.1252</span>
          <span class="chip">User #{uid}</span>
          <span class="chip">7 Personalised + 3 Discovery</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_stats()
    render_profiles()

    # Fetch recommendations
    with st.spinner("Loading your recommendations…"):
        result = api_get(f"/recommend/{uid}",
                         params={"top_k": 14})

    if result is None:
        st.markdown("""
        <div style="padding:60px 48px;text-align:center;">
          <div style="font-size:56px">🔌</div>
          <div style="font-family:'Bebas Neue',cursive;
                      font-size:32px;color:#e50914;
                      letter-spacing:2px;margin-top:16px">
            API NOT CONNECTED
          </div>
          <div style="color:#555;margin-top:8px;font-size:14px">
            Run: <code>uvicorn api.main:app --reload</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    recs    = result.get("recommendations", [])
    rt      = result.get("response_time_ms", 0)
    is_cold = result.get("is_cold_start", False)

    if is_cold:
        st.markdown("""
        <div class="cold-banner">
          ⚡ <strong>Cold Start User</strong> — No rating history found.
          Recommendations are based on content features (title, genre,
          tags). Rate some movies below to personalise your experience.
        </div>
        """, unsafe_allow_html=True)

    # Row 1 — Top picks (head slots = first 7)
    render_row(
        "🎯 TOP PICKS FOR YOU",
        recs[:7],
        show_rank  = True,
        accent     = f"Model-scored · {rt:.0f}ms",
    )

    # Row 2 — Discovery (tail slots = last 3 if API returns them)
    if len(recs) > 7:
        render_row(
            "💎 DISCOVERY — HIDDEN GEMS",
            recs[7:],
            show_rank  = False,
            accent     = "Long-tail exploration",
            tail_start = 0,
        )

    # Row 3 — Trending (popular endpoint)
    popular = api_get("/popular", params={"limit": 21})
    if popular and popular.get("movies"):
        pop_enriched = []
        for m in popular["movies"][:7]:
            detail = api_get(f"/movie/{m['movie_idx']}")
            if detail:
                pop_enriched.append({
                    "movie_idx": m["movie_idx"],
                    "title"    : detail.get("title",""),
                    "genres"   : detail.get("genres",""),
                    "tmdb_id"  : detail.get("tmdb_id",0),
                    "score"    : 0,
                })
        if pop_enriched:
            render_row("🔥 TRENDING NOW", pop_enriched,
                       accent="Most rated in dataset")

    # Row 4 — Similar to #1 pick
    if recs:
        sim = api_get(f"/similar/{recs[0]['movie_idx']}",
                      params={"top_k": 7})
        if sim and sim.get("similar"):
            src = sim["source_movie"]["title"][:30]
            render_row(
                "BECAUSE YOU MIGHT LIKE",
                sim["similar"][:7],
                accent=src,
            )

    # ── Feedback section ──────────────────────────────────
    st.markdown('<div class="sec-hdr"><span class="sec-title">⭐ RATE A RECOMMENDATION</span></div>',
                unsafe_allow_html=True)
    with st.container():
        st.markdown('<div style="padding:0 48px 24px">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([3,2,1,1])
        opts = {f"#{r['rank']} — {r['title'][:40]}": r["movie_idx"]
                for r in recs}
        with c1:
            sel = st.selectbox("Movie", list(opts.keys()),
                               label_visibility="collapsed")
        with c2:
            rating = st.slider("Rating", 0.5, 5.0, 4.0, 0.5,
                                label_visibility="collapsed")
        with c3:
            st.write("")
            if st.button("Submit ⭐", use_container_width=False):
                r = api_post("/feedback", {
                    "user_idx" : int(uid),
                    "movie_idx": opts[sel],
                    "rating"   : rating,
                    "from_rec" : True,
                })
                if r:
                    st.success("Rating saved!")
                else:
                    st.error("Could not save")
        with c4:
            if recs and st.button("➕ My List", use_container_width=False):
                mv = next((r for r in recs
                           if opts[sel] == r["movie_idx"]), None)
                if mv and mv not in st.session_state.my_list:
                    st.session_state.my_list.append(mv)
                    st.success("Added!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="cine-footer">
      <span>Cinemate V2 · Two-Tower Hybrid Recommender</span>
      <span>Response: {rt:.0f}ms · User #{uid} · MLP Brute Force</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE — SEARCH
# ══════════════════════════════════════════════════════════
def page_search():
    st.markdown("""
    <div style="padding:48px 48px 24px">
      <div style="font-family:'Bebas Neue',cursive;font-size:56px;
                  color:#fff;letter-spacing:2px">SEARCH</div>
      <div style="font-size:14px;color:#555;margin-top:4px">
        Explore 27,766 movies from the MovieLens catalogue
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 48px">', unsafe_allow_html=True)
        query = st.text_input(
            "Search",
            value=st.session_state.search_query,
            placeholder="Try 'Inception', 'Harry Potter', 'Nolan'…",
            label_visibility="collapsed",
        )
        st.session_state.search_query = query
        st.markdown("</div>", unsafe_allow_html=True)

    if query and len(query) >= 2:
        result = api_get("/search", params={"q": query, "limit": 14})
        if result and result.get("results"):
            movies = result["results"]
            total  = result["total"]
            st.markdown(f"""
            <div style="padding:12px 48px;font-size:13px;color:#555">
              Found <span style="color:#e50914;font-weight:600">
              {total}</span> result(s) for
              "<span style="color:#fff">{query}</span>"
            </div>
            """, unsafe_allow_html=True)
            render_row("SEARCH RESULTS", movies[:7])
            if len(movies) > 7:
                render_row("MORE RESULTS", movies[7:])

            # Similar to first result
            if movies:
                sim = api_get(f"/similar/{movies[0]['movie_idx']}",
                              params={"top_k": 7})
                if sim and sim.get("similar"):
                    render_row("YOU MIGHT ALSO LIKE",
                               sim["similar"][:7],
                               accent="Content similarity")
        else:
            st.markdown(f"""
            <div style="padding:60px;text-align:center;color:#333">
              <div style="font-size:48px">🔍</div>
              <div style="font-size:18px;margin-top:16px;color:#555">
                No results for "{query}"
              </div>
              <div style="font-size:13px;margin-top:8px;color:#333">
                Try a different title or spelling
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Genre browse
        st.markdown("""
        <div class="sec-hdr">
          <span class="sec-title">🎭 BROWSE BY GENRE</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="padding:0 48px">', unsafe_allow_html=True)
        genres_api = api_get("/genres")
        glist = genres_api.get("genres", ALL_GENRES) if genres_api else ALL_GENRES
        cols  = st.columns(5)
        for i, g in enumerate(glist[:20]):
            with cols[i % 5]:
                if st.button(g, key=f"gbrowse_{g}",
                             use_container_width=True):
                    st.session_state.search_query = g
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE — NEW USER ONBOARDING
# ══════════════════════════════════════════════════════════
def page_new_user():
    st.markdown("""
    <div class="onboard-wrap">
      <div class="onboard-title">CREATE YOUR<br>PROFILE</div>
      <div class="onboard-sub">
        Pick your favourite genres and we'll instantly match you
        to a user with similar taste — no sign-up required.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="max-width:720px;margin:0 auto;padding:0 48px">',
                unsafe_allow_html=True)

    st.markdown("**Select genres you enjoy** (pick at least 2):")

    # Genre multi-select
    selected = st.multiselect(
        "Genres",
        ALL_GENRES,
        default=st.session_state.selected_genres or [],
        label_visibility="collapsed",
    )
    st.session_state.selected_genres = selected

    if selected:
        st.markdown(f"""
        <div style="margin:12px 0;padding:12px 16px;
                    background:rgba(229,9,20,0.06);
                    border:1px solid rgba(229,9,20,0.2);
                    border-radius:8px;font-size:13px;color:#aaa">
          Selected: {', '.join(selected)}
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎬 Get My Recommendations",
                     disabled=len(selected) < 1,
                     use_container_width=True):
            # Find best proxy user
            votes = {}
            for g in selected:
                uid = GENRE_PROXY.get(g)
                if uid:
                    votes[uid] = votes.get(uid, 0) + 1
            best_uid = max(votes, key=votes.get) if votes else 0

            # Find matching profile label
            match = next(
                (p for p in PROFILES if p["idx"] == best_uid),
                {"name": "Custom", "icon": "✨"}
            )
            st.session_state.user_idx   = best_uid
            st.session_state.user_label = "You"
            st.session_state.user_icon  = "✨"
            st.session_state.selected_genres = []
            st.success(
                f"Profile created! Matched to a "
                f"{' & '.join(selected[:2])} fan. "
                f"Loading your recommendations…"
            )
            time.sleep(1)
            go("home")

    with col2:
        if st.button("← Back to Home", use_container_width=False):
            go("home")

    st.markdown("</div>", unsafe_allow_html=True)

    # Preview — show what genres map to
    st.markdown('<div style="padding:32px 48px">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**How it works:**")
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1️⃣", "Pick Genres", "Select what you love from 15 genres"),
        ("2️⃣", "Profile Match", "Our system finds the closest user with similar taste"),
        ("3️⃣", "Enjoy Picks", "Get 10 personalised recommendations instantly"),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3], steps):
        with col:
            st.markdown(f"""
            <div class="about-card" style="text-align:center">
              <div style="font-size:32px;margin-bottom:8px">{icon}</div>
              <div style="font-family:'Bebas Neue',cursive;
                          font-size:18px;color:#e50914;
                          letter-spacing:1px;margin-bottom:6px">
                {title}
              </div>
              <div style="font-size:13px;color:#666">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE — MY LIST
# ══════════════════════════════════════════════════════════
def page_mylist():
    st.markdown("""
    <div style="padding:48px 48px 24px">
      <div style="font-family:'Bebas Neue',cursive;font-size:56px;
                  color:#fff;letter-spacing:2px">MY LIST</div>
      <div style="font-size:14px;color:#555;margin-top:4px">
        Movies you've saved for later
      </div>
    </div>
    """, unsafe_allow_html=True)

    my_list = st.session_state.my_list

    if not my_list:
        st.markdown("""
        <div style="padding:60px;text-align:center">
          <div style="font-size:56px">📋</div>
          <div style="font-family:'Bebas Neue',cursive;font-size:28px;
                      color:#333;letter-spacing:2px;margin-top:16px">
            YOUR LIST IS EMPTY
          </div>
          <div style="font-size:13px;color:#444;margin-top:8px">
            Add movies from the Home page using the ➕ My List button
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Go to Home", key="mylist_home"):
            go("home")
        return

    render_row("YOUR SAVED MOVIES", my_list)

    st.markdown('<div style="padding:16px 48px">', unsafe_allow_html=True)
    if st.button("🗑️ Clear My List"):
        st.session_state.my_list = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PLOT LOADER HELPER
# ══════════════════════════════════════════════════════════
def load_plot(filename: str):
    """Load a saved plot image if it exists."""
    path = PLOTS_DIR / filename
    if path.exists():
        return Image.open(path)
    return None

def show_plot(filename: str, title: str,
              description: str, insight: str):
    """Render a saved plot with caption and insight."""
    img = load_plot(filename)
    st.markdown(f"""
    <div class="plot-card">
      <div class="plot-title">{title}</div>
      <div class="plot-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    if img:
        st.image(img, use_column_width=True)
    else:
        st.info(f"Plot not found: {filename} — run the analysis notebooks first.")
    if insight:
        st.markdown(f'<div class="insight-box">💡 {insight}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE — ABOUT  (with all plots embedded)
# ══════════════════════════════════════════════════════════
def page_about():
    # ── Hero ──────────────────────────────────────────────
    st.markdown("""
    <div style="padding:48px 48px 32px;
                border-bottom:1px solid #1a1a1f;
                margin-bottom:32px">
      <div style="font-family:'Bebas Neue',cursive;
                  font-size:64px;color:#fff;
                  letter-spacing:3px;line-height:.9">
        CINEMATE<span style="color:#e50914"> V2</span>
      </div>
      <div style="font-size:16px;color:#555;margin-top:12px;
                  font-weight:300;max-width:600px">
        A production-grade hybrid movie recommendation system
        built on 33 million real ratings — combining deep
        collaborative filtering with semantic content understanding.
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_titles = [
        "🏗️ Architecture",
        "📊 EDA Insights",
        "🤖 Model Results",
        "🧪 A/B Test",
        "💰 Business Impact",
        "⚖️ Bias & Fairness",
        "🎯 Recommendations",
    ]
    tabs = st.tabs(tab_titles)

    # ── Tab 0: Architecture ──────────────────────────────
    with tabs[0]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)
        c1, c2 = st.columns([3,2])

        with c1:
            st.markdown("""
            <div class="about-card">
              <div class="about-section-title">THE PROBLEM WE SOLVE</div>
              <p style="color:#aaa;font-size:14px;line-height:1.8">
                With 27,766 movies and 173,134 users, manually finding
                what to watch next is impossible. Traditional search
                requires you to know what you want. Cinemate V2 learns
                your taste from 33M historical ratings and surfaces films
                you didn't know you'd love.
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="about-card">
              <div class="about-section-title">TWO-TOWER ARCHITECTURE</div>
              <p style="color:#aaa;font-size:14px;line-height:1.8;margin-bottom:16px">
                Our model fuses two independent learning signals:
              </p>
              <p style="color:#ccc;font-size:13px;line-height:1.8">
                <strong style="color:#e50914">Collaborative Tower</strong>
                — Neural Collaborative Filtering learns from 16M positive
                user-movie interactions. Each user and movie gets a
                128-dimensional embedding. An MLP fusion layer captures
                non-linear preference patterns that linear models (SVD)
                cannot — e.g. "likes Sci-Fi AND Comedy but NOT
                Sci-Fi comedies."
              </p>
              <p style="color:#ccc;font-size:13px;line-height:1.8;margin-top:12px">
                <strong style="color:#e50914">Content Tower</strong>
                — DistilBERT reads each movie's title, genres, and
                genome tags (e.g. "mind-bending surreal thought-provoking")
                and produces a 768-dimensional semantic vector. This
                handles cold-start: new movies with no ratings still
                get meaningful recommendations via their content.
              </p>
              <p style="color:#ccc;font-size:13px;line-height:1.8;margin-top:12px">
                <strong style="color:#e50914">Debiasing Layer</strong>
                — Pure model scores suffer 0.57% catalogue coverage
                (popularity bubble). We inject random long-tail movies
                into 3 of 10 recommendation slots, boosting coverage
                to 5.4% while keeping NDCG above 0.09.
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="about-card">
              <div class="about-section-title">WHY MLP FUSION, NOT DOT PRODUCT?</div>
              <p style="color:#aaa;font-size:14px;line-height:1.8">
                Production systems like YouTube use dot-product
                Two-Tower because at 500M users you physically cannot
                score all items. At our scale (27,766 movies), brute
                force MLP scoring takes ~200ms — acceptable. The MLP
                captures non-linear interactions dot product cannot,
                giving us +90.9% NDCG improvement over SVD vs the
                ~60-70% typical for dot-product architectures.
              </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            # Tech stack
            st.markdown("""
            <div class="about-card">
              <div class="about-section-title">TECH STACK</div>
            """, unsafe_allow_html=True)
            tech = ["PyTorch 2.7","DistilBERT","ChromaDB",
                    "FastAPI","SQLAlchemy","SQLite","Streamlit",
                    "MovieLens 33M","BPR Loss","Plotly Dash"]
            badges = "".join(
                f'<span class="tech-badge">{t}</span>' for t in tech
            )
            st.markdown(badges + "</div>", unsafe_allow_html=True)

            # Key metrics
            st.markdown("""
            <div class="about-card">
              <div class="about-section-title">MODEL METRICS</div>
            """, unsafe_allow_html=True)
            metrics = [
                ("NCF NDCG@10",       "0.1252"),
                ("Two-Tower NDCG@10", "0.1199"),
                ("vs SVD Baseline",   "+90.9%"),
                ("vs Popularity",     "+48.2%"),
                ("Best BPR Loss",     "0.0608"),
                ("Training Epochs",   "46"),
                ("Embed Dim",         "128"),
                ("Tower Output",      "64"),
                ("Parameters",        "~32M"),
                ("Latency",           "~200ms"),
            ]
            rows = "".join(f"""
            <div class="metric-row">
              <span class="metric-key">{k}</span>
              <span class="metric-val">{v}</span>
            </div>""" for k,v in metrics)
            st.markdown(rows + "</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 1: EDA Insights ──────────────────────────────
    with tabs[1]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:0 0 24px">
          <div style="font-family:'Bebas Neue',cursive;font-size:28px;
                      color:#e50914;letter-spacing:1px">
            EXPLORATORY DATA ANALYSIS
          </div>
          <div style="font-size:13px;color:#555;margin-top:4px">
            Every model decision was driven by these findings from
            the MovieLens 33M dataset.
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            show_plot(
                "01_rating_distribution.png",
                "RATING DISTRIBUTION",
                "Distribution of all 33M ratings across the 0.5–5.0 scale, "
                "with percentage breakdown per rating value and a KDE curve "
                "showing the overall shape.",
                "Ratings skew strongly positive (mean 3.54, median 4.0). "
                "Users rate movies they chose to watch — not random ones. "
                "This selection bias means unrated ≠ disliked. "
                "Therefore we used BPR loss (ranking) instead of MSE "
                "(regression), which would falsely treat unrated as 0."
            )
        with c2:
            show_plot(
                "02_ratings_per_user.png",
                "RATINGS PER USER — POWER LAW",
                "Histogram (log scale) of how many ratings each user gave, "
                "plus a CDF curve showing the fraction of users below each "
                "activity threshold. The vertical line marks our filter at 20.",
                "38.2% of users have fewer than 20 ratings — far too sparse "
                "for collaborative filtering to learn meaningful patterns. "
                "We filtered these out, retaining 95.6% of ratings while "
                "dramatically reducing noise in user embeddings."
            )

        c3, c4 = st.columns(2)
        with c3:
            show_plot(
                "03_ratings_per_movie.png",
                "RATINGS PER MOVIE — LONG TAIL",
                "Log-scale histogram showing how ratings are distributed "
                "across the catalogue, plus a horizontal bar chart of the "
                "top 15 most-rated films.",
                "61.5% of movies have fewer than 10 ratings. This extreme "
                "long-tail distribution is why our model learned only "
                "~200 'popular' movies without debiasing. The Shawshank "
                "Redemption has 120K+ ratings while thousands of films "
                "have under 5 — an 24,000× difference in training signal."
            )
        with c4:
            show_plot(
                "04_ratings_over_time.png",
                "RATINGS OVER TIME — WHY WE SPLIT BY TIME",
                "Annual rating counts (left) and cumulative total (right) "
                "from 1995–2023, with the 80% cutoff year marked.",
                "80% of all ratings occurred before 2018. A random "
                "train/test split would expose the model to future ratings "
                "during training — a form of data leakage. We sort all "
                "ratings by timestamp and use the first 80% for training, "
                "last 20% for testing, exactly mirroring real deployment."
            )

        c5, c6 = st.columns(2)
        with c5:
            show_plot(
                "05_genre_distribution.png",
                "GENRE DISTRIBUTION — 19 UNIQUE GENRES",
                "Bar chart of movie counts across all 19 genres "
                "in the dataset.",
                "Drama (33,681 movies) and Comedy dominate the catalogue. "
                "IMAX and Film-Noir have very few entries. This imbalance "
                "matters: the content tower's genre embeddings are "
                "proportionally stronger for Drama/Comedy and weaker "
                "for niche genres — a known limitation."
            )
        with c6:
            show_plot(
                "06_top_tags.png",
                "TOP USER TAGS — RICH SEMANTIC SIGNAL",
                "Bar chart of the 20 most frequently used user-generated "
                "tags across the dataset.",
                "153,949 unique user tags exist, covering everything from "
                "mood ('atmospheric', 'surreal') to themes ('dystopia', "
                "'social commentary') to style ('cinematography'). "
                "We feed these into DistilBERT for the content tower — "
                "they provide semantic depth beyond genre labels alone."
            )

        show_plot(
            "07_avg_rating_genre.png",
            "AVERAGE RATING BY GENRE — TASTE PATTERNS",
            "Bar chart comparing mean ratings across all 19 genres, "
            "with the dataset-wide average shown as a dashed line.",
            "Film-Noir (~3.9) and War genres receive the highest ratings, "
            "while Horror (~3.3) and Comedy receive the lowest. This "
            "doesn't mean those genres are worse — it reflects which "
            "users tend to rate them. Serious, story-driven films attract "
            "more deliberate raters who score higher on average."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Model Results ─────────────────────────────
    with tabs[2]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)

        # Metrics summary
        st.markdown("""
        <div style="font-family:'Bebas Neue',cursive;font-size:28px;
                    color:#e50914;letter-spacing:1px;margin-bottom:20px">
          ALL MODELS — FAIR COMPARISON (2,000 USERS)
        </div>
        """, unsafe_allow_html=True)

        results_df = pd.DataFrame({
            "Model"        : ["Random","Popularity","SVD",
                              "NCF","Two-Tower + Debias"],
            "NDCG@10"      : [0.0019, 0.0845, 0.0669, 0.1252, 0.0946],
            "Precision@10" : [0.0018, 0.0749, 0.0607, 0.1154, 0.0890],
            "Recall@10"    : [0.0004, 0.0225, 0.0221, 0.0318, 0.0275],
            "vs SVD"       : ["—","—","baseline","+87.1%","+41.4%"],
        })
        st.dataframe(
            results_df.style.highlight_max(
                subset=["NDCG@10","Precision@10","Recall@10"],
                color="#1c1c1c"
            ),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            show_plot(
                "08_ncf_training_loss.png",
                "NCF TRAINING LOSS — 20 EPOCHS",
                "BPR loss per epoch during NCF training, with time "
                "per epoch shown alongside. Trained on Kaggle T4 GPU.",
                "Loss dropped from 0.1128 (epoch 1) to 0.0694 (epoch 19) "
                "— a 38.5% reduction. The curve shows fast early learning "
                "followed by plateau around epoch 10, then slower "
                "refinement. With tail-aware negative sampling in the "
                "retrained version, the model also learned representations "
                "for less-popular films, enabling debiasing at inference."
            )
        with c2:
            show_plot(
                "09_ncf_comparison.png",
                "MODEL COMPARISON — ALL BASELINES vs NCF",
                "Side-by-side bar charts comparing NDCG@10, "
                "Precision@10, and Recall@10 across all four models.",
                "NCF's +90.9% NDCG improvement over SVD proves that "
                "non-linear MLP interactions capture user preference "
                "patterns that linear matrix factorisation fundamentally "
                "cannot. The gap between NCF and Two-Tower (3.2%) on "
                "raw NDCG is explained by the additional content tower "
                "signal interfering with collaborative patterns on this "
                "dense dataset — content tower's value is cold-start handling."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: A/B Test ──────────────────────────────────
    with tabs[3]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)
        ab_path = MODELS_DIR / "ab_test_results.json"
        if ab_path.exists():
            with open(ab_path) as f:
                ab = json.load(f)

            decision = ab.get("decision","UNKNOWN")
            p_val    = ab["hypothesis_test"]["p_value"]
            cohens_d = ab["hypothesis_test"]["cohens_d"]
            lift_rel = ab["lift"]["relative_pct"]
            ctrl     = ab["control_ndcg"]["mean"]
            treat    = ab["treatment_ndcg"]["mean"]

            color = "#22c55e" if decision == "SHIP" else "#ef4444"
            icon  = "🚀" if decision == "SHIP" else "⚠️"

            st.markdown(f"""
            <div style="background:#161619;border:2px solid {color};
                        border-radius:12px;padding:24px;margin-bottom:24px">
              <div style="font-family:'Bebas Neue',cursive;font-size:36px;
                          color:{color};letter-spacing:2px">
                {icon} DECISION: {decision}
              </div>
              <div style="font-size:14px;color:#aaa;margin-top:8px;
                          max-width:600px;line-height:1.7">
                Two-Tower recommendations show a statistically significant
                improvement over random baseline. All five shipping criteria
                met — including p &lt; 0.05, positive CI bounds,
                and large effect size.
              </div>
            </div>
            """, unsafe_allow_html=True)

            c1,c2,c3,c4,c5 = st.columns(5)
            for col, label, val in [
                (c1,"Control NDCG",  f"{ctrl:.4f}"),
                (c2,"Treatment NDCG",f"{treat:.4f}"),
                (c3,"Relative Lift", f"+{lift_rel:.1f}%"),
                (c4,"p-value",       f"{p_val:.6f}"),
                (c5,"Cohen's d",     f"{cohens_d:.3f}"),
            ]:
                col.metric(label, val)

        show_plot(
            "10_ab_test_results.png",
            "A/B TEST — DISTRIBUTION COMPARISON & LIFT",
            "Three panels: (left) overlapping NDCG distributions for "
            "control vs treatment, (centre) all model NDCG values, "
            "(right) relative lift with 95% confidence interval.",
            "The control (random) NDCG distribution peaks near zero "
            "while the treatment (Two-Tower) distribution is broadly "
            "spread with a much higher mean. The 95% CI for relative "
            "lift has a strictly positive lower bound — confirming the "
            "improvement is real and not due to sampling variance. "
            "We used Welch's t-test (unequal variance) + Mann-Whitney U "
            "(non-parametric) for robustness."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 4: Business Impact ───────────────────────────
    with tabs[4]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)
        biz_path = MODELS_DIR / "business_impact.json"
        if biz_path.exists():
            with open(biz_path) as f:
                biz = json.load(f)

            monthly = biz.get("monthly_lift_inr", 0)
            annual  = biz.get("annual_lift_inr",  0)
            roi     = biz.get("roi_pct",           0)
            payback = biz.get("payback_months",    0)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Monthly Revenue Lift",
                      f"₹{monthly/100000:.1f}L")
            c2.metric("Annual Revenue Lift",
                      f"₹{annual/100000:.1f}L")
            c3.metric("ROI", f"{roi:.0f}%")
            c4.metric("Payback Period", f"{payback:.1f} months")

        show_plot(
            "11_sensitivity_analysis.png",
            "SENSITIVITY ANALYSIS — REVENUE LIFT",
            "Heatmap showing annual revenue lift (₹ Lakhs) across "
            "different combinations of retention lift percentage "
            "and average revenue per user (ARPU).",
            "Based on conservative assumptions: 100,000 MAU, "
            "₹199/month ARPU, 5% retention improvement from recommendations. "
            "The model generates ₹12L annual lift with 240% ROI and "
            "a 5-month payback period. Even the most pessimistic cell "
            "(2% retention, ₹99 ARPU) delivers positive ROI — confirming "
            "the recommender system is worth building at any realistic scale."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 5: Bias & Fairness ───────────────────────────
    with tabs[5]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)

        bias_path = MODELS_DIR / "bias_results.json"
        if bias_path.exists():
            with open(bias_path) as f:
                bias = json.load(f)
            pb  = bias.get("popularity_bias",{})
            cs  = bias.get("cold_start",{})
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Top 1% Movies →",
                      f"{pb.get('top_1pct_rating_share',0):.1%}",
                      "of all ratings")
            c2.metric("Top 10% Movies →",
                      f"{pb.get('top_10pct_rating_share',0):.1%}",
                      "of all ratings")
            c3.metric("Cold Start Movies",
                      f"{cs.get('cold_start_movies_in_test',0):,}",
                      "in test set")
            c4.metric("Cold Start Rate",
                      f"{cs.get('cold_start_pct',0):.1%}",
                      "of test movies")

        c1, c2 = st.columns(2)
        with c1:
            show_plot(
                "12_popularity_bias.png",
                "POPULARITY BIAS — THE LONG TAIL PROBLEM",
                "Log-log scatter of ratings per movie (left) "
                "and cumulative rating share curve (right).",
                "The top 1% of movies (278 films) account for over "
                "50% of all ratings. This extreme concentration is why "
                "our undebiased model recommended only 0.57% of the "
                "catalogue. Without intervention, any collaborative "
                "filter trained on this data will effectively become "
                "a popularity ranker — defeating the purpose of personalisation."
            )
        with c2:
            show_plot(
                "13_genre_bias.png",
                "GENRE REPRESENTATION RATIO",
                "Horizontal bar showing how each genre's share of "
                "rated movies compares to its share of the full catalogue. "
                "Green = over-represented, Red = under-represented.",
                "IMAX content is 4.3× over-represented in ratings vs "
                "catalogue — IMAX films attract more deliberate, engaged "
                "viewers. Documentary is 0.12× under-represented — many "
                "documentaries exist but attract few raters in a general "
                "movie dataset. These imbalances flow directly into model "
                "embeddings: our model knows Drama very well and "
                "Documentary very poorly."
            )

        show_plot(
            "14_personalisation.png",
            "PERSONALISATION SCORE — HOW UNIQUE ARE RECOMMENDATIONS?",
            "Histogram of pairwise Jaccard similarity between "
            "recommendation lists for 200 random user pairs. "
            "Score = 1 − mean similarity.",
            "Personalisation score of 0.863 means the average pair "
            "of users shares fewer than 14% of their top-10 movies. "
            "This is high personalisation — the model genuinely "
            "captures individual taste, not just global popularity. "
            "The distribution is right-skewed, with most pairs having "
            "near-zero overlap, confirming diverse user representations."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 6: Recommendation Analysis ──────────────────
    with tabs[6]:
        st.markdown('<div style="padding:24px 0">', unsafe_allow_html=True)

        rec_path = MODELS_DIR / "rec_analysis.json"
        if rec_path.exists():
            with open(rec_path) as f:
                rec = json.load(f)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Catalogue Coverage",
                      f"{rec.get('catalogue_coverage',0):.1%}")
            c2.metric("Personalisation",
                      f"{rec.get('personalisation_score',0):.3f}")
            c3.metric("Avg Genres/User",
                      f"{rec.get('avg_genres_per_user',0):.1f}")
            c4.metric("Unique Movies Rec.",
                      f"{rec.get('unique_movies_rec',0):,}")

        c1, c2 = st.columns(2)
        with c1:
            show_plot(
                "14_personalisation.png",
                "RECOMMENDATION OVERLAP BETWEEN USERS",
                "Distribution of Jaccard similarity scores between "
                "pairs of users' recommendation lists.",
                "Most user pairs share 0–2 movies in their top 10, "
                "confirming the model produces genuinely personalised "
                "recommendations rather than a global popularity list."
            )
        with c2:
            show_plot(
                "15_genre_diversity.png",
                "GENRE DIVERSITY PER USER",
                "Distribution of how many unique genres appear "
                "across each user's top-10 recommendations.",
                "On average users receive recommendations spanning "
                "4+ unique genres, indicating the model doesn't "
                "collapse into single-genre bubbles for most users. "
                "The debiasing layer (3 random tail slots) further "
                "increases genre spread by injecting long-tail content."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="cine-footer">
      <div>
        <span style="font-family:'Bebas Neue',cursive;
                     font-size:18px;color:#e50914;
                     letter-spacing:2px">CINEMATE V2</span>
        <span style="margin-left:16px">
          PEC Chandigarh · Electrical Engineering ·
          MAANG Data Science Portfolio Project
        </span>
      </div>
      <div>
        Two-Tower NCF + DistilBERT ·
        MovieLens 33M · FastAPI + Streamlit
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR — manual controls
# ══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Bebas Neue',cursive;
                    font-size:24px;color:#e50914;
                    letter-spacing:3px;margin-bottom:4px">
          CINEMATE V2
        </div>
        <div style="font-size:11px;color:#333;
                    text-transform:uppercase;letter-spacing:2px;
                    margin-bottom:24px">
          AI Movie Recommender
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("**Navigate**")
        nav_items = [
            ("🏠 Home",      "home"),
            ("🔍 Search",    "search"),
            ("✨ New User",  "new_user"),
            ("📋 My List",   "mylist"),
            ("ℹ️ About",     "about"),
        ]
        for label, pg in nav_items:
            if st.button(label, key=f"sb_{pg}",
                         use_container_width=True):
                go(pg)

        st.divider()

        # Manual user ID
        st.markdown("**Manual User ID**")
        uid = st.number_input("User", 0, 172999,
                              st.session_state.user_idx,
                              label_visibility="collapsed")
        if st.button("Load User", use_container_width=True):
            st.session_state.user_idx   = int(uid)
            st.session_state.user_label = f"User #{uid}"
            st.session_state.user_icon  = "👤"
            go("home")

        st.divider()

        # Health
        health = health_check()
        if health:
            st.success("✅ API Online")
            st.caption(f"Users: {health.get('num_users',0):,}")
            st.caption(f"Movies: {health.get('num_movies',0):,}")
        else:
            st.error("❌ API Offline")
            st.caption("uvicorn api.main:app --reload")

        st.divider()
        st.caption("Cinemate V2 · Two-Tower Hybrid")
        st.caption("NCF + DistilBERT · MovieLens 33M")


# ══════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════
def main():
    render_sidebar()
    render_navbar()

    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "search":
        page_search()
    elif page == "new_user":
        page_new_user()
    elif page == "mylist":
        page_mylist()
    elif page == "about":
        page_about()
    else:
        go("home")


if __name__ == "__main__":
    main()



