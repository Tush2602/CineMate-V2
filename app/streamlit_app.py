import os
import sys
import json
import time
import asyncio
import aiohttp
import requests
import streamlit as st
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

API_URL        = os.getenv("API_URL", "http://localhost:8000")
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
TMDB_IMG       = "https://image.tmdb.org/t/p/w500"
TMDB_BACK      = "https://image.tmdb.org/t/p/w1280"
NEW_USERS_FILE = Path(__file__).parent / "new_users.json"
NUM_USERS_BASE = 173134

st.set_page_config(
    page_title  = "Cinemate",
    page_icon   = "🎬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Space+Grotesk:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0a0a;
    --bg2:      #111111;
    --bg3:      #1a1a1a;
    --border:   #1f1f1f;
    --border2:  #2a2a2a;
    --text:     #e2e2e2;
    --muted:    #555;
    --muted2:   #333;
    --gold:     #c9a84c;
    --gold2:    #e8c97a;
    --red:      #c0392b;
    --green:    #27ae60;
}

* { box-sizing:border-box; margin:0; padding:0; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

#MainMenu, footer, header { visibility:hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
    width: 240px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.25rem !important;
}
[data-testid="stSidebarNav"] { display:none; }

/* ── Main content ── */
.block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar logo ── */
.sb-logo {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 900;
    color: var(--gold);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.sb-tagline {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}

/* ── Nav buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    text-align: left !important;
    transition: all .15s ease !important;
    letter-spacing: .5px !important;
}
.stButton > button:hover {
    background: var(--bg3) !important;
    color: var(--text) !important;
    transform: none !important;
}
.stButton > button[kind="primary"] {
    background: var(--bg3) !important;
    color: var(--gold) !important;
    border-left: 2px solid var(--gold) !important;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    letter-spacing: -1px;
}
.page-subtitle {
    font-size: 13px;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: .5px;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 0;
    margin-bottom: 32px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg2);
}
.stat-cell {
    flex: 1;
    padding: 16px 20px;
    border-right: 1px solid var(--border);
    text-align: center;
}
.stat-cell:last-child { border-right: none; }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.stat-lbl {
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin: 36px 0 16px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -.3px;
}
.section-tag {
    font-size: 10px;
    color: var(--gold);
    background: rgba(201,168,76,.1);
    border: 1px solid rgba(201,168,76,.2);
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Movie card ── */
.card-meta {
    margin-top: 8px;
}
.card-title-text {
    font-size: 12px;
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}
.card-genre-text {
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: .3px;
}
.card-badges {
    display: flex;
    gap: 4px;
    margin-top: 4px;
    flex-wrap: wrap;
}
.badge-rank {
    background: var(--gold);
    color: #000;
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: .5px;
}
.badge-score {
    background: var(--bg3);
    color: var(--muted);
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid var(--border2);
}
.badge-new {
    background: rgba(39,174,96,.15);
    color: var(--green);
    font-size: 9px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid rgba(39,174,96,.3);
    letter-spacing: .5px;
}

/* ── No poster ── */
.no-poster-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    aspect-ratio: 2/3;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 16px;
    text-align: center;
}
.no-poster-icon { font-size: 28px; margin-bottom: 8px; opacity: .4; }
.no-poster-title { font-size: 11px; color: var(--muted); line-height: 1.4; }

/* ── User cards ── */
.who-watching {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.u-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    transition: border-color .2s;
}
.u-card.active {
    border-color: var(--gold);
    background: rgba(201,168,76,.05);
}
.u-avatar { font-size: 28px; margin-bottom: 6px; }
.u-name { font-size: 11px; color: var(--text); font-weight: 600; }
.u-desc { font-size: 9px; color: var(--muted); margin-top: 2px; letter-spacing: .3px; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0f0c1a 0%, #0a0a0a 60%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 48px 48px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(201,168,76,.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 10px;
    color: var(--gold);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: var(--gold);
}
.hero-name {
    font-family: 'Playfair Display', serif;
    font-size: 56px;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    letter-spacing: -2px;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 480px;
}

/* ── Inputs ── */
.stTextInput input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,.15) !important;
}
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
}
.stSlider > div > div > div { background: var(--gold) !important; }

/* ── Sidebar section labels ── */
.sb-section {
    font-size: 9px;
    color: var(--muted2);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 20px 0 8px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── stImage rounded ── */
.stImage img { border-radius: 6px; }

/* ── Number input ── */
.stNumberInput input {
    background: var(--bg2) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
}

/* ── Alert overrides ── */
.stSuccess { background: rgba(39,174,96,.1) !important; border-color: rgba(39,174,96,.3) !important; }
.stError   { background: rgba(192,57,43,.1) !important; border-color: rgba(192,57,43,.3) !important; }
.stWarning { background: rgba(201,168,76,.1) !important; border-color: rgba(201,168,76,.3) !important; }

/* ── Metric ── */
div[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
div[data-testid="stMetricValue"] { color: var(--gold) !important; }
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

_defaults = {
    'user_idx'       : 320,
    'user_name'      : 'Alex',
    'is_new_user'    : False,
    'page'           : 'home',
    'search_query'   : '',
    'my_list'        : [],
    'alpha'          : 0.3,
    'lam'            : 0.3,
    'new_user_genres': [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# USER STORAGE
def _load_new_users() -> dict:
    if NEW_USERS_FILE.exists():
        try:
            return json.loads(NEW_USERS_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_new_users(users: dict):
    NEW_USERS_FILE.write_text(json.dumps(users, indent=2))

def create_new_user(name, genres, avatar) -> int:
    users  = _load_new_users()
    new_id = NUM_USERS_BASE + len(users)
    users[str(new_id)] = {
        'name': name, 'avatar': avatar, 'genres': genres,
        'created': datetime.now().isoformat(), 'user_idx': new_id,
    }
    _save_new_users(users)
    return new_id

def get_new_user(user_idx):
    return _load_new_users().get(str(user_idx))

def list_new_users():
    return list(_load_new_users().values())


# API
@st.cache_data(ttl=300)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_health():
    return api_get("/health")

@st.cache_data(ttl=3600)
def get_poster(tmdb_id: int):
    if not TMDB_API_KEY or not tmdb_id:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY}, timeout=5
        )
        if r.status_code != 200:
            return None
        d = r.json()
        return {
            "poster"  : f"{TMDB_IMG}{d['poster_path']}"   if d.get("poster_path")   else None,
            "backdrop": f"{TMDB_BACK}{d['backdrop_path']}" if d.get("backdrop_path") else None,
            "overview": d.get("overview", ""),
            "year"    : d.get("release_date", "")[:4],
            "rating"  : d.get("vote_average", 0),
        }
    except Exception:
        return None

async def _fetch_one(session, tmdb_id):
    if not TMDB_API_KEY or not tmdb_id:
        return tmdb_id, None
    try:
        async with session.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=aiohttp.ClientTimeout(total=5)
        ) as r:
            if r.status != 200:
                return tmdb_id, None
            d = await r.json()
            return tmdb_id, {
                "poster"  : f"{TMDB_IMG}{d['poster_path']}"   if d.get("poster_path")   else None,
                "backdrop": f"{TMDB_BACK}{d['backdrop_path']}" if d.get("backdrop_path") else None,
                "overview": d.get("overview", ""),
                "year"    : d.get("release_date", "")[:4],
                "rating"  : d.get("vote_average", 0),
            }
    except Exception:
        return tmdb_id, None

@st.cache_data(ttl=3600)
def fetch_posters_batch(tmdb_ids: tuple) -> dict:
    async def _all():
        async with aiohttp.ClientSession() as s:
            return dict(await asyncio.gather(*[_fetch_one(s, t) for t in tmdb_ids]))
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_all())
    except Exception:
        return {}


# STATIC

PRESET_PROFILES = [
    {"name": "Alex",   "icon": "🎬", "idx": 320,  "desc": "Action fan"},
    {"name": "Sam",    "icon": "🎭", "idx": 370,  "desc": "Drama lover"},
    {"name": "Jordan", "icon": "🚀", "idx": 280,  "desc": "Sci-Fi geek"},
    {"name": "Morgan", "icon": "😂", "idx": 430,  "desc": "Comedy first"},
    {"name": "Casey",  "icon": "👻", "idx": 460,  "desc": "Horror fan"},
]

GENRE_LIST = [
    "Action","Adventure","Animation","Comedy","Crime",
    "Documentary","Drama","Fantasy","Horror","Mystery",
    "Romance","Sci-Fi","Thriller","War","Western",
]

AVATARS = ["🎬","🎭","🚀","😂","👻","🌙","⚡","🔥","🎵","🌊"]



# COMPONENTS

def render_movie_card(movie, rank=None, show_score=True, is_new=False, poster_data=None):
    tmdb_id = movie.get('tmdb_id') or 0
    title   = str(movie.get('title', 'Unknown'))
    genres  = str(movie.get('genres', ''))
    score   = movie.get('score') or 0

    if poster_data is None:
        poster_data = get_poster(int(tmdb_id)) if tmdb_id else None
    poster_url = poster_data['poster'] if poster_data else None

    safe_title   = title[:32]
    short_genres = " · ".join(genres.split()[:2]) if genres else ""

    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        no_photo_path = Path(__file__).parent / "no_photo.png"
        if no_photo_path.exists():
            st.image(str(no_photo_path), use_container_width=True)
        else:
            st.markdown(f"""
            <div class="no-poster-box">
                <div class="no-poster-icon">🎞</div>
                <div class="no-poster-title">{safe_title}</div>
            </div>""", unsafe_allow_html=True)

    badges = ""
    if rank:
        badges += f'<span class="badge-rank">#{rank:02d}</span>'
    if is_new:
        badges += '<span class="badge-new">NEW</span>'
    if show_score and score:
        badges += f'<span class="badge-score">{score:.2f}</span>'

    st.markdown(f"""
    <div class="card-meta">
        <div class="card-title-text">{safe_title}</div>
        <div class="card-genre-text">{short_genres}</div>
        <div class="card-badges">{badges}</div>
    </div>""", unsafe_allow_html=True)


def render_section(title, tag=""):
    tag_html = f'<span class="section-tag">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">{title}</span>
        {tag_html}
    </div>""", unsafe_allow_html=True)


def render_movie_row(title, movies, show_rank=False, tag="", mark_new=None):
    if not movies:
        return
    mark_new = mark_new or set()
    render_section(title, tag)

    n    = min(len(movies), 7)
    tids = tuple(int(m.get('tmdb_id') or 0) for m in movies[:n])
    pcache = fetch_posters_batch(tids)

    cols = st.columns(n, gap="small")
    for i, (col, movie) in enumerate(zip(cols, movies[:n])):
        with col:
            render_movie_card(
                movie,
                rank        = i+1 if show_rank else None,
                is_new      = movie.get('movie_idx') in mark_new,
                poster_data = pcache.get(int(movie.get('tmdb_id') or 0)),
            )


def render_stats(health, stats):
    num_users  = health.get('num_users',  0) if health else 0
    num_movies = health.get('num_movies', 0) if health else 0
    total_recs = 25_868_311
    coverage   = 0.1792

    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-cell">
            <div class="stat-val">{num_users:,}</div>
            <div class="stat-lbl">Users</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">{num_movies:,}</div>
            <div class="stat-lbl">Movies</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">0.1203</div>
            <div class="stat-lbl">NDCG@10</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">+79.9%</div>
            <div class="stat-lbl">vs SVD</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">17.9%</div>
            <div class="stat-lbl">Coverage</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">25M+</div>
            <div class="stat-lbl">Ratings Trained</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">4.6hrs</div>
            <div class="stat-lbl">Train Time</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_who_watching():
    st.markdown('<div class="who-watching">Who\'s Watching</div>', unsafe_allow_html=True)

    new_users    = list_new_users()[-3:]
    all_profiles = PRESET_PROFILES + [
        {"name": u['name'], "icon": u['avatar'], "idx": u['user_idx'], "desc": "Custom"}
        for u in new_users
    ]

    cols = st.columns(len(all_profiles) + 1, gap="small")
    for col, p in zip(cols, all_profiles):
        with col:
            active  = st.session_state.user_idx == p['idx']
            cls     = "u-card active" if active else "u-card"
            st.markdown(f"""
            <div class="{cls}">
                <div class="u-avatar">{p['icon']}</div>
                <div class="u-name">{p['name']}</div>
                <div class="u-desc">{p['desc']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("▶", key=f"p_{p['idx']}", use_container_width=True):
                st.session_state.user_idx  = p['idx']
                st.session_state.user_name = p['name']
                st.session_state.is_new_user = p['idx'] >= NUM_USERS_BASE
                st.session_state.page = 'home'
                st.rerun()

    with cols[-1]:
        st.markdown("""
        <div class="u-card" style="border-style:dashed">
            <div class="u-avatar">＋</div>
            <div class="u-name">New</div>
            <div class="u-desc">Create profile</div>
        </div>""", unsafe_allow_html=True)
        if st.button("▶", key="p_new", use_container_width=True):
            st.session_state.page = 'new_user'
            st.rerun()


def get_cold_start_recs(genres, top_k=14):
    result = api_post("/recommend/cold-start", {"genres": genres, "top_k": top_k})
    if result and result.get("recommendations"):
        return result["recommendations"]

    seen, mixed = set(), []
    for genre in genres:
        res = api_get("/search", params={"q": genre, "limit": top_k})
        if not res:
            continue
        for m in res.get("results", []):
            mid = m.get("movie_idx")
            if mid not in seen:
                seen.add(mid)
                mixed.append(m)
            if len(mixed) >= top_k:
                break
        if len(mixed) >= top_k:
            break

    if not mixed:
        pop = api_get("/popular", params={"limit": top_k})
        if pop and pop.get("movies"):
            mixed = [{"movie_idx": m["movie_idx"], "title": m.get("title",""),
                      "genres": m.get("genres",""), "tmdb_id": m.get("tmdb_id",0),
                      "score": 0, "rank": i+1}
                     for i, m in enumerate(pop["movies"][:top_k])]
    return mixed[:top_k]


def page_home():
    user_idx  = st.session_state.user_idx
    is_new    = st.session_state.is_new_user
    user_name = st.session_state.user_name
    alpha     = st.session_state.alpha
    lam       = st.session_state.lam
    recs      = []
    all_gems  = []

    health = api_health()
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
    except Exception:
        stats = None

    # Hero
    eyebrow = "Cold Start · Genre Mode" if is_new else "Personalised Recommendations"
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">{eyebrow}</div>
        <div class="hero-name">{user_name if user_name else "Cinemate"}</div>
        <div class="hero-sub">
            {"Recommendations built from your genre preferences. Rate movies to enable full personalisation."
              if is_new else
             "Two-Tower hybrid model · Neural CF + DistilBERT content embeddings · User #" + str(user_idx)}
        </div>
    </div>""", unsafe_allow_html=True)

    render_stats(health, stats)
    render_who_watching()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Recommendations
    if is_new:
        user_data = get_new_user(user_idx)
        genres    = user_data['genres'] if user_data else st.session_state.new_user_genres

        if not genres:
            st.markdown("""
            <div style="padding:64px;text-align:center;color:var(--muted)">
                <div style="font-size:40px;margin-bottom:16px">🎭</div>
                <div style="font-size:16px;font-family:'Playfair Display',serif">No genres set</div>
                <div style="font-size:13px;margin-top:8px">Create a profile to see recommendations.</div>
            </div>""", unsafe_allow_html=True)
            return

        with st.spinner("Building recommendations…"):
            recs = get_cold_start_recs(genres, top_k=14)

        render_movie_row(
            f"Picks for {' · '.join(genres[:2])}",
            recs[:7], show_rank=True,
            tag=f"Updated {datetime.now().strftime('%H:%M')}"
        )
        render_movie_row("More You Might Like", recs[7:])

    else:
        with st.spinner("Loading recommendations…"):
            try:
                result = requests.get(
                    f"{API_URL}/recommend/{user_idx}",
                    params={"top_k": 14, "alpha": alpha, "lambda": lam},
                    timeout=60
                ).json()
            except Exception:
                result = None

        if result is None:
            st.markdown("""
            <div style="padding:64px;text-align:center;color:var(--muted)">
                <div style="font-size:40px;margin-bottom:16px">🔌</div>
                <div style="font-size:16px;font-family:'Playfair Display',serif">API Offline</div>
                <div style="font-size:13px;margin-top:8px;font-family:monospace">
                    uvicorn api.app:app --reload
                </div>
            </div>""", unsafe_allow_html=True)
            return

        recs = result.get('recommendations', [])
        rt   = result.get('response_time_ms', 0)

        # Hidden Gems — dedicated tail endpoint (no cache)
        try:
            gems_r   = requests.get(
                f"{API_URL}/recommend/gems/{user_idx}",
                params={"top_k": 7}, timeout=60
            )
            all_gems = gems_r.json().get('gems', []) if gems_r.status_code == 200 else []
        except Exception:
            all_gems = []

        if all_gems:
            render_movie_row(
                "Beyond the Obvious", all_gems,
                tag="Experimental · Tail Debiased",
            )

        # Top Picks SECOND
        render_movie_row(
            "Top Picks", recs[:7],
            show_rank=True,
            tag=f"{rt:.0f}ms"
        )

        # Because You Watched — Hidden Gems ke #1 se
        source_movie = all_gems[0]['movie_idx'] if all_gems else recs[0]['movie_idx']
        if recs:
            similar = api_get(f"/similar/{source_movie}", params={"top_k": 7})
            if similar and similar.get('similar'):
                source = similar['source_movie']['title'][:28]
                render_movie_row("Because You Watched", similar['similar'][:7], tag=source)

    # Rating
    st.markdown("<hr>", unsafe_allow_html=True)
    render_section("Rate a Recommendation")
    if recs:
        c1, c2, c3 = st.columns([3, 2, 1])
        opts = {f"#{i+1} — {str(r.get('title',''))[:40]}": r['movie_idx']
                for i, r in enumerate(all_gems if all_gems else recs)}
        with c1:
            sel = st.selectbox("Movie", list(opts.keys()), label_visibility="collapsed")
        with c2:
            rat = st.slider("Rating", 0.5, 5.0, 4.0, 0.5, label_visibility="collapsed")
        with c3:
            if st.button("Submit", use_container_width=True):
                resp = api_post("/feedback", {
                    "user_idx": int(user_idx),
                    "movie_idx": opts[sel],
                    "rating": rat, "from_rec": True
                })
                st.success("Saved!" if resp else "Saved locally.")


def page_search():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Search</div>
        <div class="page-subtitle">27,766 titles · type to find anything</div>
    </div>""", unsafe_allow_html=True)

    query = st.text_input("Search", placeholder="e.g. Inception, Parasite, 2001…",
                          label_visibility="collapsed",
                          value=st.session_state.search_query)

    if query and len(query) >= 2:
        with st.spinner("Searching…"):
            result = api_get("/search", params={"q": query, "limit": 14})

        if result and result.get('results'):
            movies = result['results']
            total  = result['total']
            st.markdown(f"""
            <div style="font-size:12px;color:var(--muted);margin-bottom:8px">
                {total} results for
                <span style="color:var(--gold)">"{query}"</span>
            </div>""", unsafe_allow_html=True)

            render_movie_row("Results", movies[:7])
            if len(movies) > 7:
                render_movie_row("More Results", movies[7:])

            if movies:
                sim = api_get(f"/similar/{movies[0]['movie_idx']}", params={"top_k":7})
                if sim and sim.get('similar'):
                    render_movie_row("You Might Also Like", sim['similar'][:7],
                                     tag="Content Similarity")
        else:
            st.markdown(f"""
            <div style="padding:64px;text-align:center;color:var(--muted)">
                <div style="font-size:40px;margin-bottom:16px">🔍</div>
                <div style="font-family:'Playfair Display',serif;font-size:16px">
                    Nothing found for "{query}"
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        render_section("Browse by Genre")
        gdata  = api_get("/genres")
        glist  = gdata.get('genres', GENRE_LIST) if gdata else GENRE_LIST
        cols   = st.columns(5, gap="small")
        for i, g in enumerate(glist[:20]):
            with cols[i % 5]:
                if st.button(g, key=f"g_{g}", use_container_width=True):
                    st.session_state.search_query = g
                    st.rerun()


def page_new_user():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Create Profile</div>
        <div class="page-subtitle">Tell us your taste — we'll handle the rest</div>
    </div>""", unsafe_allow_html=True)

    c_form, c_prev = st.columns([2, 1], gap="large")

    with c_form:
        st.markdown('<div class="sb-section">Your Name</div>', unsafe_allow_html=True)
        name = st.text_input("Name", placeholder="e.g. Jamie",
                             label_visibility="collapsed", max_chars=30)

        st.markdown('<div class="sb-section" style="margin-top:20px">Avatar</div>',
                    unsafe_allow_html=True)
        av_cols = st.columns(len(AVATARS), gap="small")
        chosen  = st.session_state.get('chosen_avatar', AVATARS[0])
        for col, av in zip(av_cols, AVATARS):
            with col:
                border = "border:1px solid var(--gold);" if av == chosen else "border:1px solid var(--border);"
                st.markdown(f"""
                <div style="text-align:center;font-size:22px;background:var(--bg2);
                            border-radius:6px;padding:6px;{border}">{av}</div>""",
                            unsafe_allow_html=True)
                if st.button(av, key=f"av_{av}", use_container_width=True):
                    st.session_state.chosen_avatar = av
                    st.rerun()

        st.markdown('<div class="sb-section" style="margin-top:20px">Favourite Genres (1–5)</div>',
                    unsafe_allow_html=True)
        sel_genres = st.multiselect("Genres", GENRE_LIST,
                                    default=["Action","Drama"],
                                    max_selections=5,
                                    label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("Create Profile", use_container_width=True):
                if not name.strip():
                    st.warning("Enter a name.")
                elif not sel_genres:
                    st.warning("Pick at least one genre.")
                else:
                    av    = st.session_state.get('chosen_avatar', AVATARS[0])
                    nid   = create_new_user(name.strip(), sel_genres, av)
                    st.session_state.update({
                        'user_idx': nid, 'user_name': name.strip(),
                        'is_new_user': True, 'new_user_genres': sel_genres,
                        'page': 'home'
                    })
                    st.success(f"Welcome, {name.strip()}!")
                    time.sleep(0.8)
                    st.rerun()
        with b2:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = 'home'
                st.rerun()

    with c_prev:
        av_p   = st.session_state.get('chosen_avatar', AVATARS[0])
        nm_p   = name if name else "Your Name"
        gn_p   = " · ".join(sel_genres[:3]) if sel_genres else "No genres selected"
        ex     = list_new_users()
        ex_txt = f"{len(ex)} custom profile{'s' if len(ex)!=1 else ''} created" if ex else ""

        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);
                    border-radius:12px;padding:32px;text-align:center;margin-top:48px">
            <div style="font-size:52px;margin-bottom:12px">{av_p}</div>
            <div style="font-family:'Playfair Display',serif;font-size:24px;
                        color:#fff;letter-spacing:-1px">{nm_p}</div>
            <div style="font-size:11px;color:var(--gold);margin-top:6px;
                        letter-spacing:1px">{gn_p}</div>
            <div style="font-size:10px;color:var(--muted);margin-top:16px">
                New User · Cold Start Mode
            </div>
            <div style="font-size:10px;color:var(--muted2);margin-top:8px">{ex_txt}</div>
        </div>""", unsafe_allow_html=True)


def page_about():
    PLOTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "processed", "plots"
    )

    def show_plot(filename, caption=""):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)

    st.markdown("""
    <div class="page-header">
        <div class="page-title">About Cinemate</div>
        <div class="page-subtitle">Architecture · Data · Training · Evaluation · Debiasing</div>
    </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗  Architecture",
        "📊  Data Analysis",
        "🧠  Model Training",
        "📈  Evaluation",
        "⚖️  Debiasing",
    ])

    
    # TAB 1 — ARCHITECTURE
    
    with tab1:
        c1, c2 = st.columns([3, 2], gap="large")

        with c1:
            st.markdown("""
            <div style="font-size:11px;color:var(--gold);letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:12px">The Model</div>
            <p style="color:#aaa;font-size:14px;line-height:1.9;margin-bottom:24px">
                Cinemate is a <strong style="color:#fff">Two-Tower hybrid recommender</strong>
                combining Neural Collaborative Filtering with DistilBERT content embeddings.
                Trained on MovieLens 33M across 50 epochs with popularity-aware BPR loss (γ=0.1).
            </p>
            <div style="font-size:11px;color:var(--gold);letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:12px">Architecture</div>
            <p style="color:#aaa;font-size:14px;line-height:1.9;margin-bottom:24px">
                <strong style="color:#fff">Collaborative Tower</strong> — learns user-movie
                interaction patterns from 33M ratings via a non-linear MLP (128-dim → 64-dim).<br><br>
                <strong style="color:#fff">Content Tower</strong> — DistilBERT embeddings
                (768-dim → 64-dim) of title + genres + genome tags for content-aware ranking.<br><br>
                <strong style="color:#fff">Fusion</strong> — CF ⊕ Content ⊕ (CF ⊙ Content)
                → MLP → relevance score.<br><br>
                <strong style="color:#fff">Debiasing</strong> — Content-similar tail injection
                (7/3 head/tail split, 60th percentile threshold) achieves 17.9% catalogue coverage.
            </p>
            <div style="font-size:11px;color:var(--gold);letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:12px">Cold Start</div>
            <p style="color:#aaa;font-size:14px;line-height:1.9">
                New users (not in training set) receive genre-based recommendations
                via content similarity search — not model-personalised. This is honest
                cold-start handling, not pretending the model generalises to unseen users.
            </p>
            <div style="font-size:11px;color:var(--gold);letter-spacing:2px;
                        text-transform:uppercase;margin:24px 0 12px">Feedback Loop</div>
            <p style="color:#aaa;font-size:14px;line-height:1.9">
                User ratings submitted via the app are stored in SQLite for A/B analysis.
                They do not retrain the model in real-time — offline retraining would be
                required to incorporate new signals.
            </p>""", unsafe_allow_html=True)

        with c2:
            metrics = [
                ("NDCG@10",      "0.1203", "Two-Tower"),
                ("Precision@10", "0.1092", "top-10 accuracy"),
                ("Recall@10",    "0.0286", "test set"),
                ("vs Random",    "+63×",   "NDCG improvement"),
                ("vs SVD",       "+79.9%", "NDCG improvement"),
                ("vs NCF",       "−3.9%",  "marginal gap"),
                ("Coverage",     "17.9%",  "catalogue (7/3 split)"),
                ("Dataset",      "33M",    "ratings"),
                ("Users",        "173K",   "training set"),
                ("Movies",       "27,766", "catalogue"),
                ("Epochs",       "50",     "Two-Tower training"),
                ("BPR γ",        "0.1",    "popularity penalty"),
            ]
            for label, val, sub in metrics:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:10px 14px;margin-bottom:5px;
                            background:var(--bg2);border:1px solid var(--border);
                            border-radius:6px;">
                    <div>
                        <div style="font-size:11px;color:var(--muted);letter-spacing:.5px">
                            {label}
                        </div>
                        <div style="font-size:10px;color:var(--muted2);margin-top:1px">
                            {sub}
                        </div>
                    </div>
                    <div style="font-family:'Playfair Display',serif;font-size:20px;
                                color:var(--gold);font-weight:700">
                        {val}
                    </div>
                </div>""", unsafe_allow_html=True)

    
    # TAB 2 — DATA ANALYSIS
    
    with tab2:
        render_section("Dataset Overview", "MovieLens 33M")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:24px">
            MovieLens 33M contains 33M ratings from 173K users across 27,766 movies.
            The dataset exhibits classic long-tail distribution — a small number of popular
            movies receive the majority of ratings, which motivates the debiasing strategy.
        </p>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            show_plot("01_rating_distribution.png", "Rating Distribution")
        with c2:
            show_plot("02_ratings_per_user.png", "Ratings per User")

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            show_plot("03_ratings_per_movie.png", "Ratings per Movie (long-tail)")
        with c4:
            show_plot("04_ratings_over_time.png", "Ratings over Time")

        render_section("Genre & Tag Analysis")
        c5, c6 = st.columns(2, gap="medium")
        with c5:
            show_plot("05_genre_distribution.png", "Genre Distribution")
        with c6:
            show_plot("07_avg_rating_genre.png", "Average Rating by Genre")

        st.markdown("<br>", unsafe_allow_html=True)
        show_plot("06_top_tags.png", "Top Genome Tags")

    
    # TAB 3 — MODEL TRAINING
    
    with tab3:
        render_section("Two-Tower Training", "50 Epochs · BPR Loss")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            Two-Tower trained for 50 epochs on 16M BPR triplets with popularity-aware loss
            (γ=0.1). ReduceLROnPlateau scheduler with patience=5 — LR halved on plateau.
            Training took ~4.6 hours on a single NVIDIA GPU. Best loss: 0.0605 at epoch 48.
        </p>""", unsafe_allow_html=True)
        show_plot("08_training_loss.png", "Two-Tower Training Loss + Epoch Time")

        st.markdown("<br>", unsafe_allow_html=True)
        render_section("NCF Training", "27 Epochs · BPR Loss")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            Neural Collaborative Filtering trained for 27 epochs. Best NDCG@10: 0.1252 —
            marginally ahead of Two-Tower on pure ranking metrics. NCF is a stronger
            baseline than expected given its simpler architecture.
        </p>""", unsafe_allow_html=True)
        show_plot("08_ncf_training_loss.png", "NCF Training Loss")

    
    # TAB 4 — EVALUATION
    
    with tab4:
        render_section("Model Comparison", "5 Models · NDCG · Precision · Recall")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            Five models benchmarked on 2,000 held-out users. Evaluation uses standard
            IR metrics: NDCG@10 (ranking quality), Precision@10 (hit rate),
            Recall@10 (coverage of relevant items). Positive threshold: rating ≥ 3.5.
        </p>""", unsafe_allow_html=True)

        show_plot("_model_comparison.png", "All Models — NDCG · Precision · Recall @10")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            show_plot("09_model_comparison.png", "Two-Tower vs Baselines")
        with c2:
            show_plot("09_ncf_comparison.png", "NCF vs Baselines")

    
    # TAB 5 — DEBIASING
    
    with tab5:
        render_section("Popularity Bias", "The Problem")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            Collaborative filtering models inherently amplify popularity bias — popular movies
            get more ratings, better representations, and dominate recommendations.
            Without intervention, the raw Two-Tower model recommended from only 0.63%
            of the 27,766-movie catalogue across 500 users.
        </p>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            show_plot("12_popularity_bias.png", "Popularity Bias in Recommendations")
        with c2:
            show_plot("13_genre_bias.png", "Genre Bias — Before Debiasing")

        render_section("Content-Similar Tail Injection", "The Fix")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            7/3 head/tail split: 7 slots filled by model scores (head movies, ≥60th percentile
            popularity), 3 slots filled by cosine similarity to the user's content profile
            (tail movies, &lt;60th percentile). This achieves <strong style="color:var(--gold)">
            17.9% catalogue coverage</strong> with only −16.5% NDCG drop vs raw model —
            significantly better than random tail injection (−21% NDCG drop).
        </p>""", unsafe_allow_html=True)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            show_plot("14_personalisation.png", "Personalisation Score Distribution")
        with c4:
            show_plot("15_genre_diversity.png", "Genre Diversity — After Debiasing")

        render_section("A/B Test Results", "Debiased vs Raw")
        st.markdown("""
        <p style="color:#aaa;font-size:13px;line-height:1.8;margin-bottom:20px">
            A/B test comparing raw model recommendations vs content-similar tail injection.
            Debiased recommendations show improved genre diversity and catalogue coverage
            at an acceptable ranking quality cost.
        </p>""", unsafe_allow_html=True)
        show_plot("10_ab_test_results.png", "A/B Test — Debiased vs Raw Model")



# SIDEBAR

def render_sidebar():
    with st.sidebar:
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
        except Exception:
            health = None
        api_status = "🟢" if health else "🔴"

        st.markdown(f"""
        <div class="sb-logo">Cinemate</div>
        <div class="sb-tagline">{api_status} · User #{st.session_state.user_idx}</div>
        """, unsafe_allow_html=True)

        nav = {
            "Home"        : ("🏠", "home"),
            "Search"      : ("🔍", "search"),
            "New Profile" : ("✨", "new_user"),
            "About"       : ("ℹ️",  "about"),
        }
        for label, (icon, key) in nav.items():
            active = st.session_state.page == key
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        st.markdown('<div class="sb-section">Manual User ID</div>', unsafe_allow_html=True)
        uid = st.number_input("uid", min_value=1, max_value=999999,
                              value=st.session_state.user_idx,
                              label_visibility="collapsed")
        if st.button("Load User", use_container_width=True):
            nd = get_new_user(int(uid))
            st.session_state.user_idx    = int(uid)
            st.session_state.user_name   = nd['name'] if nd else f"User {uid}"
            st.session_state.is_new_user = int(uid) >= NUM_USERS_BASE
            st.session_state.page        = "home"
            st.rerun()

        # st.markdown("---")
        # st.markdown('<div class="sb-section">Debiasing</div>', unsafe_allow_html=True)
        # alpha = st.slider("Popularity Penalty α", 0.0, 1.0,
        #                   st.session_state.alpha, 0.1,
        #                   help="Higher = fewer blockbusters")
        # lam   = st.slider("Genre Diversity λ",    0.0, 1.0,
        #                   st.session_state.lam,   0.1,
        #                   help="Higher = more genre variety")
        # st.session_state.alpha = alpha
        # st.session_state.lam   = lam

        st.markdown("---")
        if health:
            st.success("API Online ✅")
            st.caption(f"Users: {health.get('num_users',1):,} · "
                       f"Movies: {health.get('num_movies',0):,}")
        else:
            st.error("API Offline ❌")
            st.caption("uvicorn api.app:app --reload")

        n_new = len(list_new_users())
        if n_new:
            st.markdown(f"<div style='font-size:10px;color:var(--muted2);margin-top:8px'>"
                        f"{n_new} custom profile{'s' if n_new!=1 else ''}</div>",
                        unsafe_allow_html=True)



# MAIN

def main():
    render_sidebar()

    page = st.session_state.page
    if   page == 'home'    : page_home()
    elif page == 'search'  : page_search()
    elif page == 'new_user': page_new_user()
    elif page == 'about'   : page_about()
    else:
        st.session_state.page = 'home'
        st.rerun()


if __name__ == "__main__":
    main()