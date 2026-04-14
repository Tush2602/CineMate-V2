"""
streamlit_app.py
────────────────
Cinemate — Movie Recommendation Frontend

Pages:
    🎬 Recommendations  → get top-10 for any user
    🔍 Search           → search movies by title
    🎯 Similar Movies   → find similar to a movie
    📊 Model Stats      → live system metrics
    ℹ️  About           → project info + model details

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import requests
import streamlit as st
import pandas as pd 
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#Config
API_URL      = os.getenv("API_URL", "http://localhost:8000")
TMDB_IMG_URL = "https://image.tmdb.org/t/p/w500"
# TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API_KEY = "be50cfdf6bda073b5c74b0a42f7e6dca"
print(f"TMDB id is as follow : {TMDB_API_KEY}" if TMDB_API_KEY else "No TMDB id given")

st.set_page_config(
    page_title="Cinemate V2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#custom CSS
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0A0A0F; }

    /* Movie card */
    .movie-card {
        background    : #13131A;
        border        : 1px solid #2A2A3A;
        border-radius : 12px;
        padding       : 16px;
        margin-bottom : 12px;
        transition    : border-color 0.2s;
    }
    .movie-card:hover {
        border-color  : #6366F1;
    }

    /* Rank badge */
    .rank-badge {
        background    : #6366F1;
        color         : white;
        border-radius : 6px;
        padding       : 2px 10px;
        font-size     : 13px;
        font-weight   : 600;
        display       : inline-block;
        margin-bottom : 8px;
    }

    /* Score badge */
    .score-badge {
        background    : #1A2A1A;
        color         : #4ADE80;
        border        : 1px solid #166534;
        border-radius : 6px;
        padding       : 2px 10px;
        font-size     : 12px;
        font-weight   : 500;
        display       : inline-block;
        margin-left   : 8px;
    }

    /* Genre tag */
    .genre-tag {
        background    : #1E1E2E;
        color         : #94A3B8;
        border-radius : 4px;
        padding       : 2px 8px;
        font-size     : 11px;
        display       : inline-block;
        margin-right  : 4px;
        margin-top    : 4px;
    }

    /* KPI card */
    .kpi-card {
        background    : #13131A;
        border        : 1px solid #2A2A3A;
        border-radius : 12px;
        padding       : 20px;
        text-align    : center;
    }
    .kpi-value {
        font-size     : 32px;
        font-weight   : 700;
        color         : #6366F1;
    }
    .kpi-label {
        font-size     : 13px;
        color         : #64748B;
        margin-top    : 4px;
    }

    /* Model metrics table */
    .metrics-table {
        width         : 100%;
        border-collapse: collapse;
    }
    .metrics-table th {
        background    : #1E1E2E;
        padding       : 10px 16px;
        text-align    : left;
        font-size     : 13px;
        color         : #94A3B8;
    }
    .metrics-table td {
        padding       : 10px 16px;
        border-bottom : 1px solid #1E1E2E;
        font-size     : 13px;
    }
    .best-row td {
        color         : #6366F1;
        font-weight   : 600;
    }

    /* Divider */
    .section-divider {
        border        : none;
        border-top    : 1px solid #1E1E2E;
        margin        : 24px 0;
    }

    /* Cold start warning */
    .cold-start-warning {
        background    : #1A1510;
        border        : 1px solid #92400E;
        border-radius : 8px;
        padding       : 12px 16px;
        color         : #FCD34D;
        font-size     : 13px;
        margin-bottom : 16px;
    }
</style>
""", unsafe_allow_html=True)

#API helpers

def api_get(endpoint :str, params : dict =None):
    """Make GET request to API — returns JSON or None."""
    try:
        response = requests.get(f"{API_URL}{endpoint}",
                                params=params,
                                timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None
            

def api_post(endpoint:str, data:dict):
    """Make POST request to API — returns JSON or None."""
    try:
        response= requests.post(f"{API_URL}{endpoint}",
                                json=data,
                                timeout=10)
        if response.status_code ==200:
            return response.json()
        return None
    except Exception:
        return None
    
def check_api_health():
    """Check if API is running"""
    result = api_get("/health")
    return result is not None

def get_poster_url(tmdb_id:int)->str:
    """Fetch poster URL via TMDB API."""
    if not TMDB_API_KEY or not tmdb_id:
        return ""
    try: 
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params, timeout=60)
        if response.status_code==200:
            data = response.json()
            path = data.get("poster_path", "")
            if path:
                return f"{TMDB_IMG_URL}{path}"
            
    except Exception:
        pass
    return ""

#UI Components 
def render_movie_card(movie:dict, show_score: bool= True):
    """Render a single movie as a styled card."""
    title  = movie.get('title', 'Unknown')
    genres = movie.get('genres', '')
    rank   = movie.get('rank', '')
    score  = movie.get('score', 0.0)

    # Clean up genre tags
    genre_tags = ""
    if genres:
        # Split by space and take first 4
        for g in genres.split()[:4]:
            genre_tags += f'<span class="genre-tag">{g}</span>'

    # Build badges
    rank_html  = f'<span class="rank-badge">#{rank}</span>' if rank else ""
    score_html = f'<span class="score-badge">score: {score:.3f}</span>' if (show_score and score) else ""

    # THE FIX: Wrap in a single clean string block
    html_content = f"""
    <div class="movie-card">
        <div>{rank_html}{score_html}</div>
        <div style="font-size:15px; font-weight:600; margin-top:8px; margin-bottom:6px; color:#F1F5F9;">
            {title}
        </div>
        <div style="display: flex; flex-wrap: wrap;">
            {genre_tags}
        </div>
    </div>
    """
    # Use st.write or st.markdown with strict allow_html
    st.markdown(html_content, unsafe_allow_html=True)

def render_movie_grid(movies:list, cols:int=2, show_score: bool=True):
    """
    Render movies in a responsive grid layout.
    Optionally shows poster images if TMDB key available.
    """
    columns = st.columns(cols)
    for i, movie in enumerate(movies):
        with columns[i%cols]:
            #Try to show poster
            tmdb_id =movie.get("tmdb_id", 0)
            poster_url = get_poster_url(tmdb_id)
            if poster_url:
                st.image(poster_url, width=180)
            
            render_movie_card(movie, show_score)


def render_kpi(value, label:str, col):
    """Render a KPI metric card."""
    with col: 
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


#Sidebar
def render_sidebar():
    """Render navigation sidebar.."""
    with st.sidebar:
        st.markdown("## 🎬 Cinemate")
        st.markdown(
            "*Two-Tower Hybrid Recommender*",
            help="NCF + DistilBERT content embeddings"
        )
        st.markdown("---")

        # API status
        if check_api_health():
            st.success("API connected", icon="✅")
        else:
            st.error("API not reachable", icon="❌")
            st.markdown(
                "Run: `uvicorn api.main:app --reload`"
            )

        st.markdown("---")

        #Navigation
        page = st.radio("Navigate", options=["🎬 Recommendations",
                                            "🔍 Search Movies",
                                            "🎯 Similar Movies",
                                            "📊 Model Stats",
                                            "📉 Dashboard",
                                            "ℹ️  About",
                                            ],
                                            label_visibility="collapsed")
        st.markdown("---")

        # Quick stats from health endpoint
        health = api_get("/health")
        if health:
            st.markdown("**System Info**")
            st.caption(
                f"Users: {health.get('num_users', 0):,}"
            )
            st.caption(
                f"Movies: {health.get('num_movies', 0):,}"
            )
            st.caption(
                f"Model: {health.get('version', '1.0.0')}"
            )

    return page


# Page 1. Recommendations page
def page_recommendations():
    st.markdown("## 🎬 Get Recommendations")
    st.markdown(
        "Enter a user ID to get personalised movie recommendations "
        "from the Two-Tower hybrid model."
    )
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        user_idx = st.number_input("User ID",
                                   min_value=0,
                                   max_value=172999,
                                   value=0,
                                   step=1,
                                   help="Encoded user index (0 to 173,133)")
        
    with col2:
        top_k =st.slider("Number of recommendations", min_value=5, max_value=50, value=10)

    with col3:
        strategy= st.selectbox("Strategy", options=["ann", "brute_force"], help=("ann = fast ANN search via ChromaDB\n"
                       "brute_force = exact, slower"))
        
    get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

    if get_recs:
        with st.spinner("Generating Recommendations ....."):
            result = api_get(f"/recommend/{user_idx}",
                              params={"top_k":top_k,
                                      "strategy": strategy})
            
        if result is None:
            st.error("Could not connect to API. "
                "Make sure the API is running.")
            return
        
        #Cold Start warning
        if result.get('is_cold_start'):
            st.markdown("""
            <div class="cold-start-warning">
                ⚠️ Cold start user — this user has no rating
                history in training data. Recommendations are
                based on content features only.
            </div>
            """, unsafe_allow_html=True)

        #Response time 
        rt = result.get("response_time_ms", 0)
        st.caption(f"Response time : {rt:.1f}ms | "
                   f"Strategy : {result.get('strategy')} | "
                   f"Model : {result.get('model_version')}")
        
        st.markdown("---")

        #Render recommendations
        recs = result.get("recommendations", [])
        if not recs:
            st.warning("No recommendations returned.")
            return
        
        st.markdown(f"### Top {len(recs)} recommendations "
            f"for User {user_idx}")
        
        render_movie_grid(recs, cols=2, show_score=True)

        #Feedback section
        st.markdown("---")
        st.markdown("### ⭐ Rate a Recommendation")
        st.caption("Your feedback is logged for A/B test analysis.")

        movie_options = {f"#{r['rank']} {r['title']}": r['movie_idx'] for r in recs}

        selected = st.selectbox("Select a movie to rate", options=list(movie_options.keys()))
        rating = st.slider("Your rating", min_value = 0.5, max_value = 5.0, value = 4.0, step = 0.5)
        if st.button("Submit Rating"):
            feedback= api_post("/feedback", {
                "user_idx" : int(user_idx),
                "movie_idx": movie_options[selected],
                "rating"   : rating,
                "from_rec" : True
            })
            if feedback:
                st.success(
                    f"Rating {rating}⭐ logged. "
                    f"Thank you!"
                )
            else:
                st.error("Failed to submit rating.")

# PAGE 2 — SEARCH
def page_search():
    st.markdown("## 🔍 Search Movies")
    st.markdown("Search the MovieLens catalogue by title.")
    st.markdown("---")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Movie title", placeholder = "e.g. Inception, The Dark Knight...",
                            label_visibility = "collapsed"
        )
    with col2:
        limit = st.number_input("Max results", min_value=5, max_value=50, value=10 )

    if query:
        with st.spinner("Searching..."):
            result = api_get("/search",params={"q": query, "limit": limit})

        if result is None:
            st.error("Search failed — API not reachable.")
            return

        movies = result.get('results', [])
        total  = result.get('total', 0)

        if not movies:
            st.info(f"No movies found for '{query}'.")
            return

        st.markdown(f"**{total}** results for *'{query}'*")
        st.markdown("---")

        render_movie_grid(movies, cols=2, show_score=False)


# PAGE 3 — SIMILAR MOVIES
def page_similar():
    st.markdown("## 🎯 Similar Movies")
    st.markdown("Find movies similar to one you already know. "
                "Uses ChromaDB ANN search on content embeddings."
                )
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        # Search for the source movie first
        query = st.text_input("Search for source movie", placeholder = "Type a movie title to start...")
    with col2:
        top_k = st.slider("Similar movies", 5, 20, 10)

    movie_idx = None

    if query:
        search_result = api_get("/search", params={"q": query, "limit": 8})
        if search_result and search_result.get('results'):
            movies = search_result['results']
            options = {
                f"{m['title']} [{m['genres'][:30]}]":
                m['movie_idx']
                for m in movies
            }
            selected_title = st.selectbox(
                "Select movie",
                options = list(options.keys())
            )
            movie_idx = options[selected_title]

    if movie_idx is not None:
        if st.button(
            "Find Similar Movies",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Finding similar movies..."):
                result = api_get(
                    f"/similar/{movie_idx}",
                    params={"top_k": top_k}
                )

            if result is None:
                st.error(
                    "Similar movie search failed."
                )
                return

            source = result.get('source_movie', {})
            similar = result.get('similar', [])

            st.markdown("---")
            st.markdown(
                f"**Source:** {source.get('title', '')}  "
                f"— *{source.get('genres', '')}*"
            )
            st.markdown(
                f"### {len(similar)} similar movies"
            )

            render_movie_grid(
                similar, cols=2, show_score=True
            )

#Page stats


# ══════════════════════════════════════════════════════════
# PAGE 4 — MODEL STATS
# ══════════════════════════════════════════════════════════

def page_stats():
    st.markdown("## 📊 Model Performance & System Stats")
    st.markdown("---")

    # ── Model Results Table ───────────────────────────────
    st.markdown("### Model Comparison")
    st.caption(
        "Evaluated on MovieLens 33M | "
        "Time-based 80/20 split | 1,000 test users"
    )

    results_data = {
        "Model"         : ["Random", "Popularity",
                           "SVD", "NCF", "Two-Tower"],
        "NDCG@10"       : [0.0019, 0.0845, 0.0669,
                           0.1276, 0.1203],
        "Precision@10"  : [0.0018, 0.0749, 0.0607,
                           0.1154, 0.1084],
        "Recall@10"     : [0.0004, 0.0225, 0.0221,
                           0.0318, 0.0280],
    }

    df = pd.DataFrame(results_data)

    # Highlight best per column
    def highlight_best(col):
        if col.name == "Model":
            return [""] * len(col)
        max_val = col.max()
        return [
            "background-color: #1E1A3A; color: #6366F1; "
            "font-weight: bold"
            if v == max_val else ""
            for v in col
        ]

    st.dataframe(
        df.style.apply(highlight_best),
        use_container_width = True,
        hide_index          = True
    )

    st.caption(
        "NDCG@10 = ranking quality (primary metric). "
        "Higher is better. "
        "Best value per column highlighted in purple."
    )

    # ── Training History ──────────────────────────────────
    st.markdown("---")
    st.markdown("### Training Loss Curves")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**NCF Training (20 epochs)**")
        ncf_loss = [
            0.1128, 0.0881, 0.0792, 0.0765, 0.0748,
            0.0735, 0.0723, 0.0714, 0.0709, 0.0703,
            0.0703, 0.0700, 0.0699, 0.0697, 0.0695,
            0.0694, 0.0695, 0.0693, 0.0693, 0.0694
        ]
        ncf_df = pd.DataFrame({
            "Epoch": range(1, 21),
            "BPR Loss": ncf_loss
        })
        st.line_chart(
            ncf_df.set_index("Epoch"),
            color = "#6366F1"
        )

    with col2:
        st.markdown("**Two-Tower Training (30 epochs)**")
        tt_loss = [
            0.1189, 0.1098, 0.1035, 0.0981, 0.0965,
            0.0961, 0.0945, 0.0921, 0.0913, 0.0908,
            0.0908, 0.0908, 0.0907, 0.0904, 0.0903,
            0.0903, 0.0902, 0.0902, 0.0902, 0.0901,
            0.0902, 0.0902, 0.0904, 0.0902, 0.0828,
            0.0818, 0.0818, 0.0817, 0.0818, 0.0817
        ]
        tt_df = pd.DataFrame({
            "Epoch": range(1, 31),
            "BPR Loss": tt_loss
        })
        st.line_chart(
            tt_df.set_index("Epoch"),
            color = "#06B6D4"
        )

    # ── Live System Stats ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Live System Stats")

    stats = api_get("/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        render_kpi(
            f"{stats.get('total_recommendations', 0):,}",
            "Total Recommendations", c1
        )
        render_kpi(
            f"{stats.get('unique_users', 0):,}",
            "Unique Users Served", c2
        )
        render_kpi(
            f"{stats.get('unique_movies', 0):,}",
            "Unique Movies Recommended", c3
        )
        coverage = stats.get('catalogue_coverage', 0)
        render_kpi(
            f"{coverage:.1%}",
            "Catalogue Coverage", c4
        )
    else:
        st.info(
            "Live stats unavailable — "
            "API not connected."
        )

    # ── A/B Test Results ──────────────────────────────────
    st.markdown("---")
    st.markdown("### A/B Test — Recommendation vs Organic")
    st.caption(
        "Compares average rating of recommended movies "
        "vs organically discovered movies."
    )

    ab_stats = api_get("/ab-test")
    if ab_stats:
        treatment = ab_stats.get('treatment', {})
        control   = ab_stats.get('control', {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label = "Recommended Movies (Treatment)",
                value = f"{treatment.get('avg_rating', 0):.2f} ⭐",
                delta = f"n={treatment.get('n', 0):,}"
            )
        with col2:
            st.metric(
                label = "Organic Movies (Control)",
                value = f"{control.get('avg_rating', 0):.2f} ⭐",
                delta = f"n={control.get('n', 0):,}"
            )

        t_avg = treatment.get('avg_rating', 0)
        c_avg = control.get('avg_rating', 0)
        if t_avg and c_avg:
            lift = ((t_avg - c_avg) / max(c_avg, 1)) * 100
            if lift > 0:
                st.success(
                    f"Recommendations show {lift:.1f}% "
                    f"higher average rating vs organic."
                )
            else:
                st.warning(
                    "Not enough feedback data yet. "
                    "Submit ratings via the "
                    "Recommendations page."
                )
    else:
        st.info(
            "No A/B test data yet. "
            "Use the Recommendations page and "
            "submit ratings to populate this."
        )

def dashboard():
    import plotly.express      as px
    import plotly.graph_objects as go

    st.markdown("## 📊 Model Performance & System Stats")
    st.markdown("---")

    # ── Model comparison ──────────────────────────────────
    st.markdown("### Model Comparison")

    models    = ['Random','Popularity','SVD','NCF','Two-Tower']
    ndcg_vals = [0.0019, 0.0845, 0.0669, 0.1276, 0.1235]
    colors    = ['#94A3B8','#64748B','#3B82F6',
                 '#8B5CF6','#10B981']

    fig = go.Figure(go.Bar(
        x            = models,
        y            = ndcg_vals,
        marker_color = colors,
        text         = [f'{v:.4f}' for v in ndcg_vals],
        textposition = 'outside',
    ))
    fig.update_layout(
        yaxis_title  = 'NDCG@10',
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor= 'rgba(0,0,0,0)',
        font_color   = 'white',
        showlegend   = False,
        height       = 400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Training loss ─────────────────────────────────────
    st.markdown("### Training Loss Curves")

    ncf_loss = [0.1128,0.0881,0.0792,0.0765,0.0748,
                0.0735,0.0723,0.0714,0.0709,0.0703,
                0.0703,0.0700,0.0699,0.0697,0.0695,
                0.0694,0.0695,0.0693,0.0693,0.0694]
    tt_loss  = [0.1185,0.1111,0.1066,0.0995,0.0956,
                0.0933,0.0921,0.0920,0.0919,0.0917,
                0.0915,0.0917,0.0916,0.0916,0.0916,
                0.0915,0.0915,0.0840,0.0834,0.0833,
                0.0832,0.0829,0.0828,0.0827,0.0827,
                0.0827,0.0827,0.0829,0.0826,0.0826,
                0.0825,0.0826,0.0826,0.0828,0.0827,
                0.0827,0.0827,0.0766,0.0760,0.0764,
                0.0764,0.0762,0.0757,0.0753,0.0748,
                0.0742,0.0740,0.0738,0.0737,0.0736]

    col1, col2 = st.columns(2)
    with col1:
        fig_ncf = px.line(
            x=list(range(1,21)), y=ncf_loss,
            title='NCF — 20 epochs',
            labels={'x':'Epoch','y':'BPR Loss'}
        )
        fig_ncf.update_traces(line_color='#8B5CF6')
        fig_ncf.update_layout(
            plot_bgcolor ='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300
        )
        st.plotly_chart(fig_ncf, use_container_width=True)

    with col2:
        fig_tt = px.line(
            x=list(range(1,51)), y=tt_loss,
            title='Two-Tower — 50 epochs',
            labels={'x':'Epoch','y':'BPR Loss'}
        )
        fig_tt.update_traces(line_color='#10B981')
        fig_tt.update_layout(
            plot_bgcolor ='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300
        )
        st.plotly_chart(fig_tt, use_container_width=True)

    # ── A/B test summary ──────────────────────────────────
    st.markdown("---")
    st.markdown("### A/B Test Summary")

    ab_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "ab_test_results.json"
    )
    if os.path.exists(ab_path):
        with open(ab_path) as f:
            ab = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Control NDCG",
                  f"{ab['control_ndcg']['mean']:.4f}")
        c2.metric("Treatment NDCG",
                  f"{ab['treatment_ndcg']['mean']:.4f}",
                  delta=f"+{ab['lift']['relative_pct']:.1f}%")
        c3.metric("p-value",
                  f"{ab['hypothesis_test']['p_value']:.4f}")
        c4.metric("Decision",
                  ab.get('decision', 'UNKNOWN'))
    else:
        st.info("Run notebook 05_ab_test.ipynb first.")

    # ── Live stats ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Live System Stats")
    stats = api_get("/stats")
    if stats:
        c1,c2,c3,c4 = st.columns(4)
        render_kpi(
            f"{stats.get('total_recommendations',0):,}",
            "Recommendations Served", c1
        )
        render_kpi(
            f"{stats.get('unique_users',0):,}",
            "Unique Users", c2
        )
        render_kpi(
            f"{stats.get('unique_movies',0):,}",
            "Unique Movies", c3
        )
        render_kpi(
            f"{stats.get('catalogue_coverage',0):.1%}",
            "Catalogue Coverage", c4
        )
    else:
        st.info("API not connected — start FastAPI first.")

# ══════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════

def page_about():
    st.markdown("## ℹ️ About Cinemate")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
### Project Overview

**Cinemate** is a Two-Tower hybrid movie recommendation
system trained on the MovieLens 33M dataset.

#### Architecture
- **Collaborative Tower** — Neural Collaborative Filtering
  learns user-movie interaction patterns from 25M ratings
- **Content Tower** — DistilBERT embeddings of movie
  title + genres + genome tags
- **Fusion Layer** — combines both towers via MLP
- **ANN Retrieval** — ChromaDB approximate nearest
  neighbour search for millisecond inference

#### Dataset
- **33,832,162** ratings across **330,975** users
  and **83,239** movies
- **MovieLens 33M** — time-based 80/20 train/test split
- **99.88% sparse** user-item matrix
- Genome scores, user tags, and TMDB poster data

#### Results

| Model | NDCG@10 | vs SVD |
|---|---|---|
| Random | 0.0019 | — |
| Popularity | 0.0845 | — |
| SVD | 0.0669 | baseline |
| NCF | 0.1276 | +90.9% |
| Two-Tower | 0.1203 | +79.9% |

#### Tech Stack
`PyTorch` `DistilBERT` `ChromaDB` `FastAPI`
`SQLAlchemy` `SQLite` `Streamlit` `Plotly`
        """)

    with col2:
        st.markdown("### Model Details")

        st.markdown("**NCF Model**")
        ncf_details = {
            "embed_dim"   : 64,
            "MLP layers"  : "256→128→64→32→1",
            "Parameters"  : "12,933,889",
            "Loss"        : "BPR",
            "Epochs"      : 20,
            "Best loss"   : 0.0694,
        }
        for k, v in ncf_details.items():
            st.caption(f"**{k}**: {v}")

        st.markdown("---")

        st.markdown("**Two-Tower Model**")
        tt_details = {
            "CF embed_dim"    : 128,
            "Tower output"    : 32,
            "BERT model"      : "distilbert-base-uncased",
            "BERT dim"        : 768,
            "Parameters"      : "~32M",
            "Loss"            : "BPR",
            "Epochs"          : 30,
            "Best loss"       : 0.0817,
        }
        for k, v in tt_details.items():
            st.caption(f"**{k}**: {v}")

        st.markdown("---")
        st.markdown("**Links**")
        st.markdown(
            "📁 [GitHub Repository](#)  \n"
            "📊 [Plotly Dashboard](#)  \n"
            "📝 [Medium Article](#)"
        )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if page == "🎬 Recommendations":
        page_recommendations()
    elif page == "🔍 Search Movies":
        page_search()
    elif page == "🎯 Similar Movies":
        page_similar()
    elif page == "📊 Model Stats":
        page_stats()
    elif page == "📉 Dashboard":
        dashboard()
    elif page == "ℹ️  About":
        page_about()


if __name__ == "__main__":
    main()       

        
