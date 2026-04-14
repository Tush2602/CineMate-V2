"""
main.py
───────
FastAPI application — recommendation serving layer.

Endpoints:
    GET  /health                     → system health check
    GET  /recommend/{user_idx}       → top-K recommendations
    POST /recommend                  → same with request body
    GET  /movie/{movie_idx}          → single movie detail
    GET  /similar/{movie_idx}        → similar movies
    GET  /search?q=inception         → movie title search
    POST /feedback                   → log user rating
    GET  /stats                      → system stats for dashboard
    GET  /genres                     → all available genres

Run locally:
    uvicorn api.main:app --reload --port 8000

Visit auto-generated docs:
    http://localhost:8000/docs
"""


import os 
import sys 
import time 
import pickle 
import requests
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db.database import get_db, init_db
from db import crud
from api.schemas import (RecommendationRequest, RecommendationResponse, RecommendedMovie, MovieDetail, MovieBase, 
                         SearchResponse, SimilarMoviesResponse, FeedbackRequest, FeedbackResponse, HealthResponse,StatsResponse)
from src.recommendation import Recommender

#paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

#TMDB config
TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_URL  = "https://image.tmdb.org/t/p/w500"

#app lifecycle

# Global recommender instance — loaded once at startup
recommender : Optional[Recommender] = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    """
    Startup and shutdown logic.
    Model is loaded ONCE at startup — not per request.
    This is critical for performance.
    """
    global recommender
    print("Starting Cinemate API....")
    print()

    #Initialize database tables
    init_db()

    # Load recommendation model
    print("Loading Two-Tower model...")
    recommender = Recommender()
    recommender.load(model_type="two_tower")
    print("Model loaded successfully.")
    print()
    print("API ready.")

    yield

    # Shutdown cleanup
    print("Shutting down API...")
    recommender = None

#App instance

app = FastAPI(title="Cinemate V2 API",
              description=("Two-Tower Hybrid Movie Recommendation Engine\n\n"
                            "Built on MovieLens 33M | "
                            "Neural Collaborative Filtering + DistilBERT"),
                            version="1.0.0",
                            lifespan=lifespan)

# CORS — allows Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

#Helper function 
def get_poster(tmdbid : int) -> Optional[str]:
    """
    Fetch movie poster URL from TMDB API.
    Returns None if no API key or tmdb_id=0.
    """
    if not TMDB_API_KEY or not tmdbid :
        return None
    
    try: 
        url = f"{TMDB_BASE_URL}/movie/{tmdbid}"
        params= {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code ==200:
            data = response.json()
            path = data.get("poster_path")
            if path:
                return f"{TMDB_IMG_URL}{path}"
    except Exception:
        pass 
    return None

def build_recommended_movie(rank:int, rec: dict, fetch_poster:bool = False) -> RecommendedMovie:
    """
    Convert raw recommendation dict to schema object.
    Optionally fetches TMDB poster.
    """
    poster = None
    if fetch_poster:
        poster = get_poster(rec.get('tmdb_id', 0))

    return RecommendedMovie(
        rank       = rank,
        movie_idx  = rec['movie_idx'],
        title      = rec.get('title', 'Unknown'),
        genres     = rec.get('genres', ''),
        tmdb_id    = rec.get('tmdb_id', 0),
        score      = round(rec.get('score', 0.0), 4),
        poster_url = poster
    )

#endpoints
@app.get("/health", response_model=HealthResponse, tags=['System'])
def health_check(db:Session=Depends(get_db)):
    """
    System health check.
    Returns model status, DB connection, and dataset stats.
    """
    model_loaded = recommender is not None
    db_connected = False
    num_movies = 0

    try:
        num_movies   = crud.get_total_movies(db)
        db_connected = True
    except Exception as e:
        print("Logging failed:", e)

    num_users= recommender.constants.get("NUM_USERS", 0) if model_loaded else 0
    num_movies_model = recommender.constants.get("NUM_MOVIES") if model_loaded else 0

    return HealthResponse(
        status       = "healthy" if model_loaded else "degraded",
        model_loaded = model_loaded,
        db_connected = db_connected,
        num_users    = num_users,
        num_movies   = num_movies_model,
    )

@app.get("/recommend/{user_idx}", response_model = RecommendationResponse, tags=["Recommendations"])
def get_recommendations(user_idx: int, top_k:int = Query(10, ge=1, le=50),
                        strategy : str = Query("ann"), fetch_posters: bool = Query(False), db : Session = Depends(get_db)):
    """
    Get top-K movie recommendations for a user.

    - user_idx: encoded user index (0 to NUM_USERS-1)
    - top_k: number of recommendations (default 10)
    - strategy: ann (fast) or brute_force (exact)
    - fetch_posters: fetch TMDB poster URLs (slower)
    """

    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    #validate user exists
    num_users = recommender.constants['NUM_USERS']
    if user_idx < 0 or user_idx >= num_users:
        raise HTTPException(
            status_code = 404,
            detail      = f"user_idx {user_idx} out of range "
                          f"(0 to {num_users - 1})"
        )

    start = time.time()

    #check cold start 
    is_cold_start = (user_idx not in getattr(recommender, "user_positive_sets", {}))

    #Get recommendations
    try:
        recs = recommender.recommend(user_idx=user_idx, top_k=top_k, strategy=strategy)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Recommendation failed: {str(e)}")
    
    raw_recs = recommender.recommend(user_idx=user_idx, top_k=top_k, strategy=strategy)
    hydrated_recs = []
    for r in raw_recs:
        movie_from_db = crud.get_movie_by_idx(db, r['movie_idx'])
        if movie_from_db:
            hydrated_recs.append({
                "movie_idx": r['movie_idx'],
                "score": r['score'],
                "title": movie_from_db.title,
                "genres": movie_from_db.genres,
                "tmdb_id": movie_from_db.tmdb_id  # <--- THIS FIXES THE 0
            })

    response_time = (time.time() - start) * 1000

    #build response objects
    rec_movies = [build_recommended_movie(rank =i+1, rec= rec , fetch_poster=fetch_posters)
                  for i , rec in enumerate(recs)]
    
    #Log into database asynchronously 
    try:
        crud.get_or_create_users(db, user_idx, is_cold_start)
        crud.log_recommendations(db=db, user_idx=user_idx, recommendations=[{'movie_idx': r.movie_idx, "score": r.score}
                                for r in rec_movies], model_version="two_tower", strategy=strategy)
    except Exception as e:
        pass 

    return RecommendationResponse(
        user_idx=user_idx,
        recommendations=rec_movies,
        model_version="two_tower",
        strategy=strategy,
        response_time_ms=round(response_time, 2),
        is_cold_start=is_cold_start
    )
        
@app.post("/recommend", response_model = RecommendationResponse, tags=['Recommendations'])
def post_recommendations(request:RecommendationRequest, db:Session = Depends(get_db)):
    """
    Same as GET /recommend/{user_idx} but via POST body.
    Useful for programmatic access with more options.
    """
    return get_recommendations(user_idx = request.user_idx,
                               top_k = request.top_k,
                               strategy=request.strategy,
                               fetch_posters=False,
                               db =db
                               )

@app.get("/movie/{movie_idx}", response_model = MovieDetail, tags=["Movies"])
def get_movie(movie_idx :int, db:Session= Depends(get_db)):
    """
    Get full details for a single movie.
    """
    movie = crud.get_movie_by_idx(db, movie_idx)
    if not movie:
        raise HTTPException(status_code=404, 
                            detail =f"Movie {movie_idx} not found"
                            )
    return movie

@app.get("/similar/{movie_idx}", response_model=SimilarMoviesResponse, tags=["Movies"])
def get_similar_movies(movie_idx : int, top_k:int =Query(10, ge=1, le=50), db:Session = Depends(get_db)):
    """
    Get movies similar to a given movie.
    Uses ChromaDB ANN search on content embeddings.
    """
    if recommender is None:
        raise HTTPException(status_code=404,
                            detail=f"Movie {movie_idx} not found")
    
    # Get source movie
    source = crud.get_movie_by_idx(db, movie_idx)
    if not source:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_idx} not found")

    #use content embeddings for similarity search
    try: 
        import torch
        from src.chroma_db import query_similar

        movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(recommender.device)
    
        with torch.no_grad():
            all_embs = recommender.model\
                    .get_all_movie_embeddings()
            movie_emb = all_embs[movie_idx].cpu().numpy()

        candidates = query_similar(movie_emb, n_results=top_k + 1)

        # Exclude the source movie itself
        similar = [c for c in candidates if c['movie_idx']!= movie_idx][:top_k]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")
    

    source_schema = MovieBase(
        movie_idx = source.movie_idx,
        title     = source.title,
        genres    = source.genres,
        tmdb_id   = source.tmdb_id,
        imdb_id   = source.imdb_id
    )

    similar_movies = [build_recommended_movie(i + 1, rec) for i, rec in enumerate(similar)]

    return SimilarMoviesResponse(source_movie = source_schema, similar= similar_movies)


@app.get("/search", response_model=SearchResponse, tags=["Movies"])
def search_movie(q: str= Query(..., min_length=1, description="Movie title search query"),
                               limit: int =Query(10, ge=1, le=50),
                               db:Session = Depends(get_db)):
    """
    Search movies by title.
    Used by Streamlit search bar.
    """
    results = crud.search_movies(db, q, limit)
    movies = [MovieBase(movie_idx=m.movie_idx,
                        title = m.title,
                        genres=m.genres,
                        tmdb_id =m.tmdb_id,
                        imdb_id= m.imdb_id) for m in results]
    
    return SearchResponse(query=q, results=movies, total= len(movies))

@app.post("/feedback", response_model = FeedbackResponse, tags=['Feedback'])
def submit_feedback(request: FeedbackRequest , db:Session = Depends(get_db)):
    """
    Log user rating feedback.

    from_rec=True → user rated a recommended movie
    from_rec=False → organic rating

    Used for A/B test analysis in dashboard.
    """
    crud.log_rating(db=db, user_idx=request.user_idx, movie_idx = request.movie_idx,
                    rating= request.rating, from_rec= request.from_rec)
    return FeedbackResponse(status="logged",
                            user_idx=request.user_idx,
                            movie_idx=request.movie_idx,
                            rating=request.rating)

@app.get("/genres", tags=["Movies"])
def get_genres(db:Session= Depends(get_db)):
    """
    Get all unique genres in the database.
    Used by Streamlit genre filter dropdown.
    """
    genres = crud.get_all_genres(db)
    return {"genres": genres, "total": len(genres)}

@app.get("/stats", response_model=StatsResponse, tags=['Analytics'])
def get_stats(db:Session=Depends(get_db)):
    """
    System stats for Plotly dashboard.
    Returns recommendation counts, coverage, model breakdown.
    """
    stats= crud.get_recommendation_stats(db)
    return StatsResponse(**stats)

@app.get("/popular", tags=['Movies'])
def get_popular_movies(limit:int=Query(20, ge=1, le=20), db:Session = Depends(get_db)):
    """
    Most recommended movies — for dashboard bias analysis.
    """
    popular = crud.get_most_recommended_movies(db, limit)
    return {
        "movies" : popular,
        "total"  : len(popular)
    }

@app.get("/ab-test", tags=["Analytics"])
def get_ab_test_results(db:Session= Depends(get_db)):
    """
    A/B test results for dashboard Tab 4.
    Compares ratings from recommended vs organic movies.
    """
    stats = crud.get_ab_test_stats(db)
    return stats

#Root
@app.get("/", tags=['System'])
def root():
    return {
        "name" : "Cinemate V2 API",
        "version" : "2.0.0",
        "description" : "Two tower hybrid movie recommender.",
        "docs": "/docs",
        "health": "/health"
    }
        


    











