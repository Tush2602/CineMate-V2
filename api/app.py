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
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()
from db.database import get_db, init_db
from db import crud
from api.schemas import (RecommendationRequest, RecommendationResponse, RecommendedMovie, MovieDetail, MovieBase, 
                         SearchResponse, SimilarMoviesResponse, FeedbackRequest, FeedbackResponse, HealthResponse,StatsResponse)
from src.recommendation import Recommender

#paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

#TMDB config
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_URL = "https://image.tmdb.org/t/p/w500"

#app lifecycle
# Global recommender instance
recommender : Optional[Recommender] = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    global recommender
    print("Starting Cinemate API....")
    print()

    #Initialize database tables
    init_db()

    # Populate movies if DB is empty
    from db.database import SessionLocal
    from db.init_db import populate_movies
    db_session = SessionLocal()
    try:
        populate_movies(db_session)
    finally:
        db_session.close()

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Helper function 
def get_poster(tmdbid : int) -> Optional[str]:
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
    model_loaded = recommender is not None
    db_connected = False
    num_movies = 0

    try:
        num_movies = crud.get_total_movies(db)
        db_connected = True
    except Exception as e:
        print("Logging failed:", e)

    num_users= recommender.constants.get("NUM_USERS", 0) if model_loaded else 0
    num_movies_model = recommender.constants.get("NUM_MOVIES") if model_loaded else 0

    return HealthResponse(
        status = "healthy" if model_loaded else "degraded",
        model_loaded = model_loaded,
        db_connected = db_connected,
        num_users = num_users,
        num_movies = num_movies_model,
    )

@app.get("/recommend/gems/{user_idx}", tags=["Recommendations"])
def get_hidden_gems(user_idx: int, top_k: int = Query(7, ge=1, le=20), db: Session = Depends(get_db)):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    num_users = recommender.constants['NUM_USERS']
    if user_idx < 0 or user_idx >= num_users:
        raise HTTPException(status_code=404, detail=f"user_idx out of range")

    try:
        recs = recommender.recommend_tail_only(user_idx=user_idx, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"user_idx": user_idx, "gems": recs}


@app.get("/recommend/{user_idx}", response_model = RecommendationResponse, tags=["Recommendations"])
def get_recommendations(user_idx: int, top_k:int = Query(10, ge=1, le=50),
                        strategy : str = Query("two_tower"), fetch_posters: bool = Query(False), db : Session = Depends(get_db)):

    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    #validate user exists
    num_users = recommender.constants['NUM_USERS']
    if user_idx < 0 or user_idx >= num_users:
        raise HTTPException(
            status_code = 404,
            detail  = f"user_idx {user_idx} out of range "
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
    
    # raw_recs = recommender.recommend(user_idx=user_idx, top_k=top_k, strategy=strategy)
    # hydrated_recs = []
    # for r in raw_recs:
    #     movie_from_db = crud.get_movie_by_idx(db, r['movie_idx'])
    #     if movie_from_db:
    #         hydrated_recs.append({
    #             "movie_idx": r['movie_idx'],
    #             "score": r['score'],
    #             "title": movie_from_db.title,
    #             "genres": movie_from_db.genres,
    #             "tmdb_id": movie_from_db.tmdb_id  # <--- THIS FIXES THE 0
    #         })

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
        print(f"[WARN] Failed to log recommendations: {e}")

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
    return get_recommendations(user_idx = request.user_idx,
                               top_k = request.top_k,
                               strategy=request.strategy,
                               fetch_posters=False,
                               db =db
                               )

@app.get("/movie/{movie_idx}", response_model = MovieDetail, tags=["Movies"])
def get_movie(movie_idx :int, db:Session= Depends(get_db)):
    movie = crud.get_movie_by_idx(db, movie_idx)
    if not movie:
        raise HTTPException(status_code=404, 
                            detail =f"Movie {movie_idx} not found"
                            )
    return movie

@app.get("/similar/{movie_idx}", response_model=SimilarMoviesResponse, tags=["Movies"])
def get_similar_movies(movie_idx : int, top_k:int =Query(10, ge=1, le=50), db:Session = Depends(get_db)):
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
        import torch.nn.functional as F

        with torch.no_grad():
            query_emb = recommender.content_emb_norm[movie_idx].unsqueeze(0)  # (1, 768)
            sims = (query_emb @ recommender.content_emb_norm.T).squeeze(0)    # (N,)
            sims[movie_idx] = -1  # exclude self
            top_idxs = sims.topk(top_k).indices.tolist()

        candidates = crud.get_movie_by_idxs(db, top_idxs)
        similar = [{"movie_idx": m.movie_idx, "title": m.title,
                    "genres": m.genres, "tmdb_id": m.tmdb_id,
                    "score": float(sims[m.movie_idx])} 
                   for m in candidates]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")
    

    source_schema = MovieBase(
        movie_idx = source.movie_idx,
        title = source.title,
        genres = source.genres,
        tmdb_id= source.tmdb_id,
        imdb_id = source.imdb_id
    )

    similar_movies = [build_recommended_movie(i + 1, rec) for i, rec in enumerate(similar)]

    return SimilarMoviesResponse(source_movie = source_schema, similar= similar_movies)


@app.get("/search", response_model=SearchResponse, tags=["Movies"])
def search_movie(q: str= Query(..., min_length=1, description="Movie title search query"),
                               limit: int =Query(10, ge=1, le=50),
                               db:Session = Depends(get_db)):
    results = crud.search_movies(db, q, limit)
    movies = [MovieBase(movie_idx=m.movie_idx,
                        title = m.title,
                        genres=m.genres,
                        tmdb_id =m.tmdb_id,
                        imdb_id= m.imdb_id) for m in results]
    
    return SearchResponse(query=q, results=movies, total= len(movies))

@app.post("/feedback", response_model = FeedbackResponse, tags=['Feedback'])
def submit_feedback(request: FeedbackRequest , db:Session = Depends(get_db)):
    crud.log_rating(db=db, user_idx=request.user_idx, movie_idx = request.movie_idx,
                    rating= request.rating, from_rec= request.from_rec)
    return FeedbackResponse(status="logged",
                            user_idx=request.user_idx,
                            movie_idx=request.movie_idx,
                            rating=request.rating)

@app.get("/genres", tags=["Movies"])
def get_genres(db:Session= Depends(get_db)):
    genres = crud.get_all_genres(db)
    return {"genres": genres, "total": len(genres)}

@app.get("/stats", response_model=StatsResponse, tags=['Analytics'])
def get_stats(db:Session=Depends(get_db)):
    stats= crud.get_recommendation_stats(db)
    return StatsResponse(**stats)

@app.get("/popular", tags=['Movies'])
def get_popular_movies(limit:int=Query(20, ge=1, le=20), db:Session = Depends(get_db)):
    popular = crud.get_most_recommended_movies(db, limit)
    return {
        "movies" : popular,
        "total"  : len(popular)
    }

@app.get("/ab-test", tags=["Analytics"])
def get_ab_test_results(db:Session= Depends(get_db)):
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
        
# ── Auto-download large files from HuggingFace ────────────────────────────
# def download_large_files():
#     from huggingface_hub import hf_hub_download
#     from pathlib import Path

#     REPO_ID = "Tush2602/cinemate-v2"

#     files = [
#         "models/two_tower_best.pt",
#         "data/processed/content_embeddings.pt",
#         "data/processed/user_positive_sets.pkl",
#         "data/processed/dataset_constants.pkl",
#         "data/processed/movies_clean.parquet",
#         "data/processed/popularity_lookup.npy",
#         "data/processed/movie_content.csv",
#         "data/processed/encoders/idx2movie.pkl",
#         "data/processed/encoders/idx2user.pkl",
#         "data/processed/encoders/movie2idx.pkl",
#         "data/processed/encoders/user2idx.pkl",
#     ]

#     for file_path in files:
#         local = Path(file_path)
#         if not local.exists():
#             print(f"Downloading {file_path}...")
#             local.parent.mkdir(parents=True, exist_ok=True)
#             hf_hub_download(
#                 repo_id   = REPO_ID,
#                 filename  = file_path,
#                 local_dir = ".",
#                 repo_type = "model"
#             )
#             print(f"Done: {file_path}")
#         else:
#             print(f"Exists: {file_path}")

def download_large_files():
    from huggingface_hub import hf_hub_download, login
    from pathlib import Path
    import os


    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    REPO_ID = "Tush2602/cinemate-v2"
    BASE_DIR = Path("/app")

    files = [
        "models/two_tower_best.pt",
        "data/processed/content_embeddings.pt",
        "data/processed/user_positive_sets.pkl",
        "data/processed/dataset_constants.pkl",
        "data/processed/movies_clean.parquet",
        "data/processed/popularity_lookup.npy",
        "data/processed/movie_content.csv",
        "data/processed/encoders/idx2movie.pkl",
        "data/processed/encoders/idx2user.pkl",
        "data/processed/encoders/movie2idx.pkl",
        "data/processed/encoders/user2idx.pkl",
        "data/processed/plots/01_rating_distribution.png",
        "data/processed/plots/02_ratings_per_user.png",
        "data/processed/plots/03_ratings_per_movie.png",
        "data/processed/plots/04_ratings_over_time.png",
        "data/processed/plots/05_genre_distribution.png",
        "data/processed/plots/06_top_tags.png",
        "data/processed/plots/07_avg_rating_genre.png",
        "data/processed/plots/08_ncf_training_loss.png",
        "data/processed/plots/08_training_loss.png",
        "data/processed/plots/09_model_comparison.png",
        "data/processed/plots/09_ncf_comparison.png",
        "data/processed/plots/10_ab_test_results.png",
        "data/processed/plots/12_popularity_bias.png",
        "data/processed/plots/13_genre_bias.png",
        "data/processed/plots/14_personalisation.png",
        "data/processed/plots/15_genre_diversity.png",
        "data/processed/plots/_model_comparison.png",
        "app/no_photo.png",
    ]

    for file_path in files:
        local = BASE_DIR / file_path
        if not local.exists():
            print(f"Downloading {file_path}...")
            local.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id   = REPO_ID,
                filename  = file_path,
                local_dir = str(BASE_DIR),
                repo_type = "model"
            )
            print(f"✅ Done: {file_path}")
        else:
            print(f"⏭️ Exists: {file_path}")

download_large_files()