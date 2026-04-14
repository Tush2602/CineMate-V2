"""
Schema.py
──────────
Pydantic models for request validation
and response serialisation.

Every API endpoint uses these — never returns
raw SQLAlchemy objects directly.

Why Pydantic?
    - Automatic validation of request data
    - Clean JSON serialisation of responses
    - Auto-generated API docs at /docs
    - Type safety throughout the API layer
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

#Movies Schema
class MovieBase(BaseModel):
    """Minimal movie info — used in recommendation lists."""
    movie_idx : int
    title : str
    genres : str
    tmdb_id : int = 0
    imdb_id : int = 0

    class Config:
        from_attributes = True

class MovieDetail(MovieBase):
    """Full movie detail — used in /movie/{movie_idx}."""
    genome_tags    : str = ""
    content_string : str = ""
    created_at     : Optional[datetime] = None

    class Config:
        from_attributes = True

#Recommendations Schema
class RecommendedMovie(BaseModel):
    """
    Single movie in a recommendation list.
    Includes score and rank for frontend display.
    """
    rank : int
    movie_idx : int
    title : str
    genres : str
    tmdb_id : int = 0
    score : float = 0.0
    poster_url: Optional[str] = None

class RecommendationRequest(BaseModel):
    """Request body for POST /recommend."""
    user_idx : int = Field(..., ge=0, description="Encoded user index")
    top_k : int = Field(10, ge=1, le=50, description="Number of recommendations")
    strategy : str = Field("ann", description="ann or brute force")
    model_type : str = Field("two_tower", description="two_tower or ncf")

class RecommendationResponse(BaseModel):
    """Response from GET /recommend/{user_idx}."""
    user_idx        : int
    recommendations : List[RecommendedMovie]
    model_version   : str
    strategy        : str
    response_time_ms: float
    is_cold_start   : bool = False


#Search Schema
class SearchSchema(BaseModel):
    """Request body for movie search."""
    query :str = Field(..., min_length=1, max_length=200)
    limit : int = Field(10, ge=1, le=50)

class SearchResponse(BaseModel):
    """Response from GET /search."""
    query   : str
    results : List[MovieBase]
    total   : int

#Similiar movies Schema
class SimilarMoviesResponse(BaseModel):
    """Response from GET /similar/{movie_idx}."""
    source_movie : MovieBase
    similar      : List[RecommendedMovie]


#Feedback Schema
class FeedbackRequest(BaseModel):
    """
    User feedback on a recommendation.
    Logged to RatingEvent table for A/B analysis.
    """
    user_idx  : int
    movie_idx : int
    rating    : float = Field(..., ge=0.5, le=5.0)
    from_rec  : bool  = True


class FeedbackResponse(BaseModel):
    """Response after logging feedback."""
    status    : str = "logged"
    user_idx  : int
    movie_idx : int
    rating    : float

#health Schema 
class HealthResponse(BaseModel):
    """Response from GET /health."""
    status       : str
    model_loaded : bool
    db_connected : bool
    num_users    : int
    num_movies   : int
    version      : str = "1.0.0"

#Stats Schema 
class StatsResponse(BaseModel):
    """Response from GET /stats — for dashboard."""
    total_recommendations : int
    unique_users          : int
    unique_movies         : int
    catalogue_coverage    : float
    avg_score             : float
    model_breakdown       : dict

    


