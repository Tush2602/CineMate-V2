from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

#Movies Schema
class MovieBase(BaseModel):
    movie_idx : int
    title : str
    genres : str
    tmdb_id : int = 0
    imdb_id : int = 0

    class Config:
        from_attributes = True

class MovieDetail(MovieBase):
    genome_tags : str = ""
    content_string : str = ""
    created_at : Optional[datetime] = None

    class Config:
        from_attributes = True

#Recommendations Schema
class RecommendedMovie(BaseModel):
    rank : int
    movie_idx : int
    title : str
    genres : str
    tmdb_id : int = 0
    score : float = 0.0
    poster_url: Optional[str] = None

class RecommendationRequest(BaseModel):
    user_idx : int = Field(..., ge=0, description="Encoded user index")
    top_k : int = Field(10, ge=1, le=50, description="Number of recommendations")
    strategy : str = Field("ann", description="ann or brute force")
    model_type : str = Field("two_tower", description="two_tower or ncf")

class RecommendationResponse(BaseModel):
    user_idx : int
    recommendations : List[RecommendedMovie]
    model_version : str
    strategy : str
    response_time_ms: float
    is_cold_start : bool = False


#Search Schema
class SearchSchema(BaseModel):
    query :str = Field(..., min_length=1, max_length=200)
    limit : int = Field(10, ge=1, le=50)

class SearchResponse(BaseModel):
    query : str
    results : List[MovieBase]
    total : int

#Similiar movies Schema
class SimilarMoviesResponse(BaseModel):
    source_movie : MovieBase
    similar : List[RecommendedMovie]


#Feedback Schema
class FeedbackRequest(BaseModel):
    user_idx : int
    movie_idx : int
    rating : float = Field(..., ge=0.5, le=5.0)
    from_rec : bool  = True


class FeedbackResponse(BaseModel):
    status : str = "logged"
    user_idx : int
    movie_idx : int
    rating : float

#health Schema 
class HealthResponse(BaseModel):
    status : str
    model_loaded : bool
    db_connected : bool
    num_users : int
    num_movies : int
    version : str = "1.0.0"

#Stats Schema 
class StatsResponse(BaseModel):
    total_served : int
    unique_users : int
    unique_movies : int
    catalogue_coverage : float
    avg_score : float
    model_breakdown : dict
    


