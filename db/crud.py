from typing import List, Optional
from datetime import datetime
from time import time

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from db.models import Movie, UserSession, Recommendation, RatingEvent

#Movies queries

def get_movie_by_idx(db: Session, movie_idx : int) -> Optional[Movie]:
    return db.query(Movie).filter(Movie.movie_idx == movie_idx).first()

def get_movie_by_idxs(db:Session, movie_idxs:List[int]) -> List[Movie]:
    movies= db.query(Movie).filter(Movie.movie_idx.in_(movie_idxs)).all()
    idx_to_movies = {m.movie_idx : m for m in movies}
    return [ idx_to_movies[idx] for idx in movie_idxs if idx in idx_to_movies ]

def search_movies(db:Session, query:str, limit: int=20)-> List[Movie]:
    return db.query(Movie).filter(Movie.title.ilike(f"%{query}%")).limit(limit).all()

def get_movies_by_genres(db:Session, genre:str, limit : int= 50) ->List[Movie]:
    return db.query(Movie).filter(Movie.genres.ilike(f"%{genre}%")).limit(limit).all()

def get_all_genres(db:Session) -> List[str]:
    movies =db.query(Movie.genres).all()
    genres = set()
    for genre_str in movies:
        if genre_str:
            for g in genre_str.split():
                if g:
                    genres.add(g.strip())
    return sorted(genres)


def get_total_movies(db:Session) -> int:
    return db.query(func.count(Movie.id)).scalar()

# USER SESSION QUERIES
def get_or_create_users(db:Session, user_idx:int, is_cold_start : bool= False,) -> UserSession:
    user=  db.query(UserSession).filter(UserSession.user_idx==user_idx).first()
    if user:
        user.n_requests+=1
        user_last_seen = datetime.now()
        db.commit()
        db.refresh(user)
        return user
    
    user = UserSession(user_idx = user_idx,
                       is_cold_start= is_cold_start,
                       n_requests =1)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user(db:Session, user_idx:int) -> Optional[UserSession]:
    return db.query(UserSession).filter(UserSession.user_idx==user_idx).first()

def get_total_users(db:Session)->int:
    return db.query(func.count(UserSession.id)).scalar()


## Recommendations
def log_recommendations(db:Session, user_idx:int, recommendations : List[dict], model_version : str = "two_tower", strategy:str ="ann")-> None:
    rec_objects = [Recommendation(user_idx=user_idx,
                                  movie_idx = rec['movie_idx'],
                                  rank = i + 1,
                                  score = rec.get("score", 0.0),
                                  model_version = model_version,
                                  strategy = strategy
                                ) 
                                for i , rec in enumerate(recommendations)
                ]
    db.bulk_save_objects(rec_objects)
    db.commit()

def get_user_recommendation_history(db:Session, user_idx : int, limit:int= 50) -> List[Recommendation]:
    return db.query(Recommendation).filter(Recommendation.user_idx==user_idx).order_by(desc(Recommendation.served_at)).limit(limit).all()

def get_total_recommendations(db: Session) -> int:
    return db.query(func.count(Recommendation.id)).scalar()

def get_recommendation_stats(db:Session)-> dict:
    total = db.query(func.count(Recommendation.id)).scalar() or 0
    unique_users= db.query(func.count(func.distinct(Recommendation.user_idx))).scalar() or 0
    unique_movies= db.query(func.count(func.distinct(Recommendation.movie_idx))).scalar() or 0
    avg_score = db.query(func.avg(Recommendation.score)).scalar() or 0.0
    
    # Model version breakdown
    model_counts = db.query(Recommendation.model_version,func.count(Recommendation.id)
                            ).group_by(Recommendation.model_version).all()
    
    return {
        "total_served": total,
        "unique_users": unique_users,
        "unique_movies": unique_movies,
        "avg_score": round(float(avg_score), 4),
        "catalogue_coverage": round(unique_movies / max(get_total_movies(db), 1), 4),
        "model_breakdown": {name: count for name, count in model_counts}
    }

def get_most_recommended_movies(db: Session,limit: int = 20) -> List[dict]:
    results = db.query(Recommendation.movie_idx, func.count(Recommendation.id).label('rec_count')).group_by(
                Recommendation.movie_idx).order_by(desc('rec_count')).limit(limit).all()

    return [
        {"movie_idx": r.movie_idx, "rec_count": r.rec_count} for r in results
    ]

def log_rating(db:Session, user_idx: int, movie_idx : int, rating:float, from_rec: bool=False) -> RatingEvent:
    event = RatingEvent(
        user_idx  = user_idx,
        movie_idx = movie_idx,
        rating    = rating,
        from_rec  = from_rec
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_ab_test_stats(db:Session) -> dict:
    # Treatment group — rated recommended movies
    treatment = db.query(func.count(RatingEvent.id).label("n"), func.avg(RatingEvent.rating).label("avg_rating")).filter(
                                RatingEvent.from_rec==True).first()
    #Control group - Organic ratings
    control = db.query(func.count(RatingEvent.id).label('n'), func.avg(RatingEvent.rating).label('avg_rating')).filter(
                                RatingEvent.from_rec == False).first()
    
    return {
        "treatment": {
            "n"          : treatment.n or 0,
            "avg_rating" : round(float(treatment.avg_rating or 0), 4),
            "label"      : "Recommended movies"
        },
        "control": {
            "n"          : control.n or 0,
            "avg_rating" : round(float(control.avg_rating or 0), 4),
            "label"      : "Organic ratings"
        }
    }

