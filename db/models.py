from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.sql import func
from db.database import Base

class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    movie_idx = Column(Integer, unique=True, nullable=False, index=True)
    movie_id = Column(Integer, nullable=False)
    title = Column(String(500), nullable=False)
    genres= Column(String(500), default="")
    genome_tags= Column(Text, default="")
    raw_tags = Column(Text, default="")
    content_string = Column(Text, default="")
    tmdb_id = Column(Integer, default=0)
    imdb_id = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

    def to_dict(self):
        return {
            "movie_idx":self.movie_idx,
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "genome_tags": self.genome_tags,
            "tmdb_id": self.tmdb_id,
            "imdb_id": self.imdb_id,
        }
    
    def __repr__(self):
        return f"<Movie {self.movie_idx} : {self.title}>"
    

class UserSession(Base):

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, server_default=func.now())
    last_seen = Column(DateTime, onupdate=func.now(), server_default=func.now())
    n_requests = Column(Integer, default=1)
    is_cold_start = Column(Boolean, default=False)

class Recommendation(Base):
    __tablename__ = "recommendations"

    id= Column(Integer, primary_key=True, index=True)
    user_idx = Column(Integer, unique=True, nullable= False, index=True)
    movie_idx= Column(Integer, nullable=False)
    rank = Column(Integer, nullable=False)
    score = Column(Float, default=0.0)
    model_version = Column(String(50), default="two_tower")
    strategy      = Column(String(50), default="ann")
    served_at     = Column(DateTime, server_default=func.now())

    # Index for fast user lookup
    __tableargs__  = (Index("idx_rec_user_served", "user_idx", "served_at"))
    
    def __repr__(self):
        return (f"<Recommendation user = {self.user_idx}"
                f"movie = {self.movie_idx} rank = {self.rank}")

class RatingEvent(Base):
    __tablename__ = "rating_events"
    id          = Column(Integer, primary_key=True, index=True)
    user_idx    = Column(Integer, nullable=False, index=True)
    movie_idx   = Column(Integer, nullable=False)
    rating      = Column(Float, nullable=False)
    from_rec    = Column(Boolean, default=False)
    rated_at    = Column(DateTime, server_default=func.now())

    __table_args__ = (Index('idx_rating_user_movie','user_idx', 'movie_idx'),)
    def __repr__(self):
        return (f"<RatingEvent user={self.user_idx} "
                f"movie={self.movie_idx} "
                f"rating={self.rating}>")

    

