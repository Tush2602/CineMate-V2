import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db.database import engine, SessionLocal, init_db
from db.models   import Movie

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


def populate_movies(db):
    existing= db.query(Movie).count()
    if existing > 0:
        print(f"Movie table already has {existing:,} rows.")
        print("Skipping population, Delete db file to repopulate.")
        return

    print("Loading movies_clean.parquet ............")
    movies_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "movies_clean.parquet"))

    print(f"Populating movies with {len(movies_df)} movies.")

    movies_obj = []
    for _ , row in movies_df.iterrows():
        movies_obj.append(Movie(
            movie_idx = int(row['movie_idx']),
            movie_id = int(row['movieId']),
            title = str(row['title']),
            genres = str(row.get('genres_clean', '')),
            genome_tags = str(row.get('genome_tag_string', '')),
            raw_tags = str(row.get('raw_tag_string', '')),
            content_string = str(row.get('content_string', '')),
            tmdb_id = int(row.get('tmdbId', 0)),
            imdb_id = int(row.get('imdbId', 0)),
        ))

        # Bulk insert every 1000 rows
        if len(movies_obj) >= 1000:
            db.bulk_save_objects(movies_obj)
            db.commit()
            movies_obj = []

    # Insert remaining
    if movies_obj:
        db.bulk_save_objects(movies_obj)
        db.commit()

    total = db.query(Movie).count()
    print(f"Movie table populated: {total} movies")


def verify_db(db):
    """Quick verification all tables are accessible."""
    from db.models import (Movie, UserSession,Recommendation, RatingEvent)

    print()
    print("=" * 45)
    print("DATABASE VERIFICATION")
    print("=" * 45)
    print(f"  Movies : {db.query(Movie).count()}")
    print(f"  UserSessions : {db.query(UserSession).count()}")
    print(f"  Recommendations : {db.query(Recommendation).count()}")
    print(f"  RatingEvents : {db.query(RatingEvent).count()}")
    print()

    # Sample movie check
    sample = db.query(Movie).first()
    if sample:
        print(f"  Sample movie : {sample.title}")
        print(f"  Genres : {sample.genres}")
        print(f"  tmdb_id : {sample.tmdb_id}")
    print()
    print("Database ready for API.")


def main():
    print("Initialising Cinemate database...")
    print()
    # Create tables
    init_db()

    # Populate movies
    db = SessionLocal()
    try:
        populate_movies(db)
        verify_db(db)
    finally:
        db.close()

if __name__ == "__main__":
    main()

        
