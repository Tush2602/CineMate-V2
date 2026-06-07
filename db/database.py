import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#Getting the project root path 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR   = os.path.join(DATA_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "cinemate.db")
DB_URL = f"sqlite:///{DB_PATH}"


engine = create_engine(DB_URL,
                      connect_args={"check_same_thread": False},
                      echo=False
                      )

#Session
SessionLocal =sessionmaker(bind=engine,
                           autocommit=False,
                           autoflush=False)

#base -> all orm models inherit from this
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Import models so Base knows about them
    from db import models
    Base.metadata.create_all(bind=engine)
    print(f"Database initialised at : {DB_PATH}")

if __name__ == "__main__":
    init_db()

    
