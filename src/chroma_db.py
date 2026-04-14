"""
chromadb.py
───────────
Builds and queries a ChromaDB vector store
for fast Approximate Nearest Neighbour (ANN)
movie retrieval.

Why ANN instead of brute force?
    Brute force: score all 27,766 movies per user
    → O(n) per query, slow at serving time
    ANN search : retrieve top-K candidates in O(log n)
    → millisecond retrieval in production

Workflow:
    1. Run build_index() once after Two-Tower training
    2. Use query_similar() at serving time in recommend.py

Usage:
    python src/chromadb.py --build    ← build index once
"""

import os
import sys
import pickle 
import numpy as np 
import pandas as pd 
import torch
import chromadb
from chromadb.config import Settings

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR           = os.path.dirname(os.path.dirname(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
CHROMA_DIR         = os.path.join(DATA_DIR, "chromadb")

os.makedirs(CHROMA_DIR, exist_ok=True)

def get_client():
    """Return persistent ChromaDB client."""

    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
def build_index(model, device, collection_name="movies"):
    print("Building ChromaDB index...")
    print(f"Collection : {collection_name}")
    print(f"Store path : {CHROMA_DIR}")
    print()

    client = get_client()
    try:
        client.delete_collection(collection_name)
        print("Existing collection deleted — rebuilding")
    except Exception:
        pass

    collection = client.create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "ip"}
    )

    model.eval()
    with torch.no_grad():
        # ── Fix: CF movie embeddings only (32-dim) ────────
        num_movies = model.content_embeddings.shape[0]
        movie_idxs = torch.arange(num_movies).to(device)
        movie_vecs = model.cf_tower.get_movie_vector(
            movie_idxs
        )
        # Normalize for cosine/ip consistency
        movie_vecs = torch.nn.functional.normalize(
            movie_vecs, p=2, dim=1
        )
        movie_vecs = movie_vecs.detach().cpu().numpy()

    print(f"Movie embedding shape : {movie_vecs.shape}")
    # Should print: (27766, 32)

    movies_clean = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR,
                     "movies_clean.parquet")
    )
    movie_meta = movies_clean.set_index("movie_idx")

    BATCH = 1000
    n     = movie_vecs.shape[0]

    for i in range(0, n, BATCH):
        batch_idxs = list(range(i, min(i + BATCH, n)))
        batch_vecs = movie_vecs[batch_idxs].tolist()
        batch_ids  = [str(idx) for idx in batch_idxs]

        batch_meta = []
        for idx in batch_idxs:
            if idx in movie_meta.index:
                row = movie_meta.loc[idx]
                batch_meta.append({
                    'movie_idx': int(idx),
                    'title'    : str(row.get('title', '')),
                    'genres'   : str(row.get('genres_clean', '')),
                    'tmdbId'   : int(row.get('tmdbId', 0)),
                })
            else:
                batch_meta.append({'movie_idx': int(idx)})

        collection.upsert(
            ids        = batch_ids,
            embeddings = batch_vecs,
            metadatas  = batch_meta,
        )

        if (i // BATCH) % 5 == 0:
            print(f"  Indexed {min(i+BATCH, n):,}/{n:,} movies")

    print(f"\nIndex built. Total movies indexed: {n:,}")
    return collection


def query_similar(user_embedding, n_results=50,
                  collection_name="movies"):
    """
    Retrieve top-N most similar movies to a user embedding.

    Called by recommend.py at serving time.

    Args:
        user_embedding  : numpy array (tower_output_dim*2,)
                          get this from model.get_user_embedding()
        n_results       : number of candidates to retrieve
        collection_name : ChromaDB collection name

    Returns:
        list of dicts with movie_idx, title, genres, tmdbId
    """
    client     = get_client()
    collection = client.get_collection(collection_name)
    user_vec = torch.from_numpy(user_embedding).float()
    user_vec = torch.nn.functional.normalize(user_vec.unsqueeze(0), p=2, dim=1)
    user_embedding = user_vec.squeeze(0).numpy()

    results = collection.query(
        query_embeddings = [user_embedding.tolist()],
        n_results        = n_results,
        include          = ['metadatas', 'distances']
    )

    candidates = []
    for meta, dist in zip(
        results['metadatas'][0],
        results['distances'][0]
    ):
        score = 1.0 - dist
        candidates.append({
            'movie_idx' : meta.get('movie_idx'),
            'title'     : meta.get('title', ''),
            'genres'    : meta.get('genres', ''),
            'tmdbId'    : meta.get('tmdbId', 0),
            'distance'  : dist,
            'score': score
        })

    return candidates


def get_collection_stats(collection_name="movies"):
    """Return basic stats about the indexed collection."""
    client     = get_client()
    collection = client.get_collection(collection_name)
    count      = collection.count()
    print(f"Collection : {collection_name}")
    print(f"Movies indexed : {count:,}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--stats', action='store_true')
    args = parser.parse_args()

    if args.stats:
        get_collection_stats()

    if args.build:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        from src.hybrid_model import TwoTowerModel

        with open(os.path.join(PROCESSED_DATA_DIR,
                  "dataset_constants.pkl"), "rb") as f:
            constants = pickle.load(f)

        NUM_USERS  = constants['NUM_USERS']
        NUM_MOVIES = constants['NUM_MOVIES']

        embed_path  = os.path.join(
            PROCESSED_DATA_DIR, "content_embeddings.pt"
        )
        content_emb = torch.load(
            embed_path, map_location='cpu'
        )

        model = TwoTowerModel(
            num_users                 = NUM_USERS,
            num_movies                = NUM_MOVIES,
            content_embedding_matrix = content_emb,  # ← fixed name
            embed_dim                 = 128,
            tower_output_dim          = 64,            # ← fixed
            dropout                   = 0.0
        )

        ckpt_path  = os.path.join(
            MODELS_DIR, "two_tower_best.pt"           # ← fixed ext
        )
        checkpoint = torch.load(
            ckpt_path, map_location=device
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        print(f"Model loaded from epoch {checkpoint['epoch']}")

        build_index(model, device)