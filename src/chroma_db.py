import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import chromadb
from chromadb.config import Settings

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")

COLLECTION_NAME = "movies_cf"
EMBED_DIM = 64          
BATCH_SIZE = 1000

os.makedirs(CHROMA_DIR, exist_ok=True)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path = CHROMA_DIR,
        settings = Settings(anonymized_telemetry=False)
    )

def build_index(model, device, collection_name: str = COLLECTION_NAME):
    print("=" * 55)
    print("Building ChromaDB index")
    print(f"Collection : {collection_name}")
    print(f"Store path : {CHROMA_DIR}")
    print(f"Embed dim  : {EMBED_DIM}")
    print("=" * 55)

    client = get_client()
    try:
        client.delete_collection(collection_name)
        print("Existing collection deleted — rebuilding from scratch")
    except Exception:
        pass

    collection = client.create_collection(name= collection_name,
                                        metadata = {"hnsw:space": "ip"})

    # Extract movie embeddings from CF tower
    model.eval()
    with torch.no_grad():
        num_movies = model.content_embeddings.shape[0]
        movie_idxs = torch.arange(num_movies).to(device)
        movie_vecs = model.cf_tower.get_movie_vector(movie_idxs)   # (N, 128)
        movie_vecs = F.normalize(movie_vecs, p=2, dim=1)           # (N, 128)
        movie_vecs = movie_vecs.cpu().numpy()

    actual_dim = movie_vecs.shape[1]
    print(f"Movie embedding shape : {movie_vecs.shape}")

    # Load movie metadata for ChromaDB payloads
    movies_df  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "movies_clean.parquet"))
    movie_meta = movies_df.set_index("movie_idx")

    # Batch upsert
    n_indexed = 0
    for i in range(0, num_movies, BATCH_SIZE):
        batch_idxs = list(range(i, min(i + BATCH_SIZE, num_movies)))
        batch_vecs = movie_vecs[batch_idxs].tolist()
        batch_ids  = [str(idx) for idx in batch_idxs]

        batch_meta = []
        for idx in batch_idxs:
            if idx in movie_meta.index:
                row = movie_meta.loc[idx]
                batch_meta.append({
                    "movie_idx": int(idx),
                    "title" : str(row.get("title","")),
                    "genres" : str(row.get("genres_clean", "")),
                    "tmdb_id" : int(row.get("tmdbId",0)),
                })
            else:
                batch_meta.append({"movie_idx": int(idx)})

        collection.upsert(
            ids = batch_ids,
            embeddings = batch_vecs,
            metadatas = batch_meta,
        )
        n_indexed += len(batch_idxs)
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"Indexed {n_indexed:,} / {num_movies:,} movies")

    print(f"\nIndex built successfully. Total indexed: {n_indexed:,}")
    print(f"Embed dim in index: {actual_dim}")
    return collection



def query_similar(user_embedding: np.ndarray, n_results: int = 50, collection_name: str = COLLECTION_NAME) -> list:
    client     = get_client()
    collection = client.get_collection(collection_name)

    # Ensure user embedding is L2-normalised
    vec = torch.from_numpy(user_embedding).float().unsqueeze(0)   # (1, D)
    vec = F.normalize(vec, p=2, dim=1).squeeze(0).numpy()         # (D,)

    results = collection.query(query_embeddings = [vec.tolist()], n_results = n_results, include = ["metadatas", "distances"],)

    candidates = []
    for meta, dist in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        # inner product distance → similarity score
        score = float(1.0 - dist)
        candidates.append({
            "movie_idx": int(meta.get("movie_idx", 0)),
            "title" : str(meta.get("title","")),
            "genres" : str(meta.get("genres","")),
            "tmdb_id" : int(meta.get("tmdb_id",0)),
            "distance" : float(dist),
            "score" : score,
        })

    return candidates


def get_collection_stats(collection_name: str = COLLECTION_NAME) -> int:
    """Print and return number of indexed movies."""
    client = get_client()
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        print(f"Collection : {collection_name}")
        print(f"Movies indexed : {count:,}")
        print(f"Store path : {CHROMA_DIR}")
        return count
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        print("Run: python src/chroma_db.py --build")
        return 0


def index_exists(collection_name: str = COLLECTION_NAME) -> bool:
    """Check if ChromaDB index has been built."""
    try:
        client = get_client()
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or inspect ChromaDB ANN index for Cinemate")
    parser.add_argument("--build", action="store_true", help="Build ANN index from trained Two-Tower model")
    parser.add_argument("--stats", action="store_true", help="Print index statistics")
    args = parser.parse_args()

    if args.stats:
        get_collection_stats()

    if args.build:
        from src.hybrid_model import TwoTowerModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        with open(os.path.join(PROCESSED_DATA_DIR,"dataset_constants.pkl"), "rb") as f:
            constants = pickle.load(f)

        NUM_USERS  = constants["NUM_USERS"]
        NUM_MOVIES = constants["NUM_MOVIES"]

        embed_path  = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")
        content_emb = torch.load(embed_path, map_location="cpu")

        model = TwoTowerModel(
            num_users = NUM_USERS,
            num_movies = NUM_MOVIES,
            content_embedding_matrix = content_emb,
            embed_dim = 128,
            tower_output_dim = 64,
            dropout = 0.0,
        )

        ckpt_path  = os.path.join(MODELS_DIR, "two_tower_best.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        print(f"Model loaded from epoch {checkpoint['epoch']}")

        build_index(model, device)
        print("Done. Run --stats to verify.")

    if not args.build and not args.stats:
        parser.print_help()