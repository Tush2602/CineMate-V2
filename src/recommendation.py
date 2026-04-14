"""
recommendation.py
────────────
Inference module — given a user_idx, return top-K movies.

Two retrieval strategies:
    1. ANN via ChromaDB  → fast, production-ready, O(log n)
    2. Brute force       → exact, used for evaluation

Called by:
    api/main.py     → ANN strategy
    notebooks       → brute force strategy
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR           = os.path.dirname(os.path.dirname(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")

from src.hybrid_model import TwoTowerModel
from src.cf_model import NCFModel
from src.chroma_db import query_similar
from src.debias import build_popularity_lookup


class Recommender:
    """
    Production recommender — loads model once,
    serves recommendations via ANN search.

    Usage:
        rec = Recommender()
        rec.load()
        recs = rec.recommend(user_idx=42, top_k=10)
    """
    def __init__(self):
        self.model = None
        self.movies_clean = None
        self.idx2movie = None
        self.movie2idx = None
        self.user_positive_sets = None
        self.constants = None
        self.popularity_lookup = None
        self.movies_df = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load(self, model_type="two_tower"):
        """
        Load model, encoders, and movie metadata.

        Args:
            model_type : "two_tower" or "ncf"
        """
        print(f"Loading {model_type} model...")

        #Load constants 
        with open(os.path.join(PROCESSED_DATA_DIR, "dataset_constants.pkl"), "rb") as f:
            self.constants = pickle.load(f)

        NUM_USERS  = self.constants['NUM_USERS']
        NUM_MOVIES = self.constants['NUM_MOVIES']

         # Load encoders
        with open(os.path.join(PROCESSED_DATA_DIR, "encoders/idx2movie.pkl"), "rb") as f:
            self.idx2movie = pickle.load(f)

        with open(os.path.join(PROCESSED_DATA_DIR, "encoders/movie2idx.pkl"), "rb") as f:
            self.movie2idx = pickle.load(f)

        with open(os.path.join(PROCESSED_DATA_DIR, "user_positive_sets.pkl"), "rb") as f:
            self.user_positive_sets = pickle.load(f)

        # Load movie metadata
        self.movies_clean = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR,"movies_clean.parquet"))
        self.movies_clean = self.movies_clean.set_index('movie_idx')
        from src.debias import build_popularity_lookup

        lookup_path = os.path.join(
            PROCESSED_DATA_DIR, "popularity_lookup.npy"
        )

        if os.path.exists(lookup_path):
            self.popularity_lookup = np.load(lookup_path)
            print(f"Popularity lookup loaded: "
                f"{self.popularity_lookup.shape}")
        else:
            print("Building popularity lookup...")
            train = pd.read_parquet(
                os.path.join(PROCESSED_DATA_DIR, "train.parquet")
            )
            self.popularity_lookup = build_popularity_lookup(
                train,
                self.constants['NUM_MOVIES'],
                save_path=lookup_path
            )

        # Keep unindexed movies_clean for genre lookup
        self.movies_df = pd.read_parquet(
            os.path.join(PROCESSED_DATA_DIR,
                        "movies_clean.parquet")
    )

        #Load model 
        if model_type=="two_tower":
            embed_path = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")
            content_emb = torch.load(embed_path, map_location='cpu')

            self.model = TwoTowerModel(
                num_users                 = NUM_USERS,
                num_movies                = NUM_MOVIES,
                content_embedding_matrix = content_emb,
                embed_dim                 = 128,
                tower_output_dim          = 64,
                dropout                   = 0.0 
            )

            ckpt_path = os.path.join(MODELS_DIR, "two_tower_best.pt")

        else:
            from src.cf_model import NCFModel

            self.model = NCFModel(
                num_users  = NUM_USERS,
                num_movies = NUM_MOVIES,
                embed_dim  = 128,
                dropout    = 0.0
            )
            ckpt_path = os.path.join(MODELS_DIR, "ncf_best.pth")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Device : {self.device}")

    def recommend(self, user_idx, top_k=10, strategy="brute_force"):
        """
        Generate top-K recommendations for a user.

        Args:
            user_idx : int — encoded user index
            top_k    : number of recommendations
            strategy : "brute_force" or "ann"

        Returns:
            list of dicts with movie_idx, title,
            genres, tmdbId, score
        """
        if strategy =="ann":
            return self._recommend_ann(user_idx, top_k)
        return self._recommend_brute(user_idx, top_k)


    def _recommend_brute(self, user_idx, top_k=10):
        from src.debias import recommend_with_random_tail

        NUM_MOVIES  = self.constants["NUM_MOVIES"]
        all_movies  = torch.arange(NUM_MOVIES).to(self.device)
        user_tensor = torch.full(
            (NUM_MOVIES,), user_idx, dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(
                user_tensor, all_movies
            ).cpu().numpy()

        if self.popularity_lookup is not None:
            # Hybrid — model head + random tail
            final_idxs = recommend_with_random_tail(
                scores             = scores,
                popularity_lookup  = self.popularity_lookup,
                user_positive_sets = self.user_positive_sets,
                user_idx           = user_idx,
                num_movies         = NUM_MOVIES,
                head_slots         = 7,     # ← production setting
                tail_slots         = 3,     # ← production setting
                top_k              = top_k,
            )
            scores_dict = {
                idx: float(scores[idx]) for idx in final_idxs
            }
            return self._enrich(final_idxs, scores_dict)

        # Fallback — no popularity lookup available
        seen = self.user_positive_sets.get(user_idx, set())
        for m in seen:
            if m < len(scores):
                scores[m] = -np.inf
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return self._enrich(
            top_idxs,
            dict(zip(top_idxs.tolist(), scores[top_idxs].tolist()))
        )
    
    def _recommend_ann(self, user_idx, top_k):
        user_tensor = torch.tensor(
            [user_idx], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            user_emb_tensor = self.model.get_user_embeddings( 
                user_tensor
            )
            user_emb_norm = torch.nn.functional.normalize(
                user_emb_tensor, p=2, dim=1
            )
            user_emb = user_emb_norm.cpu().numpy()[0]

        seen = self.user_positive_sets.get(user_idx, set())

        candidates = query_similar(
            user_emb, n_results=top_k * 5
        )

        filtered = [
            c for c in candidates
            if c['movie_idx'] not in seen
        ][:top_k * 2]

        if not filtered:
            return []

        # ── Re-score with actual model ─────────────────────
        movie_idxs = [c['movie_idx'] for c in filtered]
        user_t     = torch.tensor(
            [user_idx] * len(movie_idxs),
            dtype=torch.long
        ).to(self.device)
        movie_t    = torch.tensor(
            movie_idxs, dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(user_t, movie_t).cpu().numpy()

        scores_dict = dict(zip(movie_idxs, scores.tolist()))

        filtered.sort(
            key     = lambda x: scores_dict[x['movie_idx']],
            reverse = True
        )

        top_idxs = [c['movie_idx'] for c in filtered[:top_k]]
        return self._enrich(top_idxs, scores_dict)
    
    def _enrich(self, movie_idxs ,scores):
        """Add metadata to raw movie indices."""
        results = []
        for idx in movie_idxs:
            idx = int(idx)
            row= {}
            if idx in self.movies_clean.index:
                movie = self.movies_clean.loc[idx]
                row = {
                    'movie_idx' : idx,
                    'title'     : str(movie.get('title', '')),
                    'genres'    : str(movie.get('genres_clean', '')),
                    'tmdbId'    : int(movie.get('tmdbId', 0)),
                    'score'     : float(scores[idx])
                }
            else: 
                row = {
                    'movie_idx' : idx,
                    'title'     : 'Unknown',
                    'genres'    : '',
                    'tmdbId'    : 0,
                    'score'     : float(scores[idx])
                }
            results.append(row)
        return results


#Quick test  
if __name__ == "__main__":

    print("--------------BRUTE FORCE Search--------------")
    print("-" * 55)
    print()
    print()
    rec = Recommender()
    rec.load(model_type="two_tower")

    # Test recommendation
    test_user = 0
    recs = rec.recommend(user_idx=test_user, top_k=10)

    print(f"\nTop 10 recommendations for user {test_user}:")
    print("-" * 50)
    for i, r in enumerate(recs, 1):
        print(f"{i:>2}. {r['title']:<40s} "
              f"score: {r['score']:.4f}")
        
    # Test multiple users to ensure variety
    print("\nSanity check — recommendations for 3 users:")
    print("=" * 55)

    for test_user in [0, 100, 500]:
        recs = rec.recommend(user_idx=test_user, top_k=3)
        print(f"\nUser {test_user}:")
        for r in recs:
            print(f"  {r['title'][:40]:<40s} "
                f"score: {r['score']:.4f}")

    # Check cold start user
    print("\nCold start user (user 173133 — near boundary):")
    recs = rec.recommend(user_idx=173133, top_k=3)
    for r in recs:
        print(f"  {r['title'][:40]:<40s} "
            f"score: {r['score']:.4f}")
        
    print()
    print()
    print("-"* 55)
    print("--------------ANN Search--------------")
    print("-" * 55)
    print()
    print()
    rec = Recommender()
    rec.load(model_type="two_tower")

    # Test recommendation
    test_user = 0
    recs = rec.recommend(user_idx=test_user, top_k=10, strategy="ann")

    print(f"\nTop 10 recommendations for user {test_user}:")
    print("-" * 50)
    for i, r in enumerate(recs, 1):
        print(f"{i:>2}. {r['title']:<40s} "
              f"score: {r['score']:.4f}")
        
    # Test multiple users to ensure variety
    print("\nSanity check — recommendations for 3 users:")
    print("=" * 55)

    for test_user in [0, 100, 500]:
        recs = rec.recommend(user_idx=test_user, top_k=3)
        print(f"\nUser {test_user}:")
        for r in recs:
            print(f"  {r['title'][:40]:<40s} "
                f"score: {r['score']:.4f}")

    # Check cold start user
    print("\nCold start user (user 173133 — near boundary):")
    recs = rec.recommend(user_idx=173133, top_k=3)
    for r in recs:
        print(f"  {r['title'][:40]:<40s} "
            f"score: {r['score']:.4f}")
        




