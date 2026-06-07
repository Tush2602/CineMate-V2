import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from src.hybrid_model import TwoTowerModel
from src.cf_model import NCFModel
from src.debias import build_popularity_lookup

HEAD_TAIL_PERCENTILE = 60   # movies below 60th pct popularity → tail
HEAD_SLOTS = 7
TAIL_SLOTS = 3
ANCHOR_TOP_N = 50   # top-N scored movies used to build user content profile
TAIL_BLACKLIST = {1650, 17752, 22664, 26211, 19598, 1015, 24792, 24793, 24378, 5915, 23577, 24146, 3962, 6641, 6713, 10963, 21424, 18959, 24791, 25571, 5231, 7682, 8636, 13554, 19116, 19273, 19521, 22297, 2315, 6393, 6727, 6728, 6729, 7679, 7951, 7953, 7954, 7958, 8601, 13375, 13689, 15711, 20237, 21131, 22443, 22452, 22455, 22456, 22457, 25746}

class Recommender:
    def __init__(self):
        self.model              = None
        self.movies_clean       = None
        self.movies_df          = None
        self.idx2movie          = None
        self.movie2idx          = None
        self.user_positive_sets = None
        self.constants          = None
        self.popularity_lookup  = None
        self.content_emb_norm   = None
        self.head_mask          = None
        self.tail_mask          = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, model_type: str = "two_tower"):
        print(f"Loading {model_type} model...")

        # Constants
        with open(os.path.join(PROCESSED_DATA_DIR, "dataset_constants.pkl"), "rb") as f:
            self.constants = pickle.load(f)
        NUM_USERS = self.constants["NUM_USERS"]
        NUM_MOVIES = self.constants["NUM_MOVIES"]

        # Encoders
        with open(os.path.join(PROCESSED_DATA_DIR, "encoders/idx2movie.pkl"), "rb") as f:
            self.idx2movie = pickle.load(f)
        with open(os.path.join(PROCESSED_DATA_DIR, "encoders/movie2idx.pkl"), "rb") as f:
            self.movie2idx = pickle.load(f)

        # User history
        with open(os.path.join(PROCESSED_DATA_DIR, "user_positive_sets.pkl"), "rb") as f:
            self.user_positive_sets = pickle.load(f)

        # Movie metadata — two copies: indexed (fast lookup) + unindexed (genre search)
        movies_path = os.path.join(PROCESSED_DATA_DIR, "movies_clean.parquet")
        self.movies_df = pd.read_parquet(movies_path)
        self.movies_clean = self.movies_df.set_index("movie_idx")

        # Popularity lookup
        lookup_path = os.path.join(PROCESSED_DATA_DIR, "popularity_lookup.npy")
        if os.path.exists(lookup_path):
            self.popularity_lookup = np.load(lookup_path)
        else:
            print("Building popularity lookup (first run)...")
            train = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
            self.popularity_lookup = build_popularity_lookup(train, NUM_MOVIES, save_path=lookup_path)
        print(f"Popularity lookup: {self.popularity_lookup.shape}")

        # Pre-compute head/tail masks (fixed for lifetime of this instance)
        q60 = np.percentile(self.popularity_lookup[self.popularity_lookup > 0], HEAD_TAIL_PERCENTILE)
        self.head_mask = self.popularity_lookup >= q60
        self.tail_mask = (self.popularity_lookup < q60) & (self.popularity_lookup > 0)

        # Model
        embed_path = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")

        if model_type == "two_tower":
            content_emb = torch.load(embed_path, map_location="cpu")
            self.model  = TwoTowerModel(
                num_users = NUM_USERS,
                num_movies = NUM_MOVIES,
                content_embedding_matrix = content_emb,
                embed_dim = 128,
                tower_output_dim = 64,
                dropout = 0.2,
            )
            ckpt_path = os.path.join(MODELS_DIR, "two_tower_best.pt")

        else:
            self.model = NCFModel(
                num_users  = NUM_USERS,
                num_movies = NUM_MOVIES,
                embed_dim  = 64,
                dropout    = 0.2,
            )
            ckpt_path = os.path.join(MODELS_DIR, "ncf_best.pth")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Normalised content embeddings (CPU — only used for cosine similarity)
        raw_emb = torch.load(embed_path, map_location="cpu")
        self.content_emb_norm = F.normalize(raw_emb, dim=1) 

        print(f"Model loaded — epoch {checkpoint['epoch']} | device: {self.device}")

    def recommend(self, user_idx: int, top_k: int = 10, strategy: str = "brute_force") -> list:
        return self._recommend_brute(user_idx, top_k)

    def recommend_tail_only(self, user_idx: int, top_k: int = 7) -> list:
        NUM_MOVIES = self.constants["NUM_MOVIES"]
        seen = self.user_positive_sets.get(user_idx, set())

        # Score all movies
        all_movies = torch.arange(NUM_MOVIES).to(self.device)
        user_tensor = torch.full((NUM_MOVIES,), user_idx, dtype=torch.long).to(self.device)
        with torch.no_grad():
            scores = self.model(user_tensor, all_movies).cpu().numpy()

        masked = scores.copy()
        for m in seen:
            if m < len(masked):
                masked[m] = -np.inf

        # Build user content profile from top-N anchor movies
        anchor_idxs  = np.argsort(masked)[::-1][:ANCHOR_TOP_N].copy()
        anchor_embs  = self.content_emb_norm[torch.tensor(anchor_idxs, dtype=torch.long)] 
        user_profile = F.normalize(anchor_embs.mean(dim=0, keepdim=True), dim=1)

        # Restrict to tail movies not already seen
        tail_indices = np.where(self.tail_mask)[0].copy()
        tail_indices = np.array([m for m in tail_indices if m not in seen and m not in TAIL_BLACKLIST])

        if len(tail_indices) == 0:
            return []

        # Cosine similarity: user profile vs all tail movies
        tail_embs = self.content_emb_norm[torch.tensor(tail_indices, dtype=torch.long)]                                                         # (T, 768)
        sim_scores = (user_profile @ tail_embs.T).squeeze(0).numpy() 
        # Mix with popularity to avoid extreme outliers
        pop_scores = self.popularity_lookup[tail_indices]
        sim_scores = 0.5 * sim_scores + 0.5 * pop_scores 

        # Greedy genre-diverse selection
        sorted_local = np.argsort(sim_scores)[::-1]
        selected = []
        used_genres = set()
        used_movies = set()

        # Pass 1 — genre diverse
        for t_local in sorted_local:
            if len(selected) >= top_k:
                break
            m = int(tail_indices[t_local])
            if m in used_movies:
                continue
            row = self.movies_df[self.movies_df["movie_idx"] == m]
            if len(row) == 0:
                continue
            m_genres = set(str(row.iloc[0]["genres_clean"]).split())
            if not m_genres & used_genres:
                selected.append((m, float(sim_scores[t_local])))
                used_movies.add(m)
                used_genres |= m_genres

        # Pass 2 — fill remaining slots if genre diversity exhausted
        if len(selected) < top_k:
            for t_local in sorted_local:
                if len(selected) >= top_k:
                    break
                m = int(tail_indices[t_local])
                if m not in used_movies:
                    selected.append((m, float(sim_scores[t_local])))
                    used_movies.add(m)

        if not selected:
            return []

        # Normalise scores to [0, 1]
        movie_idxs = [m for m, _ in selected]
        raw_scores = np.array([s for _, s in selected])
        mn, mx = raw_scores.min(), raw_scores.max()
        norm = (raw_scores - mn) / (mx - mn + 1e-8)
        scores_dict = {m: round(float(norm[i]), 4) for i, m in enumerate(movie_idxs)}

        return self._enrich(movie_idxs, scores_dict)

    def _score_all(self, user_idx: int) -> np.ndarray:
        """Score all NUM_MOVIES movies for one user. Returns raw score array."""
        NUM_MOVIES = self.constants["NUM_MOVIES"]
        all_movies = torch.arange(NUM_MOVIES).to(self.device)
        user_tensor = torch.full((NUM_MOVIES,), user_idx, dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model(user_tensor, all_movies).cpu().numpy()

    def _mask_seen(self, scores: np.ndarray, seen: set) -> np.ndarray:
        """Set scores of already-seen movies to -inf."""
        masked = scores.copy()
        for m in seen:
            if m < len(masked):
                masked[m] = -np.inf
        return masked

    def _build_user_profile(self, masked_scores: np.ndarray) -> torch.Tensor:
        """Build (1, 768) L2-normalised user content profile from top-N anchor movies."""
        anchor_idxs = np.argsort(masked_scores)[::-1][:ANCHOR_TOP_N].copy()
        anchor_embs = self.content_emb_norm[
            torch.tensor(anchor_idxs, dtype=torch.long)]
        return F.normalize(anchor_embs.mean(dim=0, keepdim=True), dim=1)

    def _recommend_brute(self, user_idx: int, top_k: int) -> list:
        seen = self.user_positive_sets.get(user_idx, set())
        scores = self._score_all(user_idx)
        masked = self._mask_seen(scores, seen)
        head_scores = masked.copy()
        head_scores[~self.head_mask] = -np.inf
        head_recs = np.argsort(head_scores)[::-1][:HEAD_SLOTS].copy().tolist()

        user_profile = self._build_user_profile(masked)

        tail_indices = np.where(self.tail_mask)[0].copy()
        already_used = seen | set(head_recs)
        tail_indices = np.array([m for m in tail_indices if m not in already_used and m not in TAIL_BLACKLIST])

        tail_recs = []
        used_genres = set()

        if len(tail_indices) > 0:
            tail_embs = self.content_emb_norm[torch.tensor(tail_indices, dtype=torch.long)]
            sim_scores = (user_profile @ tail_embs.T).squeeze(0).numpy()
            sorted_local = np.argsort(sim_scores)[::-1]

            # Pass 1 — genre diverse
            for t_local in sorted_local:
                if len(tail_recs) >= TAIL_SLOTS:
                    break
                m = int(tail_indices[t_local])
                if m in already_used:
                    continue
                row = self.movies_df[self.movies_df["movie_idx"] == m]
                if len(row) == 0:
                    continue
                m_genres = set(str(row.iloc[0]["genres_clean"]).split())
                if not m_genres & used_genres:
                    tail_recs.append(m)
                    already_used.add(m)
                    used_genres |= m_genres

            # Pass 2 — fill remaining if genre diversity exhausted
            if len(tail_recs) < TAIL_SLOTS:
                for t_local in sorted_local:
                    if len(tail_recs) >= TAIL_SLOTS:
                        break
                    m = int(tail_indices[t_local])
                    if m not in already_used:
                        tail_recs.append(m)
                        already_used.add(m)

        final_idxs = head_recs + tail_recs
        raw = np.array([scores[idx] for idx in final_idxs])
        raw_clipped = np.clip(raw, np.percentile(raw, 10), np.percentile(raw, 90))
        mn, mx = raw_clipped.min(), raw_clipped.max()
        norm = (raw_clipped - mn) / (mx - mn + 1e-8)
        scores_dict = {idx: round(float(norm[i]), 4) for i, idx in enumerate(final_idxs)}

        return self._enrich(final_idxs, scores_dict)

    def _enrich(self, movie_idxs: list, scores: dict) -> list:
        results = []
        for idx in movie_idxs:
            idx = int(idx)
            if idx in self.movies_clean.index:
                movie = self.movies_clean.loc[idx]
                results.append({
                    "movie_idx": idx,
                    "title"    : str(movie.get("title", "")),
                    "genres"   : str(movie.get("genres_clean", "")),
                    "tmdb_id"  : int(movie.get("tmdbId", 0)),
                    "score"    : scores.get(idx, 0.0),
                })
            else:
                results.append({
                    "movie_idx": idx,
                    "title" : "Unknown",
                    "genres" : "",
                    "tmdb_id" : 0,
                    "score" : scores.get(idx, 0.0),
                })
        return results

if __name__ == "__main__":
    rec = Recommender()
    rec.load(model_type="two_tower")

    print("\n" + "=" * 60)
    print("BRUTE FORCE — Top 10 for user 0")
    print("=" * 60)
    recs = rec.recommend(user_idx=0, top_k=10)
    for i, r in enumerate(recs, 1):
        print(f"  {i:>2}. {r['title']:<42s} score: {r['score']:.4f}")

    print("\n" + "=" * 60)
    print("TAIL ONLY (Hidden Gems) — user 0")
    print("=" * 60)
    gems = rec.recommend_tail_only(user_idx=0, top_k=7)
    for i, r in enumerate(gems, 1):
        print(f"  {i:>2}. {r['title']:<42s} score: {r['score']:.4f}")

    print("\n" + "=" * 60)
    print("SANITY CHECK — 3 users × top 3")
    print("=" * 60)
    for uid in [0, 100, 500]:
        recs = rec.recommend(user_idx=uid, top_k=3)
        print(f"\n  User {uid}:")
        for r in recs:
            print(f"    {r['title'][:42]:<42s} {r['score']:.4f}")

    print("\n" + "=" * 60)
    print("TAIL SANITY — 3 users × top 3 gems")
    print("=" * 60)
    for uid in [0, 1, 5, 10, 50, 100, 500, 1000]:
        gems = rec.recommend_tail_only(user_idx=uid, top_k=3)
        print(f"\n  User {uid}:")
        for r in gems:
            print(f"{r['title'][:42]:<42s} {r['score']:.4f}")