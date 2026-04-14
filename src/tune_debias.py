# scripts/tune_debias.py
# Find alpha that maximises coverage without killing NDCG

import sys
import os
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.recommendation import Recommender
from src.debias import apply_full_debias

rec = Recommender()
rec.load(model_type="two_tower")

test  = pd.read_parquet("data/processed/test.parquet")
test_relevant = (
    test[test['rating'] >= 3.5]
    .groupby('user_idx')['movie_idx']
    .apply(set).to_dict()
)
eval_users = list(test_relevant.keys())[:500]

def ndcg_at_k(recommended, relevant_set, k=10):
    dcg  = sum(1.0/np.log2(i+2)
               for i, m in enumerate(recommended[:k])
               if m in relevant_set)
    idcg = sum(1.0/np.log2(i+2)
               for i in range(min(k, len(relevant_set))))
    return dcg/idcg if idcg > 0 else 0.0

NUM_MOVIES  = rec.constants['NUM_MOVIES']
all_movies  = torch.arange(NUM_MOVIES).to(rec.device)

# Test different alpha values
results = []
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    for lam in [0.0, 0.2, 0.3]:
        all_recs    = {}
        ndcg_scores = []

        with torch.no_grad():
            for user_idx in tqdm(
                eval_users,
                desc=f"alpha={alpha} lam={lam}",
                leave=False
            ):
                seen    = rec.user_positive_sets.get(
                    user_idx, set()
                )
                u_t     = torch.full(
                    (NUM_MOVIES,), user_idx,
                    dtype=torch.long
                ).to(rec.device)
                scores  = rec.model(
                    u_t, all_movies
                ).cpu().numpy()

                final = apply_full_debias(
                    scores,
                    rec.popularity_lookup,
                    rec.movies_df,
                    seen,
                    top_k           = 10,
                    alpha           = alpha,
                    candidate_pool  = 100,
                    lambda_diversity= lam,
                    max_per_genre   = 3
                )
                all_recs[user_idx] = final

                relevant = test_relevant.get(
                    user_idx, set()
                )
                ndcg_scores.append(
                    ndcg_at_k(final, relevant)
                )

        unique_movies = len(set(
            m for recs in all_recs.values()
            for m in recs
        ))
        coverage = unique_movies / NUM_MOVIES

        results.append({
            'alpha'      : alpha,
            'lambda'     : lam,
            'ndcg'       : np.mean(ndcg_scores),
            'coverage'   : coverage,
            'unique_movies': unique_movies,
        })

        print(f"alpha={alpha:.1f} λ={lam:.1f} | "
              f"NDCG={np.mean(ndcg_scores):.4f} | "
              f"Coverage={coverage:.2%} | "
              f"Unique={unique_movies:,}")

# Find sweet spot
df = pd.DataFrame(results)
print("\nBest configurations:")
print(df[df['coverage'] >= 0.10].sort_values(
    'ndcg', ascending=False
).head(5).to_string())

df.to_csv("models/debias_tuning.csv", index=False)