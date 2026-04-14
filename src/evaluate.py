"""
evaluate.py
───────────
Evaluation metrics for ranking quality.

All metrics use @K cutoff (default K=10).

Functions:
    ndcg_at_k       — primary metric, ranking quality
    precision_at_k  — hit fraction in top-K
    recall_at_k     — relevant item coverage in top-K
    evaluate_model  — full evaluation loop for any model
    evaluate_all    — evaluate multiple models, return DataFrame
"""

import numpy as np 
import pandas as pd 
import torch
from tqdm import tqdm

def ndcg_at_k(recommended, relevant_set, k=10):
    """
    Normalised Discounted Cumulative Gain at K.

    Rewards relevant items appearing HIGH in ranked list.
    Position 1 > position 2 > ... > position K.

    Args:
        recommended  : list of movie_idx, ordered best first
        relevant_set : set of movie_idx user actually liked
        k            : rank cutoff

    Returns:
        float in [0.0, 1.0]
    """
    dcg = sum(1.0/np.log2(i+2) for i, m in enumerate(recommended[:k]) if m in relevant_set)
    ideal_hits= min(k, len(relevant_set))
    idcg = sum(1.0/np.log2(i+2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(recommended , relevant_set, k=10):
    """
    Fraction of top-K recommendations that are relevant.

    Args:
        recommended  : list of movie_idx
        relevant_set : set of relevant movie_idx
        k            : rank cutoff

    Returns:
        float in [0.0, 1.0]
    """
    hits =len(set[:recommended] & relevant_set)
    return hits/k

def recall_at_k(recommended, relevant_set, k=10):
    """
    Fraction of relevant items that appear in top-K.

    Args:
        recommended  : list of movie_idx
        relevant_set : set of relevant movie_idx
        k            : rank cutoff

    Returns:
        float in [0.0, 1.0]
    """
    if not relevant_set:
        return 0.0
    
    hits = len(set(recommended[:k] & relevant_set))
    return hits/len(relevant_set)

def evaluate_model(model, eval_users, test_relevant, user_positive_sets, num_movies, device, k=10, batch_size= 512):
    """
    Full evaluation loop for any scoring model.

    For each user:
        1. Score all num_movies movies
        2. Mask out already-seen movies
        3. Take top-K by score
        4. Compute NDCG, Precision, Recall vs test_relevant

    Args:
        model              : PyTorch model with forward(user, movie)
        eval_users         : list of user_idx to evaluate
        test_relevant      : dict {user_idx: set of relevant movie_idx}
        user_positive_sets : dict {user_idx: set of seen movie_idx}
        num_movies         : total number of movies
        device             : torch.device
        k                  : rank cutoff
        batch_size         : movies scored per forward pass

    Returns:
        dict with NDCG@k, Precision@k, Recall@k, n_users_eval
    """
    model.eval()
    all_movies= torch.arange(num_movies).to(device)

    ndcg_scores = []
    precision_scores = []
    recall_scores= []

    with torch.no_grad():
        for user_idx in tqdm(eval_users, desc="Evaluating"):
            relevant =test_relevant.get(user_idx , set())
            if not relevant:
                continue
            seen = user_positive_sets.get(user_idx, set())

            #scores all movies in the batch
            all_scores = []
            for i in range(0, num_movies, batch_size):
                batch_movies= all_movies[i:i+batch_size]
                batch_users = torch.full_like(batch_movies, user_idx)
                scores = model(batch_users, batch_movies)
                all_scores.append(scores.cpu())

            scores_np = torch.cat(all_scores).numpy()

            #Mask seen movies
            for m in seen:
                if m < len(scores_np):
                    scores_np[m]= -np.inf

            top_k = np.argsort(scores_np)[::-1][:k].tolist()

            ndcg_scores.append(ndcg_at_k(top_k, relevant_set=relevant, k=k))
            precision_scores.append(precision_at_k(top_k, relevant, k))
            recall_scores.append(recall_at_k(top_k, relevant, k))


    return {
        f'NDCG@{k}'     : float(np.mean(ndcg_scores)),
        f'Precision@{k}': float(np.mean(precision_scores)),
        f'Recall@{k}'   : float(np.mean(recall_scores)),
        'n_users_eval'  : len(ndcg_scores)
    }

def evaluate_all(models_dict, eval_users, test_relevant, user_positive_sets,num_movies, device, k=10):
    """
    Evaluate multiple models and return comparison DataFrame.

    Args:
        models_dict : dict {model_name: model}
        others      : same as evaluate_model

    Returns:
        pd.DataFrame with one row per model
    """

    rows= []
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_model(
            model, eval_users, test_relevant,
            user_positive_sets, num_movies, device, k
        )
        rows.append({
            'Model'         : name,
            f'NDCG@{k}'     : results[f'NDCG@{k}'],
            f'Precision@{k}': results[f'Precision@{k}'],
            f'Recall@{k}'   : results[f'Recall@{k}'],
            'Users Evaluated': results['n_users_eval']
        })

    return pd.DataFrame(rows).set_index('Model')

