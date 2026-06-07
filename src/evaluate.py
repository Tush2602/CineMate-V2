import numpy as np 
import pandas as pd 
import torch
from tqdm import tqdm

def ndcg_at_k(recommended, relevant_set, k=10):
    dcg = sum(1.0/np.log2(i+2) for i, m in enumerate(recommended[:k]) if m in relevant_set)
    ideal_hits= min(k, len(relevant_set))
    idcg = sum(1.0/np.log2(i+2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(recommended , relevant_set, k=10):
    hits = len(set(recommended[:k]) & relevant_set)
    return hits/k

def recall_at_k(recommended, relevant_set, k=10):
    if not relevant_set:
        return 0.0
    
    hits = len(set(recommended[:k]) & relevant_set)
    return hits/len(relevant_set)

def evaluate_model(model, eval_users, test_relevant, user_positive_sets, num_movies, device, k=10, batch_size= 512):
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
        f'NDCG@{k}' : float(np.mean(ndcg_scores)),
        f'Precision@{k}': float(np.mean(precision_scores)),
        f'Recall@{k}' : float(np.mean(recall_scores)), 'n_users_eval' : len(ndcg_scores)
    }

def evaluate_all(models_dict, eval_users, test_relevant, user_positive_sets,num_movies, device, k=10):
    rows= []
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_model(
            model, eval_users, test_relevant,
            user_positive_sets, num_movies, device, k
        )
        rows.append({
            'Model' : name,
            f'NDCG@{k}' : results[f'NDCG@{k}'],
            f'Precision@{k}': results[f'Precision@{k}'],
            f'Recall@{k}' : results[f'Recall@{k}'],
            'Users Evaluated': results['n_users_eval']
        })

    return pd.DataFrame(rows).set_index('Model')

