import numpy as np
import pandas as pd


def build_popularity_lookup(train_df, num_movies, save_path=None):
    # Count ratings per movie
    counts = train_df.groupby('movie_idx').size()

    # Build dense array
    pop_array = np.zeros(num_movies)
    for movie_idx, count in counts.items():
        if movie_idx < num_movies:
            pop_array[movie_idx] = count

    # Log-normalise to [0, 1]
    max_count = pop_array.max()
    log_pop = np.log1p(pop_array)
    log_pop_norm= log_pop / np.log1p(max_count)

    if save_path:
        np.save(save_path, log_pop_norm)
        print(f"Popularity lookup saved: {save_path}")
        print(f"Shape : {log_pop_norm.shape}")
        print(f"Min : {log_pop_norm.min():.4f}")
        print(f"Max : {log_pop_norm.max():.4f}")
        print(f"Mean : {log_pop_norm.mean():.4f}")

    return log_pop_norm


def log_popularity_penalty(scores, popularity_lookup, alpha=0.5):
    return scores - alpha * popularity_lookup


def diverse_rerank(top_candidates, movies_clean, top_k=10, lambda_diversity=0.3, max_per_genre=3):
    if not top_candidates:
        return []

    # Build genre lookup
    movie_genres = {}
    for _, row in movies_clean.iterrows():
        genres = row.get('genres_clean', '')
        if isinstance(genres, str):
            movie_genres[row['movie_idx']] = set(genres.split())
        else:
            movie_genres[row['movie_idx']] = set()

    selected = []
    genre_counts  = {}
    remaining = list(top_candidates)

    while len(selected) < top_k and remaining:
        best_idx = None
        best_score = -np.inf

        for i, (movie_idx, raw_score) in enumerate(remaining):
            genres = movie_genres.get(movie_idx, set())

            # Count how many selected genres this movie shares
            genre_penalty = sum(genre_counts.get(g, 0) for g in genres)

            # Penalise if any genre already at max
            hard_blocked = any(genre_counts.get(g, 0) >= max_per_genre for g in genres)

            if hard_blocked and len(selected) < top_k - 2:
                continue

            # MMR score = relevance - lambda * diversity_penalty
            mmr_score = ((1 - lambda_diversity) * raw_score - lambda_diversity * genre_penalty * 0.1)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            # All blocked — take top remaining
            best_idx = 0

        movie_idx, _ = remaining.pop(best_idx)
        selected.append(movie_idx)

        # Update genre counts
        for g in movie_genres.get(movie_idx, set()):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    return selected


def apply_full_debias(scores, popularity_lookup, movies_clean, seen_movies, top_k=10, alpha=0.3, candidate_pool=100,
                      lambda_diversity=0.3, max_per_genre=3):
    # Step 1 — mask seen
    penalised = scores.copy()
    for m in seen_movies:
        if m < len(penalised):
            penalised[m] = -np.inf

    # Step 2 — apply log-popularity penalty
    valid_mask = penalised > -np.inf
    penalised[valid_mask] = log_popularity_penalty(penalised[valid_mask], popularity_lookup[valid_mask], alpha=alpha)

    # Step 3 — get candidate pool
    top_candidates_idxs = np.argsort(penalised)[::-1][:candidate_pool]

    candidates = [(int(idx), float(penalised[idx])) for idx in top_candidates_idxs if penalised[idx] > -np.inf]

    # Step 4 — diverse re-rank
    final_idxs = diverse_rerank(candidates, movies_clean, top_k = top_k, lambda_diversity= lambda_diversity, max_per_genre  = max_per_genre)

    return final_idxs


def recommend_with_random_tail(scores, popularity_lookup, user_positive_sets, user_idx, num_movies, head_slots=7, tail_slots=3, top_k=10):
    masked = scores.copy()
    seen   = user_positive_sets.get(user_idx, set())
    for m in seen:
        if m < len(masked):
            masked[m] = -np.inf

    head_threshold = np.percentile(popularity_lookup[popularity_lookup > 0], 30)
    head_mask = popularity_lookup >= head_threshold
    head_scores = masked.copy()
    head_scores[~head_mask] = -np.inf
    head_recs = np.argsort(head_scores)[::-1][:head_slots].tolist()

    # Tail — random injection
    tail_mask = ((popularity_lookup < head_threshold) & (popularity_lookup > 0))
    tail_candidates = np.where(tail_mask)[0]
    already_used = seen | set(head_recs)
    tail_pool = [m for m in tail_candidates if m not in already_used]

    np.random.seed(user_idx % 10000)
    n_tail = min(tail_slots, len(tail_pool))
    tail_recs = (np.random.choice(tail_pool, size=n_tail, replace=False).tolist() if tail_pool else [])

    return (head_recs + tail_recs)[:top_k]