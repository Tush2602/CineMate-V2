import numpy as np 
import pandas as pd 


import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, ratings_df, user_postive_sets, num_movies, positive_threshold =3.5, neg_sample_tries=10, use_tail_sampling=True):
        self.num_movies = num_movies
        self.user_positive_sets= user_postive_sets
        self.neg_sample_tries= neg_sample_tries

        positive_df = ratings_df[ratings_df['rating']>=positive_threshold].copy()

        self.users = positive_df['user_idx'].values.astype(np.int32)
        self.movies = positive_df['movie_idx'].values.astype(np.int32)
        self.tail_movies = None
        if use_tail_sampling:
            movie_counts = ratings_df.groupby('movie_idx').size()
            threshold = movie_counts.quantile(0.70)
            tail_idxs = movie_counts[movie_counts <= threshold].index.values
            self.tail_movies = tail_idxs
            print(f"Tail movies (≤70th pct popularity) : {len(self.tail_movies)}")


    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_idx = int(self.users[idx])
        pos_movie = int(self.movies[idx])
        seen_movies = self.user_positive_sets.get(user_idx, set())
        P_TAIL = 0.4  

        if (np.random.random() < P_TAIL and self.tail_movies is not None):
            neg_movie = int(np.random.choice(self.tail_movies))
        else:
            neg_movie = np.random.randint(0, self.num_movies)

        for _ in range(self.neg_sample_tries):
            if neg_movie not in seen_movies:
                break
            if (np.random.random() < P_TAIL and self.tail_movies is not None):
                neg_movie = int(np.random.choice(self.tail_movies))
            else:
                neg_movie = np.random.randint(0, self.num_movies)

        return {
            'user_idx' : torch.tensor(user_idx, dtype=torch.long),
            'pos_movie' : torch.tensor(pos_movie, dtype=torch.long),
            'neg_movie' : torch.tensor(neg_movie, dtype=torch.long)}

class TwoTowerDataset(Dataset):
    def __init__(self, ratings_df, user_positive_sets, num_movies, postive_thresholds = 3.5, neg_sample_tries= 10):
        self.num_movies= num_movies
        self.neg_sample_tries= neg_sample_tries
        self.user_positive_sets = user_positive_sets

        positive_df = ratings_df[ratings_df['rating']>=postive_thresholds].copy()

        self.users= positive_df['user_idx'].values.astype(np.int32)
        self.movies= positive_df['movie_idx'].values.astype(np.int32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_idx= int(self.users[idx])
        pos_movies= int(self.movies[idx])
        seen_movies= self.user_positive_sets.get(user_idx, set())

        neg_movies= np.random.randint(0, self.num_movies)
        for _ in range(self.neg_sample_tries):
            if neg_movies not in seen_movies:
                break
            neg_movies = np.random.randint(0, self.num_movies)

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "pos_movie": torch.tensor(pos_movies, dtype=torch.long),
            "neg_movie": torch.tensor(neg_movies, dtype=torch.long)
        }
    

class InferenceDataset(Dataset):
    def __init__(self, user_idx, num_movies, seen_movies=None):
        self.num_movies= num_movies
        self.user_idx= user_idx
        self.seen_movies = seen_movies
        self.all_movies=list(range(num_movies))

    def __len__(self):
        return self.num_movies
    
    def __getitem__(self, idx):
        movie_idx= self.all_movies[idx]
        return {
            'user_idx' : torch.tensor(self.user_idx, dtype=torch.long),
            'movie_idx' : torch.tensor(movie_idx,dtype=torch.long),
        }