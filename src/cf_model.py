"""
cf_model.py
───────────
Collaborative Filtering Tower.

Takes user_idx and movie_idx as input.
Outputs a 32-dim interaction vector.

Used standalone in NCFModel and as one
tower inside TwoTowerModel.
"""


import numpy as np 
import pandas as pd
import os 

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CollaborativeTower(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=128, output_dim=32, dropout=0.2):
        super().__init__()

        self.user_embedding= nn.Embedding(num_users, embedding_dim=embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

        #initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x= torch.cat([u, m], dim=-1)

        return self.mlp(x)
    
    def get_user_vector(self, user_idx):
        """Return raw user embedding — used by recommend.py."""
        return self.user_embedding(user_idx)
    
    def get_movie_vector(self, movie_idx):
        return self.movie_embedding(movie_idx)
    



class NCFModel(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=128, dropout=0.2):
        super().__init__()

        self.user_embedding= nn.Embedding(num_users, embedding_dim=embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        #init weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x= torch.cat([u, m], dim=-1)

        return self.mlp(x).squeeze(-1)
    
        






        