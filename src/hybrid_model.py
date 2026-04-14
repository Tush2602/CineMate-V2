"""
hybrid_model.py
───────────────
Two-Tower hybrid model combining collaborative
and content signals.

Architecture:
    user_idx  → CollaborativeTower → 32-dim ↘
    movie_idx ↗                               → Fusion → score (0-1)
    movie_idx → ContentTower      → 32-dim ↗

The content embeddings are pre-computed by embeddings.py
and stored as a non-trainable buffer inside the model.
This means they move to GPU automatically with model.to(device)
but are never updated during backpropagation.
"""

import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.cf_model import CollaborativeTower
from src.content_model import ContentTower

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies,content_embedding_matrix, embed_dim=128, tower_output_dim =32, dropout=0.2):
        super().__init__()
        self.cf_tower = CollaborativeTower(num_users=num_users,
                                           num_movies=num_movies,
                                           embed_dim=embed_dim,
                                           output_dim=tower_output_dim,
                                           dropout=dropout)
        self.content_tower = ContentTower(bert_dim=768,
                                          output_dim=tower_output_dim,
                                          dropout=dropout)
        
        self.norm = nn.LayerNorm(tower_output_dim)
        
        #Non trainable buffer - moves to device , not updated
        self.register_buffer("content_embeddings", content_embedding_matrix.float())

        #fusion 
        fusion_input =tower_output_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
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
            nn.Linear(32, 1) 
        )

    def forward(self, user_idx, movie_idx):
        cf_out = self.norm(self.cf_tower(user_idx, movie_idx))

        movie_content = self.content_embeddings[movie_idx]
        content_out = self.norm(self.content_tower(movie_content))

        interaction = cf_out * content_out
        combined= torch.cat([cf_out, content_out, interaction], dim=-1)
        score = self.fusion(combined).squeeze(-1)

        return score
    
    def get_user_embeddings(self, user_idx):
        return self.cf_tower.get_user_vector(user_idx)
    
    def get_all_movie_embeddings(self, ):
        num_movies = self.content_embeddings.shape[0]
        movies_idx= torch.arange(num_movies).to(self.content_embeddings.device)

        #cf movie vectors
        cf_movie_vecs= self.cf_tower.get_movie_vector(movies_idx)
        content_vecs= self.content_tower(self.content_embeddings)
        return cf_movie_vecs
    



        
