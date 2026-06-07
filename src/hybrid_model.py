import torch
import torch.nn as nn

from src.cf_model import CollaborativeTower
from src.content_model import ContentTower


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, content_embedding_matrix, embed_dim=128, tower_output_dim=64, dropout=0.2):
        super().__init__()

        self.cf_tower = CollaborativeTower(
            num_users  = num_users,
            num_movies = num_movies,
            embed_dim  = embed_dim,
            output_dim = tower_output_dim,
            dropout    = dropout
        )
        self.content_tower = ContentTower(
            bert_dim   = 768,
            output_dim = tower_output_dim,
            dropout    = dropout
        )

        self.norm = nn.LayerNorm(tower_output_dim)

        self.register_buffer("content_embeddings", content_embedding_matrix.float())

        self.fusion = nn.Sequential(
            nn.Linear(tower_output_dim * 3, 256),
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
        cf_out        = self.norm(self.cf_tower(user_idx, movie_idx))
        movie_content = self.content_embeddings[movie_idx]
        content_out   = self.norm(self.content_tower(movie_content))
        interaction   = cf_out * content_out
        combined      = torch.cat([cf_out, content_out, interaction], dim=-1)
        return self.fusion(combined).squeeze(-1)

    def get_user_embeddings(self, user_idx):
        return self.cf_tower.get_user_vector(user_idx)

    def get_movie_embeddings(self, movie_idx):
        return self.cf_tower.get_movie_vector(movie_idx)