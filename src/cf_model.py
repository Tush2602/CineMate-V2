import torch
import torch.nn as nn


class CollaborativeTower(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=128, output_dim=64, dropout=0.2):
        super().__init__()
        self.user_embedding  = nn.Embedding(num_users,  embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
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

        nn.init.normal_(self.user_embedding.weight,  std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        return self.mlp(torch.cat([u, m], dim=-1))

    def get_user_vector(self, user_idx):
        return self.user_embedding(user_idx)

    def get_movie_vector(self, movie_idx):
        return self.movie_embedding(movie_idx)


class NCFModel(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim=64, dropout=0.2):
        super().__init__()
        self.user_embedding  = nn.Embedding(num_users,  embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
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

        nn.init.normal_(self.user_embedding.weight,  std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        return self.mlp(torch.cat([u, m], dim=-1)).squeeze(-1)