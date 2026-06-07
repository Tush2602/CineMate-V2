"""
train.py
────────
Training script for NCF and Two-Tower models.
Runs from terminal — not a notebook.

Usage:
    python src/train.py --model ncf
    python src/train.py --model two_tower
    python src/train.py --model ncf --epochs 30 --batch_size 2048

This script is for full production training.
Notebooks are for experimentation.
"""

import os
import sys
import pickle
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TwoTowerDataset, NCFDataset
from src.cf_model import NCFModel
from src.hybrid_model import TwoTowerModel

BASE_DIR           = os.path.dirname(os.path.dirname(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,         exist_ok=True)


# ── Loss ──────────────────────────────────────────────────────────────────────

def bpr_loss(pos_score, neg_score,
             pos_movie_idx=None,
             popularity_lookup=None,
             device=None,
             gamma=0.1):
    """
    BPR loss with optional popularity penalty.
    gamma=0.0 → standard BPR
    gamma=0.1 → mild tail boost (recommended)
    """
    if gamma == 0 or popularity_lookup is None:
        return -torch.log(
            torch.sigmoid(pos_score - neg_score) + 1e-8
        ).mean()

    pop_tensor = torch.tensor(
        popularity_lookup[pos_movie_idx.cpu().numpy()],
        dtype=torch.float32
    ).to(device)

    inv_pop = 1.0 - pop_tensor
    return (-torch.log(
        torch.sigmoid(pos_score - neg_score) + 1e-8
    ) * (1.0 + gamma * inv_pop)).mean()


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device,
                    popularity_lookup=None, gamma=0.1):
    model.train()
    total_loss  = 0.0
    total_batch = 0

    for batch in loader:
        user_idx  = batch['user_idx'].to(device,  non_blocking=True)
        pos_movie = batch['pos_movie'].to(device, non_blocking=True)
        neg_movie = batch['neg_movie'].to(device, non_blocking=True)

        pos_score = model(user_idx, pos_movie)
        neg_score = model(user_idx, neg_movie)

        loss = bpr_loss(
            pos_score        = pos_score,
            neg_score        = neg_score,
            pos_movie_idx    = pos_movie,
            popularity_lookup= popularity_lookup,
            device           = device,
            gamma            = gamma
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_batch += 1

    return total_loss / total_batch


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    train = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "train.parquet")
    )
    with open(os.path.join(PROCESSED_DATA_DIR,
              "dataset_constants.pkl"), "rb") as f:
        constants = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_DIR,
              "user_positive_sets.pkl"), "rb") as f:
        user_positive_sets = pickle.load(f)

    print(f"Train ratings : {len(train):,}")
    print(f"NUM_USERS     : {constants['NUM_USERS']:,}")
    print(f"NUM_MOVIES    : {constants['NUM_MOVIES']:,}")
    return train, constants, user_positive_sets


# ── NCF training ──────────────────────────────────────────────────────────────

def train_ncf(args, train, constants, user_positive_sets, device):
    NUM_USERS  = constants['NUM_USERS']
    NUM_MOVIES = constants['NUM_MOVIES']

    dataset = NCFDataset(
        ratings_df         = train,
        user_positive_sets = user_positive_sets,
        num_movies         = NUM_MOVIES,
        positive_threshold = 3.5,
        use_tail_sampling  = True,
        p_tail             = 0.4,
    )
    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        num_workers        = 4,
        pin_memory         = True,
        persistent_workers = True
    )

    model = NCFModel(
        num_users  = NUM_USERS,
        num_movies = NUM_MOVIES,
        embed_dim  = 64,
        dropout    = 0.2
    ).to(device)

    optimizer = Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5,
        factor=0.5, min_lr=1e-5
    )

    # popularity lookup for BPR loss
    lookup_path = os.path.join(MODELS_DIR, "popularity_lookup_ncf.npy")
    if os.path.exists(lookup_path):
        popularity_lookup = np.load(lookup_path)
    else:
        from src.debias import build_popularity_lookup
        popularity_lookup = build_popularity_lookup(
            train, NUM_MOVIES, save_path=lookup_path
        )

    history    = {'train_loss': [], 'epoch_time': []}
    best_loss  = float('inf')
    model_path = os.path.join(MODELS_DIR, "ncf_best.pth")

    print(f"\nTraining NCF | {args.epochs} epochs | device: {device}")
    print("=" * 55)

    for epoch in tqdm(range(1, args.epochs + 1)):
        start    = datetime.now()
        avg_loss = train_one_epoch(
            model, loader, optimizer, device,
            popularity_lookup = popularity_lookup,
            gamma             = 0.1
        )
        elapsed  = (datetime.now() - start).seconds

        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(elapsed)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss'       : best_loss,
                'NUM_USERS'  : NUM_USERS,
                'NUM_MOVIES' : NUM_MOVIES,
            }, model_path)
            marker = " ← best"
        else:
            marker = ""

        print(f"Epoch {epoch:>2}/{args.epochs}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Time: {elapsed}s{marker}")

    pd.DataFrame(history).to_csv(
        os.path.join(MODELS_DIR, "ncf_training_history.csv"),
        index=False
    )
    print(f"\nBest loss : {best_loss:.4f}")
    print(f"Saved to  : {model_path}")
    return model


# ── Two-Tower training ────────────────────────────────────────────────────────

def train_two_tower(args, train, constants, user_positive_sets, device):
    NUM_USERS  = constants['NUM_USERS']
    NUM_MOVIES = constants['NUM_MOVIES']

    embed_path = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")
    if not os.path.exists(embed_path):
        raise FileNotFoundError(
            f"Content embeddings not found at {embed_path}\n"
            f"Run: python src/embeddings.py first"
        )
    content_embeddings = torch.load(embed_path, map_location="cpu")
    print(f"Content embeddings loaded: {content_embeddings.shape}")

    dataset = TwoTowerDataset(
        ratings_df         = train,
        user_positive_sets = user_positive_sets,
        num_movies         = NUM_MOVIES,
        positive_threshold = 3.5,
        use_tail_sampling  = True,
        p_tail             = 0.4,
    )
    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        num_workers        = 4,
        pin_memory         = True,
        persistent_workers = True
    )

    model = TwoTowerModel(
        num_users                = NUM_USERS,
        num_movies               = NUM_MOVIES,
        content_embedding_matrix = content_embeddings,
        embed_dim                = 128,
        tower_output_dim         = 64,
        dropout                  = 0.2
    ).to(device)

    lookup_path = os.path.join(
        MODELS_DIR, "popularity_lookup_two_tower.npy"
    )
    if os.path.exists(lookup_path):
        popularity_lookup = np.load(lookup_path)
    else:
        from src.debias import build_popularity_lookup
        popularity_lookup = build_popularity_lookup(
            train, NUM_MOVIES, save_path=lookup_path
        )

    optimizer = Adam([
        {'params': model.cf_tower.parameters(),      'lr': 1e-3},
        {'params': model.content_tower.parameters(), 'lr': 1e-3},
        {'params': model.fusion.parameters(),        'lr': 1e-3},
    ], weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5,
        factor=0.5, min_lr=1e-5
    )

    history    = {'train_loss': [], 'epoch_time': []}
    best_loss  = float('inf')
    model_path = os.path.join(MODELS_DIR, "two_tower_best.pt")

    print(f"\nTraining Two-Tower | {args.epochs} epochs | device: {device}")
    print("=" * 55)

    for epoch in tqdm(range(1, args.epochs + 1)):
        start    = datetime.now()
        avg_loss = train_one_epoch(
            model, loader, optimizer, device,
            popularity_lookup = popularity_lookup,
            gamma             = 0.1
        )
        elapsed  = (datetime.now() - start).seconds

        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(elapsed)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss'       : best_loss,
                'NUM_USERS'  : NUM_USERS,
                'NUM_MOVIES' : NUM_MOVIES,
            }, model_path)
            marker = " ← best"
        else:
            marker = ""

        print(f"Epoch {epoch:>2}/{args.epochs}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Time: {elapsed}s{marker}")

    pd.DataFrame(history).to_csv(
        os.path.join(MODELS_DIR, "two_tower_training_history.csv"),
        index=False
    )
    print(f"\nBest loss : {best_loss:.4f}")
    print(f"Saved to  : {model_path}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train NCF or Two-Tower model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=['ncf', 'two_tower'],
        help="Which model to train"
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train, constants, user_positive_sets = load_data()

    if args.model == "ncf":
        train_ncf(args, train, constants, user_positive_sets, device)
    else:
        train_two_tower(args, train, constants, user_positive_sets, device)


if __name__ == "__main__":
    main()