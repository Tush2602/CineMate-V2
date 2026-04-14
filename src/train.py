"""
train.py
────────
Training script for NCF and Two-Tower models.
Runs from terminal — not a notebook.

Usage:
    python src/train.py --model ncf
    python src/train.py --model two_tower
    python src/train.py --model two_tower --epochs 20

This script is what you run for full production training.
Notebooks are for experimentation.
"""

import os
import json 
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

#local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import TwoTowerDataset, NCFDataset
from src.cf_model import NCFModel
from src.hybrid_model import TwoTowerModel
from src.evaluate import evaluate_model

#Getting the project root path 
BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
MODELS_DIR = os.path.join(os.path.join(BASE_DIR, "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# In src/train.py — replace bpr_loss

def bpr_loss(pos_score, neg_score,
                              pos_movie_idx,
                              neg_movie_idx,
                              popularity_lookup,
                              device,
                              gamma=0.1):
    """
    BPR loss with popularity penalty.

    Standard BPR + penalty term that encourages
    the model to score unpopular relevant movies
    relatively higher.

    gamma=0.0 → standard BPR (no change)
    gamma=0.1 → mild tail boost
    gamma=0.3 → strong tail boost
    """
    # Standard BPR loss
    base_loss = -torch.log(
        torch.sigmoid(pos_score - neg_score) + 1e-8
    ).mean()

    if gamma == 0 or popularity_lookup is None:
        return base_loss

    # Popularity penalty:
    # scale loss by inverse popularity of positive item
    # → model penalised more for failing on rare items
    pop_tensor = torch.tensor(
        popularity_lookup[
            pos_movie_idx.cpu().numpy()
        ],
        dtype=torch.float32
    ).to(device)

    # inv_pop in [0, 1] — 1 = most unpopular
    inv_pop    = 1.0 - pop_tensor

    # Weight loss higher for unpopular items
    weighted_loss = -torch.log(
        torch.sigmoid(pos_score - neg_score) + 1e-8
    ) * (1.0 + gamma * inv_pop)

    return weighted_loss.mean()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss= 0.0
    total_batch = 0

    for batch in loader:
        user_idx  = batch['user_idx'].to(device, non_blocking=True)
        pos_movie = batch['pos_movie'].to(device, non_blocking=True)
        neg_movie = batch['neg_movie'].to(device, non_blocking=True)

        #forward pass 
        pos_score = model(user_idx, pos_movie)
        neg_score = model(user_idx, neg_movie)

        loss= bpr_loss(pos_score=pos_score, neg_score=neg_score)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batch+=1

    return total_loss/total_batch

def load_data():
    print("Loading data...")
    train = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"))
    with open(os.path.join(PROCESSED_DATA_DIR,
              "dataset_constants.pkl"), "rb") as f:
        constants = pickle.load(f)

    with open(os.path.join(PROCESSED_DATA_DIR,
              "user_positive_sets.pkl"), "rb") as f:
        user_positive_sets = pickle.load(f)

    return train, test, constants, user_positive_sets


def train_ncf(args, train, constants, user_positive_sets, device):
    NUM_USERS = constants['NUM_USERS']
    NUM_MOVIES = constants['NUM_MOVIES']

    dataset = NCFDataset(ratings_df=train,
                         user_postive_sets=user_positive_sets,
                         num_movies=NUM_MOVIES,)
    
    loader= DataLoader(dataset,
                       batch_size=args.batch_size,
                       shuffle=True,
                       num_workers=6,
                       pin_memory=True,
                       persistent_workers=True)
    
    model = NCFModel(num_users=NUM_USERS,
                     num_movies=NUM_MOVIES,
                     embed_dim=128,
                     dropout=0.2).to(device)
    
    optimizer =Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode="min",
                                  patience=5, 
                                  factor=0.5,
                                  min_lr=1e-5)
    
    history ={'train_loss': [], "epoch_time": []}
    best_loss= float("inf")
    model_path = os.path.join(MODELS_DIR, "ncf_best.pth")

    print(f"\nTraining NCF | {args.epochs} epochs | "
          f"device: {device}")
    print("=" * 55)

    for epoch in tqdm(range(1, args.epochs+1)):
        start =datetime.now()
        avg_loss= train_one_epoch(model=model,
                                  loader=loader,
                                  optimizer=optimizer,
                                  device=device)
        elapsed = (datetime.now() - start).seconds

        history['train_loss'].append(avg_loss)
        history['epoch_time'].append(elapsed)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epochs": epoch,
                "model_state": model.state_dict(),
                "optim_state":optimizer.state_dict(),
                "loss":best_loss,
                "NUM_USERS": NUM_USERS,
                "NUM_MOVIES": NUM_MOVIES
            }, model_path)
            marker =" ← best"
        else: 
            marker = ""

            print(f"Epoch {epoch:>2}/{args.epochs}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Time: {elapsed}s{marker}")

    #save history 
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(MODELS_DIR, "ncf_training_history.csv"),index=False)
    print(f"\nBest loss : {best_loss:.4f}")
    print(f"Saved to  : {model_path}")
    return model  

def train_two_tower(args, train, constant, user_postive_sets, device):
    NUM_USERS = constant['NUM_USERS']
    NUM_MOVIES= constant['NUM_MOVIES']

    #Loading pre computed embeddings embed_paths 
    embed_path = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")
    if not os.path.exists(embed_path):
        raise FileNotFoundError(
            f"Content embeddings not found at {embed_path}.\n"
            f"Run : python src/embeddings.py before running it"
        )
    content_embeddings = torch.load(embed_path,map_location="cpu")
    print(f"Content embeddings loaded : {content_embeddings.shape}")

    dataset = TwoTowerDataset(
        ratings_df=train,
        user_positive_sets=user_postive_sets,
        num_movies=NUM_MOVIES,
        postive_thresholds=3.5
    )

    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=6,
                        pin_memory=True,
                        persistent_workers=True)
    
    model = TwoTowerModel(num_users=NUM_USERS,
                          num_movies=NUM_MOVIES,
                          content_embedding_matrix=content_embeddings,
                          embed_dim=128, 
                          tower_output_dim=64,
                          dropout=0.2
                          )

    WEIGHT_DECAY  = 1e-4 
    optimizer = Adam([
        {'params': model.cf_tower.parameters(), 'lr': 1e-3},
        {'params': model.content_tower.parameters(), 'lr': 1e-3},
        {'params': model.fusion.parameters(), 'lr': 1e-3}      
    ], weight_decay=WEIGHT_DECAY)


    # Reduce LR when loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = 'min',
        patience = 5,
        factor   = 0.5,
        min_lr=1e-5
    )

    history = {"train_loss": [], "epoch_time":[]}
    best_loss = float('inf')
    model_path = os.path.join(MODELS_DIR, "two_tower_best.pth")
    print(f"\n Training Two-Tower model | {args.epochs} epochs | device : {device}")
    print("=" * 55)

    for epoch in tqdm(range(1, args.epochs +1)):
        start= datetime.now()
        avg_loss= train_one_epoch(model, loader, optimizer, device)
        elapsed = int(datetime.now() - start)

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
        
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(MODELS_DIR, "two_tower_training_history.csv"), index=False)

        print(f"\n Best loss : {best_loss}")
        print(f"Saved to  : {model_path}")
        return model
    
def main():
    parser = argparse.ArgumentParser(description="Train NCF Model or Two tower model.......")
    parser.add_argument("--model", type=str, required=True, choices=['NCF', 'Two Tower'], help="Which model to train")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    args= parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train, test, constants, user_positive_sets = load_data()

    if args.model =="NCF":
        train_ncf(args, train, constants, user_positive_sets, device)

    else: 
        train_two_tower(args, train, constants, user_positive_sets, device)

    
if __name__ == "__main__":
    main()
    




