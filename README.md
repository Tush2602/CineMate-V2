# 🎬 CineMate V2 — Hybrid Movie Recommender System

> A production-grade movie recommendation engine combining Neural Collaborative Filtering with DistilBERT content embeddings, trained on MovieLens 25M with popularity-aware debiasing.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-ff4b4b)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

[![CineMate V2 Demo](https://img.youtube.com/vi/FUzbwOAlzM0/maxresdefault.jpg)](https://youtu.be/FUzbwOAlzM0)
---

## 📌 Project Overview

CineMate V2 is an end-to-end movie recommendation system built from scratch. It progresses through 5 model iterations — from random baselines to a hybrid Two-Tower neural architecture — and addresses the real-world challenge of **popularity bias** in collaborative filtering systems.

**Key highlights:**
- Trained on **25M ratings** from 173,134 users across 27,766 movies
- **Two-Tower hybrid** — CF tower + DistilBERT content tower fused via element-wise product
- **+79.9% NDCG@10** improvement over SVD baseline
- **17.9% catalogue coverage** via content-similar tail injection (vs 0.63% raw model)
- Full stack: FastAPI REST API + SQLite + Streamlit UI + Docker

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TWO-TOWER MODEL                       │
│                                                         │
│  user_idx ──► CF Tower (128-dim → 64-dim MLP)          │
│                          │                              │
│  movie_idx ─────────────►├──► Fusion MLP ──► score     │
│                          │    [CF ⊕ Content             │
│  DistilBERT ──► Content  │     ⊕ CF ⊙ Content]         │
│  Embeddings    Tower     │                              │
│  (768-dim)  (768→64-dim) │                              │
└─────────────────────────────────────────────────────────┘

Training:  BPR Loss + Popularity Penalty (γ=0.1)
           Tail Negative Sampling (p_tail=0.4)
           ReduceLROnPlateau (patience=5)

Inference: 7/3 Head/Tail Split
           Head: Top-7 by model score (≥60th percentile popularity)
           Tail: Top-3 by cosine similarity to user content profile
```

---

## 📊 Results

| Model | NDCG@10 | Precision@10 | Recall@10 |
|-------|---------|-------------|----------|
| Random | 0.0019 | 0.0018 | 0.0004 |
| Popularity | 0.0845 | 0.0749 | 0.0225 |
| SVD | 0.0669 | 0.0607 | 0.0221 |
| NCF | 0.1252 | 0.1135 | 0.0283 |
| **Two-Tower (ours)** | **0.1203** | **0.1092** | **0.0286** |

**Debiasing Results:**

| Strategy | NDCG@10 | Coverage |
|----------|---------|----------|
| Raw Model | 0.1203 | 0.63% |
| Random Tail Injection | 0.0946 | 5.39% |
| Content-Similar Tail Injection | 0.1004 | **17.92%** |

> Evaluated on 2,000 held-out users. Positive threshold: rating ≥ 3.5.

---

## 🗂 Project Structure

```
CineMate-V2/
├── src/                          # Core ML modules
│   ├── cf_model.py               # CollaborativeTower + NCFModel
│   ├── hybrid_model.py           # TwoTowerModel
│   ├── content_model.py          # ContentTower (DistilBERT projection)
│   ├── dataset.py                # BPR datasets with tail sampling
│   ├── train.py                  # Training script (CLI)
│   ├── recommendation.py         # Inference — brute force + tail injection
│   ├── evaluate.py               # NDCG, Precision, Recall evaluation
│   ├── debias.py                 # Popularity lookup + debiasing utils
│   ├── embeddings.py             # DistilBERT embedding precomputation
│   ├── chroma_db.py              # ChromaDB ANN index (optional)
│   ├── precompute_recs.py        # Placeholder for offline rec precomputation
│   └── tune_debias.py            # Debiasing hyperparameter tuning
│
├── api/                          # FastAPI REST API
│   ├── app.py                    # Endpoints: /recommend, /search, /similar, /stats
│   └── schemas.py                # Pydantic request/response schemas
│
├── app/                          # Streamlit frontend
│   ├── claude_updated.py         # Main UI — Home, Search, About, New Profile
│   └── no_photo.png              # Fallback poster placeholder
│
├── db/                           # Database layer
│   ├── database.py               # SQLAlchemy engine + session
│   ├── models.py                 # ORM: Movie, Recommendation, RatingEvent
│   ├── crud.py                   # DB operations
│   └── init_db.py                # DB initialisation + movie population
│
├── notebook/                     # Jupyter notebooks
│   ├── 01 EDA.ipynb              # Exploratory data analysis
│   ├── 02 preprocessing.ipynb    # Data cleaning + feature engineering
│   ├── 03 baseline_SVD.ipynb     # SVD + Random + Popularity baselines
│   ├── 04 neural cf.ipynb        # NCF — NDCG@10: 0.1252
│   ├── 05 two tower.ipynb        # Two-Tower — NDCG@10: 0.1203
│   ├── 06 ab_test.ipynb          # A/B test — debiased vs raw
│   ├── 07_business_impact.ipynb  # Business metrics analysis
│   ├── 08_bias_fairness.ipynb    # Popularity + genre bias analysis
│   └── 09_recommendation_analysis.ipynb  # Coverage + diversity analysis
│
├── data/
│   ├── processed/
│   │   ├── plots/                # Training + evaluation visualisations (in repo)
│   │   ├── encoders/             # user2idx, movie2idx mappings (in repo)
│   │   ├── movies_clean.parquet  # Movie metadata (in repo)
│   │   ├── dataset_constants.pkl # NUM_USERS, NUM_MOVIES (in repo)
│   │   ├── popularity_lookup.npy # Normalised popularity scores (in repo)
│   │   ├── content_embeddings.pt # DistilBERT embeddings (NOT in repo — 212MB)
│   │   ├── train.parquet         # Training ratings (NOT in repo — large)
│   │   └── user_positive_sets.pkl# User history (NOT in repo — large)
│   └── raw/                      # MovieLens 25M raw files (NOT in repo)
│
├── models/
│   ├── two_tower_best.pt         # Trained weights (NOT in repo — 700MB)
│   ├── two_tower_results.json    # Evaluation metrics (in repo)
│   ├── ncf_results.json          # NCF metrics (in repo)
│   ├── svd_baseline_results.json # SVD metrics (in repo)
│   └── *_training_history.csv    # Loss curves (in repo)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt              # GPU setup (local development)
├── requirements_docker.txt       # CPU setup (Docker / deployment)
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA GPU recommended (CPU works but ~30-60s per request)
- TMDB API key — free at [themoviedb.org](https://www.themoviedb.org/settings/api)

### 1. Clone & Setup

```bash
git clone https://github.com/Tush2602/CineMate-V2.git
cd CineMate-V2
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env — add your TMDB_API_KEY
```

### 3. Download Data & Models

> **Note:** Large files are not included in this repo due to size constraints.
>
> - Download MovieLens 25M from [grouplens.org](https://grouplens.org/datasets/movielens/25m/)
> - Run preprocessing notebooks in order: `01 EDA → 02 preprocessing → 03 baseline → 04 NCF → 05 Two-Tower`
> - Or download preprocessed files + trained model from [HuggingFace](https://huggingface.co/Tush2602/cinemate-v2)

### 4. Initialise Database

```bash
python db/init_db.py
```

### 5. Run

**Terminal 1 — API:**
```bash
uvicorn api.app:app --reload --port 8000
```

**Terminal 2 — Streamlit:**
```bash
streamlit run app/claude_updated.py
```

Open `http://localhost:8501`

---

## 🐳 Docker

```bash
docker-compose up
```

API: `http://localhost:8000`
UI: `http://localhost:8501`

> Docker uses CPU-only PyTorch. Inference will be slower (~30-60s) compared to GPU setup.
> Ensure `data/` and `models/` directories are populated before running.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/recommend/{user_idx}` | Top-K personalised recommendations |
| GET | `/recommend/gems/{user_idx}` | Tail-debiased hidden gems |
| GET | `/similar/{movie_idx}` | Content-similar movies |
| GET | `/search` | Full-text movie search |
| GET | `/popular` | Most rated movies |
| GET | `/stats` | Recommendation analytics |
| POST | `/feedback` | Log user rating |
| GET | `/docs` | Interactive Swagger UI |

---

## 🧠 Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MovieLens 25M |
| Train ratings | 25,868,311 |
| Test ratings | 795,563 |
| Users | 173,134 |
| Movies | 27,766 |
| Epochs | 50 |
| Batch size | 2,048 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Loss | BPR + Popularity Penalty (γ=0.1) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Train time | ~4.6 hours (NVIDIA GPU) |
| Best BPR Loss | 0.0605 (epoch 48) |

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

**1. Popularity Bias**
Collaborative filtering inherently amplifies popularity bias — movies with more ratings develop stronger representations. Shawshank Redemption, The Matrix, and Pulp Fiction dominate Top Picks for most users due to their high rating volume in the training data.

**2. Tail Movie Representations**
Movies below the 70th percentile in rating count have sparse training signals, resulting in weak embeddings. The "Beyond the Obvious" debiasing section uses content-similar tail injection as a post-hoc fix — effective at increasing coverage (0.63% → 17.9%) but tail recommendations lack the relevance precision of head recommendations.

**3. Cold Start**
New users receive genre-based recommendations via content similarity — not model-personalised. This is intentionally honest cold-start handling.

**4. Static Model**
User ratings submitted via the app are logged to SQLite for analytics but do not retrain the model. Offline retraining is required to incorporate new signals.

**5. Inference Speed**
Brute-force scoring across 27,766 movies takes ~400ms on GPU, ~30-60s on CPU. Production systems would use ANN indexing (ChromaDB integration included but optional).

### Future Work

- **Exposure-aware BPR loss** — weight loss by inverse propensity score to reduce popularity bias at training time
- **Sequence modelling** — SASRec or BERT4Rec for richer user representations using watch history
- **Online learning** — incorporate real-time rating feedback into model updates
- **ANN deployment** — ChromaDB index for sub-10ms inference at scale
- **A/B testing framework** — compare debiased vs raw recommendations on real engagement metrics

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch 2.7 |
| NLP Embeddings | HuggingFace Transformers (DistilBERT) |
| API | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy + Alembic |
| Frontend | Streamlit |
| Vector Store | ChromaDB (optional ANN) |
| Containerisation | Docker + Docker Compose |
| Data | MovieLens 25M (GroupLens) |
| Poster API | TMDB API (free tier) |

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01 EDA` | Rating distributions, genre analysis, long-tail visualisation |
| `02 preprocessing` | Data cleaning, train/test split, feature engineering |
| `03 baseline_SVD` | SVD + Random + Popularity baselines |
| `04 neural cf` | Neural Collaborative Filtering — NDCG@10: 0.1252 |
| `05 two tower` | Two-Tower hybrid — training, evaluation, model comparison |
| `06 ab_test` | A/B test — debiased vs raw recommendations |
| `07 business_impact` | Business metrics — CTR simulation, coverage analysis |
| `08 bias_fairness` | Popularity + genre bias quantification |
| `09 recommendation_analysis` | Coverage, diversity, personalisation analysis |

---

## 👤 Author

**Tushar Joshi**
B.Tech Electrical Engineering, PEC Chandigarh
Aspiring ML / Data Scientist with strong interest in production ML systems

[![GitHub](https://img.shields.io/badge/GitHub-Tush2602-black?logo=github)](https://github.com/Tush2602)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tushar_Joshi-blue?logo=linkedin)](https://www.linkedin.com/in/tushar-joshi-47a5a9311)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## ⭐ Support

If this project helped you or inspired you, consider giving it a ⭐

It helps others discover production-ready ML system design patterns.

---

> *"The best recommendation is one the user didn't know they wanted."*