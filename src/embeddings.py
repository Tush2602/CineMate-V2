import os 
import sys
import pickle
import pandas as pd 
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm 
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

EMBED_PATH = os.path.join(PROCESSED_DATA_DIR, "content_embeddings.pt")
CONTENT_PATH = os.path.join(PROCESSED_DATA_DIR, "content_strings.csv")

def compute_embeddings(batch_size=64, max_length = 128, force=False):
    if os.path.exists(EMBED_PATH) and not force:
        print(f"Embeddings already exist at {EMBED_PATH}")
        print("Pass force=True to recompute.")
        emb = torch.load(EMBED_PATH, map_location='cpu')
        print(f"Shape: {emb.shape}")
        return emb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Computing embeddings on {device}")
    print(f"Batch size : {batch_size}")
    print(f"Max length : {max_length} tokens")
    print()

    #Load content strings
    content_df = pd.read_csv(CONTENT_PATH)
    with open(os.path.join(PROCESSED_DATA_DIR, "dataset_constants.pkl"), "rb") as f:
        constants = pickle.load(f)

    NUM_MOVIES = constants['NUM_MOVIES']

    #Build ordered list  index=movie_idx
    content_lookup = dict(zip(content_df['movie_idx'], content_df['content_string']))
    content_strings = [content_lookup.get(i, "unknown movie") for i in range(NUM_MOVIES)]

    print(f"Total movies to embed: {NUM_MOVIES:,}")
    print(f"Sample string: {content_strings[0][:70]}...")
    print()

    print("Loading DistilBERT models")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    bert_model.eval()

    embeddings= []
    with torch.no_grad():
        for i in tqdm(range(0, NUM_MOVIES, batch_size), desc="Embeddings"):
            batch_texts = content_strings[i: i+batch_size]

            encoded = tokenizer(batch_texts, padding = True, truncation = True, max_length  = max_length, return_tensors = 'pt').to(device)
            output = bert_model(**encoded)

            # CLS token — sentence-level representation
            cls_vecs = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_vecs.cpu())

    content_embeddings = torch.cat(embeddings, dim=0)
    torch.save(content_embeddings, EMBED_PATH)

    size_mb = os.path.getsize(EMBED_PATH) / 1024**2
    print(f"\nEmbeddings saved to : {EMBED_PATH}")
    print(f"Shape : {content_embeddings.shape}")
    print(f"File size : {size_mb:.1f} MB")

    del bert_model
    torch.cuda.empty_cache()

    return content_embeddings

if __name__ == "__main__":
    compute_embeddings(batch_size=64, max_length=128)
    






