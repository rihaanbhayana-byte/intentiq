"""Step 2a: embed every product once and save the vectors."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("data/catalog.parquet")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

emb = model.encode(
    df["search_text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,  # unit length, so dot product = cosine similarity
    show_progress_bar=True,
)
np.save("data/embeddings.npy", emb.astype("float32"))
print("saved data/embeddings.npy:", emb.shape)