"""Encode a corpus with frozen sentence encoders and cache the embeddings.

usage: python encode.py <corpus: raw|clean> <encoder key>...
"""
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

from common import CACHE, load_clean, load_raw

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "distilbert": "distilbert-base-uncased",  # frozen, mean-pooled (not fine-tuned)
}


def get_encoder(key, device):
    name = ENCODERS[key]
    if key == "distilbert":
        word = models.Transformer(name, max_seq_length=256)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[word, pool], device=device)
    m = SentenceTransformer(name, device=device)
    m.max_seq_length = 256
    return m


if __name__ == "__main__":
    corpus = sys.argv[1]
    df = load_raw() if corpus == "raw" else load_clean()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for key in sys.argv[2:]:
        t0 = time.time()
        enc = get_encoder(key, device)
        X = enc.encode(df.text.tolist(), batch_size=64, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=False)
        np.save(CACHE / f"{corpus}_{key}.npy", X.astype(np.float32))
        print(f"{corpus}/{key}: {X.shape} in {time.time()-t0:.0f}s", flush=True)
