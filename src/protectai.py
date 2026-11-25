"""Off-the-shelf ProtectAI DeBERTa-v3 prompt-injection detector on CPU; saves scores per evaluation set."""
import json, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import CACHE, DATA, RESULTS, group_heldout_split, load_corpus
from run_external import EXT

NAME = "protectai/deberta-v3-base-prompt-injection-v2"
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()
inj = model.config.label2id.get("INJECTION", 1)

@torch.no_grad()
def score(texts, bs=16, max_len=512):
    order = np.argsort([len(t) for t in texts])  # length-sorted batches -> far less padding
    out = np.zeros(len(texts))
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        out[idx] = torch.softmax(model(**b).logits, -1)[:, inj].numpy()
    return out

if __name__ == "__main__":
    path = RESULTS / "protectai_scores.npz"
    done = dict(np.load(path)) if path.exists() else {}
    sets = {k: pd.read_csv(DATA / "external" / f"{k}.csv").text.astype(str).tolist() for k in EXT}
    df = load_corpus(); _, _, te = group_heldout_split(df)
    sets["our_grouped_test"] = df.text.iloc[te].tolist()
    v2 = DATA / "v2_clean.csv"
    if v2.exists():
        sets["v2_hard"] = pd.read_csv(v2).text.astype(str).tolist()
    for k, texts in sets.items():
        if k in done:
            continue
        t0 = time.time()
        done[k] = score(texts)
        np.savez(path, **done)
        print(k, len(texts), f"{time.time()-t0:.0f}s", flush=True)
