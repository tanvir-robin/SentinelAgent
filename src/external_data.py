"""Build external evaluation sets from public datasets (label 1 = injection/jailbreak, 0 = benign).

Every external sample whose MiniLM cosine similarity to any training-corpus prompt is >= 0.95
is removed, so external results are not inflated by seed overlap.
"""
import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

from common import CACHE, DATA

OUT = DATA / "external"
OUT.mkdir(exist_ok=True)


def build():
    sets = {}
    d = load_dataset("deepset/prompt-injections")
    d = concatenate_datasets([d["train"], d["test"]])
    sets["deepset"] = pd.DataFrame({"text": d["text"], "label": d["label"]})
    d = load_dataset("jackhhao/jailbreak-classification")
    d = concatenate_datasets([d["train"], d["test"]])
    sets["jackhhao_jailbreak"] = pd.DataFrame({"text": d["prompt"], "label": [int(t == "jailbreak") for t in d["type"]]})
    d = load_dataset("xTRam1/safe-guard-prompt-injection")["test"]
    sets["safeguard"] = pd.DataFrame({"text": d["text"], "label": d["label"]})
    d = load_dataset("Lakera/gandalf_ignore_instructions")
    d = concatenate_datasets([d["validation"], d["test"]])
    sets["gandalf"] = pd.DataFrame({"text": d["text"], "label": 1})
    d = load_dataset("leolee99/NotInject")
    d = concatenate_datasets([d[k] for k in d])
    sets["notinject"] = pd.DataFrame({"text": d["prompt"], "label": 0})
    d = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")["train"].to_pandas()
    d = d.sample(2000, random_state=0)
    sets["spml"] = pd.DataFrame({"text": d["User Prompt"].values, "label": d["Prompt injection"].astype(int).values})
    return sets


if __name__ == "__main__":
    sets = build()
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    corpus = torch.nn.functional.normalize(torch.tensor(np.load(CACHE / "raw_minilm.npy")), dim=1)
    for name, df in sets.items():
        df = df.dropna().drop_duplicates("text").reset_index(drop=True)
        df = df[df.text.str.strip().str.len() > 0].reset_index(drop=True)
        E = torch.nn.functional.normalize(torch.tensor(enc.encode(df.text.tolist(), batch_size=64)), dim=1)
        mx = torch.cat([(E[i:i + 512] @ corpus.T).max(1).values for i in range(0, len(E), 512)]).numpy()
        removed = int((mx >= 0.95).sum())
        df = df[mx < 0.95].reset_index(drop=True)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: n={len(df)} benign={int((df.label==0).sum())} malicious={int((df.label==1).sum())} overlap_removed={removed}")
