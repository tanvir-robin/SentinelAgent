"""Train on the SentinelAgent corpus, test zero-shot on public external datasets.

Also runs an existing off-the-shelf detector (ProtectAI DeBERTa-v3 prompt-injection v2) for reference.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import CACHE, DATA, RESULTS, group_heldout_split, load_corpus
from encode import get_encoder
from models import predict, train_head

EXT = ["deepset", "jackhhao_jailbreak", "safeguard", "spml", "gandalf", "notinject"]


def ext_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    r = {"n": int(len(y)), "acc": float((yhat == y).mean() * 100)}
    if (y == 1).any():
        r["recall"] = float(yhat[y == 1].mean() * 100)
    if (y == 0).any():
        r["fpr"] = float(yhat[y == 0].mean() * 100)
    if len(np.unique(y)) == 2:
        tp = ((yhat == 1) & (y == 1)).sum(); fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
        r["f1"] = float(2 * tp / max(2 * tp + fp + fn, 1) * 100)
        r["auc"] = float(roc_auc_score(y, p) * 100)
    return r


if __name__ == "__main__":
    df = load_corpus(); y = df.label.values
    tr, va, te = group_heldout_split(df)
    ext = {k: pd.read_csv(DATA / "external" / f"{k}.csv") for k in EXT}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out, preds = {}, {}

    def record(model, name, p):
        out.setdefault(model, {})[name] = ext_metrics(ext[name].label.values, p)
        preds[f"{model}|{name}"] = p

    # TF-IDF + LR
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    trv = np.concatenate([tr, va])
    lr = LogisticRegression(max_iter=5000, C=10).fit(tf.fit_transform(df.text.iloc[trv]), y[trv])
    for k in EXT:
        record("TF-IDF + LR", k, lr.predict_proba(tf.transform(ext[k].text))[:, 1])

    for enc_key in ("minilm", "mpnet", "distilbert"):
        X = np.load(CACHE / f"raw_{enc_key}.npy")
        f = CACHE / f"ext_{enc_key}.npz"
        if f.exists():
            E = dict(np.load(f))
        else:
            enc = get_encoder(enc_key, device)
            E = {k: enc.encode(ext[k].text.astype(str).tolist(), batch_size=32) for k in EXT}
            np.savez(f, **E)
        sc = StandardScaler().fit(X[trv])
        lr = LogisticRegression(max_iter=5000).fit(sc.transform(X[trv]), y[trv])
        for k in EXT:
            record(f"{enc_key} (frozen) + LR", k, lr.predict_proba(sc.transform(E[k]))[:, 1])
        if enc_key == "distilbert":
            continue
        for v in ("mlp", "gate_mlp", "trust", "sentinel"):
            ps = {k: [] for k in EXT}
            for seed in range(5):
                m = train_head(X[tr], y[tr], X[va], y[va], v, seed)
                for k in EXT:
                    ps[k].append(predict(m, E[k].astype(np.float32))[0])
            for k in EXT:
                runs = [ext_metrics(ext[k].label.values, p) for p in ps[k]]
                agg = {mk: float(np.mean([r[mk] for r in runs])) for mk in runs[0]}
                agg.update({mk + "_sd": float(np.std([r[mk] for r in runs])) for mk in runs[0] if mk != "n"})
                out.setdefault(f"{enc_key}_{v}", {})[k] = agg
                preds[f"{enc_key}_{v}|{k}"] = np.mean(ps[k], 0)
        print("done", enc_key, flush=True)

    # off-the-shelf detector (no training on our data)
    if "--protectai" in sys.argv:  # slow on Apple MPS; normally run via protectai.py
        from transformers import pipeline
        clf = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2",
                       device=device, truncation=True, max_length=512)
        for k in EXT:
            res = clf(ext[k].text.astype(str).tolist(), batch_size=16)
            p = np.array([r["score"] if r["label"] == "INJECTION" else 1 - r["score"] for r in res])
            record("ProtectAI DeBERTa-v3 (off-the-shelf)", k, p)
        res = clf(df.text.iloc[te].tolist(), batch_size=16)
        p = np.array([r["score"] if r["label"] == "INJECTION" else 1 - r["score"] for r in res])
        from metrics import all_metrics
        out["ProtectAI DeBERTa-v3 (off-the-shelf)"]["our_grouped_test"] = all_metrics(y[te], p)
        np.save(RESULTS / "protectai_on_our_test.npy", p)

    json.dump(out, open(RESULTS / "external.json", "w"), indent=1)
    np.savez(RESULTS / "preds_external.npz", **preds)
    rows = []
    for m, d in out.items():
        rows.append({"model": m, **{f"{k}": round(d[k].get("f1", d[k].get("recall", 100 - d[k].get("fpr", 0))), 1) for k in EXT if k in d}})
    print(pd.DataFrame(rows).to_string(index=False))
