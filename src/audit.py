"""Dataset audit: near-duplicates, template artifacts, and shortcut baselines.

Writes results/audit_raw.json and (with --dedup THRESH) data/corpus_clean.csv.
"""
import argparse
import re
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common import CACHE, DATA, heldout_split, load_raw, save_json

KEYWORDS = ["ignore", "previous", "instruction", "system prompt", "override", "reveal",
            "disregard", "hidden", "secret", "execute", "admin", "developer", "bypass", "forget"]


def shortcut_features(texts):
    f = []
    for t in texts:
        tl = t.lower()
        f.append([len(t), len(t.split()), t.count("\n"), sum(c in "{}<>[]\"=:" for c in t),
                  sum(ord(c) > 127 for c in t), t.endswith("?")] + [tl.count(k) for k in KEYWORDS])
    return np.array(f, dtype=float)


def evaluate(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"acc": accuracy_score(yte, p > 0.5), "f1": f1_score(yte, p > 0.5), "auc": roc_auc_score(yte, p)}


def max_cross_cos(E, a_idx, b_idx, bs=2048):
    """For each item in a_idx, max cosine similarity to any item in b_idx."""
    E = torch.tensor(E)
    E = torch.nn.functional.normalize(E, dim=1)
    B = E[b_idx]
    out = []
    for i in range(0, len(a_idx), bs):
        out.append((E[a_idx[i:i + bs]] @ B.T).max(1).values)
    return torch.cat(out).numpy()


def greedy_dedup(E, labels, thresh, bs=1024):
    """Keep a sample only if its cosine similarity to every already-kept sample is < thresh."""
    E = torch.nn.functional.normalize(torch.tensor(E), dim=1)
    order = np.random.RandomState(0).permutation(len(E))
    keep = []
    keptE = torch.empty(0, E.shape[1])
    for i in range(0, len(order), bs):
        chunk = order[i:i + bs]
        C = E[chunk]
        ok = np.ones(len(chunk), bool)
        if len(keptE):
            ok &= ((C @ keptE.T).max(1).values < thresh).numpy()
        # resolve within-chunk duplicates sequentially
        S = (C @ C.T).numpy()
        for j in range(len(chunk)):
            if not ok[j]:
                continue
            for k in range(j + 1, len(chunk)):
                if ok[k] and S[j, k] >= thresh:
                    ok[k] = False
        keep.extend(chunk[ok])
        keptE = torch.cat([keptE, C[ok]])
    return np.sort(np.array(keep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup", type=float, default=None)
    args = ap.parse_args()

    df = load_raw()
    y = df.label.values
    tr, va, te = heldout_split(df)
    out = {"n": len(df), "benign": int((y == 0).sum()), "malicious": int((y == 1).sum())}

    # 1) shortcut baselines: surface features only, and keyword-only
    F = shortcut_features(df.text)
    out["shortcut_surface_features_LR"] = evaluate(LogisticRegression(max_iter=5000), F[tr], y[tr], F[te], y[te])
    out["shortcut_length_only_LR"] = evaluate(LogisticRegression(max_iter=5000), F[tr][:, :2], y[tr], F[te][:, :2], y[te])
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform(df.text.iloc[tr]); Xte = tf.transform(df.text.iloc[te])
    out["tfidf_LR"] = evaluate(LogisticRegression(max_iter=5000, C=10), Xtr, y[tr], Xte, y[te])

    # 2) template artifacts: most common trailing sentences per class
    def last_sentence(t):
        s = [x.strip() for x in re.split(r"(?<=[.?!])\s+", t.strip()) if x.strip()]
        return s[-1] if len(s) > 1 else None
    for lab, name in [(0, "benign"), (1, "malicious")]:
        c = Counter(filter(None, (last_sentence(t) for t in df.text[y == lab])))
        top = c.most_common(15)
        out[f"top_trailing_sentences_{name}"] = top
        out[f"share_with_top15_trailing_{name}"] = sum(v for _, v in top) / int((y == lab).sum())

    # 3) semantic near-duplicates between test and train (MiniLM cosine)
    E = np.load(CACHE / "raw_minilm.npy")
    m = max_cross_cos(E, te, tr)
    out["test_to_train_maxcos_quantiles"] = {q: float(np.quantile(m, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    for t in (0.90, 0.95, 0.98):
        out[f"test_frac_with_train_neighbor_cos>={t}"] = float((m >= t).mean())
    save_json(out, "audit_raw.json")
    for k, v in out.items():
        if not k.startswith("top_"):
            print(k, v)

    if args.dedup:
        keep = greedy_dedup(E, y, args.dedup)
        clean = df.iloc[keep].reset_index(drop=True)
        clean.to_csv(DATA / "corpus_clean.csv", index=False)
        np.save(DATA / "clean_index_in_raw.npy", keep)
        print("dedup", args.dedup, "kept", len(clean), clean.label.value_counts().to_dict())
        print(clean.group.value_counts().to_string())
