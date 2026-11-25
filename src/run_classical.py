"""Classical baselines: TF-IDF + {LR, RF, XGBoost} and frozen-embedding + LR.

Protocols: random held-out (paper's original), template-disjoint held-out, template-disjoint 5-fold CV x 3 seeds.
"""
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import CACHE, RESULTS, cv_splits, group_heldout_split, heldout_split, load_corpus
from metrics import all_metrics


def make(name, seed):
    if name == "tfidf_lr":
        return LogisticRegression(max_iter=5000, C=10)
    if name == "tfidf_rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "tfidf_xgb":
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, n_jobs=8,
                             random_state=seed, eval_metric="logloss")
    return LogisticRegression(max_iter=5000, C=1.0)  # embedding + LR


def features(name, df, tr, te, emb):
    if name.startswith("tfidf"):
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        return tf.fit_transform(df.text.iloc[tr]), tf.transform(df.text.iloc[te])
    X = emb[name.split("_")[0]]
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), sc.transform(X[te])


MODELS = ["tfidf_lr", "tfidf_rf", "tfidf_xgb", "minilm_lr", "mpnet_lr", "distilbert_lr"]

if __name__ == "__main__":
    df = load_corpus()
    y = df.label.values
    emb = {k: np.load(CACHE / f"raw_{k}.npy") for k in ("minilm", "mpnet", "distilbert")}
    out, preds = {}, {}
    for proto, (tr, va, te) in [("random_heldout", heldout_split(df)), ("grouped_heldout", group_heldout_split(df))]:
        preds[f"{proto}_y"] = y[te]
        preds[f"{proto}_idx"] = te
        for m in MODELS:
            t0 = time.time()
            Xtr, Xte = features(m, df, np.concatenate([tr, va]), te, emb)  # classical models: no early stopping, use train+val
            clf = make(m, 0).fit(Xtr, y[np.concatenate([tr, va])])
            p = clf.predict_proba(Xte)[:, 1]
            preds[f"{proto}_{m}"] = p
            out.setdefault(proto, {})[m] = all_metrics(y[te], p)
            print(proto, m, {k: round(v, 2) for k, v in out[proto][m].items() if k in ("acc", "f1", "auc")}, f"{time.time()-t0:.0f}s", flush=True)
    for s, f, tr, va, te in cv_splits(df, grouped=True):
        trv = np.concatenate([tr, va])
        for m in MODELS:
            if m.startswith("tfidf_lr") or m.endswith("_lr"):
                if s > 0:
                    continue  # deterministic: one run per fold
            Xtr, Xte = features(m, df, trv, te, emb)
            p = make(m, s).fit(Xtr, y[trv]).predict_proba(Xte)[:, 1]
            out.setdefault("grouped_cv", {}).setdefault(m, []).append({"seed": s, "fold": f, **all_metrics(y[te], p)})
        print("cv", s, f, flush=True)
    json.dump(out, open(RESULTS / "classical.json", "w"), indent=1, default=float)
    np.savez(RESULTS / "preds_classical.npz", **preds)
