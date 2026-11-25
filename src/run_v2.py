"""Experiments on SentinelAgent-Bench v2 (= v1 corpus + hard extension).

A. v1-trained models evaluated on the hard extension (does the original data cover hard cases?)
B. Template-disjoint 5-fold CV x 3 seeds on v1+v2, all models; metrics overall and on the hard subset
C. External zero-shot evaluation of models trained on v1+v2
"""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from clusters import components
from common import CACHE, DATA, RESULTS, FAMILY_GROUPS, group_heldout_split, load_corpus
from encode import get_encoder
from metrics import all_metrics
from models import predict, train_head
from run_external import EXT, ext_metrics

torch.set_num_threads(8)
ENCS = ["minilm", "mpnet", "distilbert"]
HEADS = ["mlp", "gate_mlp", "trust", "sentinel"]


def build():
    v1 = load_corpus()
    v1["source"] = "v1"; v1["category"] = v1["attack_family"]
    v2 = pd.read_csv(DATA / "v2_clean.csv")
    v2["source"] = "v2"; v2["attack_family"] = v2["category"]
    v2["group"] = np.where(v2.label == 0, "Benign (hard)", "Hard attack")
    Ev2 = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for k in ENCS:
        f = CACHE / f"v2_{k}.npy"
        if not f.exists():
            np.save(f, get_encoder(k, device).encode(v2.text.tolist(), batch_size=32).astype(np.float32))
        Ev2[k] = np.load(f)
    comp = components(Ev2["minilm"], 0.90)
    v2["cluster"] = pd.factorize(comp)[0] + v1.cluster.max() + 1
    cols = ["text", "label", "attack_family", "group", "cluster", "source", "category"]
    df = pd.concat([v1[cols], v2[cols]], ignore_index=True)
    X = {k: np.concatenate([np.load(CACHE / f"raw_{k}.npy"), Ev2[k]]) for k in ENCS}
    df.to_csv(DATA / "sentinelagent_bench_v2.csv", index=False)
    return df, X


def fit_predict(model, df, X, tr, va, test_sets, seed):
    """Train `model` on tr (va for early stopping) and return {name: probs} for each (texts, embeddings) test set."""
    y = df.label.values
    out = {}
    if model == "tfidf_lr":
        trv = np.concatenate([tr, va])
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        clf = LogisticRegression(max_iter=5000, C=10).fit(tf.fit_transform(df.text.iloc[trv]), y[trv])
        return {n: clf.predict_proba(tf.transform(t))[:, 1] for n, (t, _) in test_sets.items()}
    enc, kind = model.split("_", 1)
    if kind == "lr":
        trv = np.concatenate([tr, va])
        sc = StandardScaler().fit(X[enc][trv])
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(X[enc][trv]), y[trv])
        return {n: clf.predict_proba(sc.transform(e[enc]))[:, 1] for n, (_, e) in test_sets.items()}
    m = train_head(X[enc][tr], y[tr], X[enc][va], y[va], kind, seed)
    return {n: predict(m, e[enc].astype(np.float32))[0] for n, (_, e) in test_sets.items()}


MODELS = ["tfidf_lr"] + [f"{e}_lr" for e in ENCS] + [f"{e}_{h}" for e in ("minilm", "mpnet") for h in HEADS]

if __name__ == "__main__":
    df, X = build()
    y = df.label.values
    v1n = int((df.source == "v1").sum())
    print("bench v2:", len(df), df.groupby(["source", "label"]).size().to_dict(), flush=True)
    res = {}

    # A. v1-trained -> hard extension
    v1 = df.iloc[:v1n].reset_index(drop=True)
    tr, va, _ = group_heldout_split(load_corpus())
    hard = np.arange(v1n, len(df))
    hard_set = {"hard": (df.text.iloc[hard], {k: X[k][hard] for k in ENCS})}
    for m in MODELS:
        seeds = range(3) if m.split("_", 1)[1] in HEADS else [0]
        ps = [fit_predict(m, df, X, tr, va, hard_set, s)["hard"] for s in seeds]
        runs = [all_metrics(y[hard], p) for p in ps]
        cat = df.category.values[hard]
        per_cat = {c: float(np.mean([((p[cat == c] >= .5) == y[hard][cat == c]).mean() * 100 for p in ps])) for c in np.unique(cat)}
        res.setdefault("A_v1_to_hard", {})[m] = {"runs": runs, "per_category_acc": per_cat}
        print("A", m, f"acc={np.mean([r['acc'] for r in runs]):.2f} f1={np.mean([r['f1'] for r in runs]):.2f} "
              f"fpr={np.mean([r['fpr'] for r in runs]):.2f} fnr={np.mean([r['fnr'] for r in runs]):.2f}", flush=True)

    # B. template-disjoint CV on v1+v2
    oof = {}
    strat = df.label.astype(str) + "_" + df.source
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, strat, df.cluster))
    for f, (trv, te) in enumerate(folds):
        inner = StratifiedGroupKFold(9, shuffle=True, random_state=f)
        _, vi = next(inner.split(trv, strat.values[trv], df.cluster.values[trv]))
        va = trv[vi]; tr = np.setdiff1d(trv, va)
        test = {"te": (df.text.iloc[te], {k: X[k][te] for k in ENCS})}
        for m in MODELS:
            seeds = range(3) if m.split("_", 1)[1] in HEADS else [0]
            for s in seeds:
                p = fit_predict(m, df, X, tr, va, test, s)["te"]
                oof.setdefault(f"{m}|{s}", np.zeros(len(df)))[te] = p
                hm = df.source.values[te] == "v2"
                res.setdefault("B_cv", {}).setdefault(m, []).append({
                    "fold": f, "seed": s, "all": all_metrics(y[te], p), "hard": all_metrics(y[te][hm], p[hm]),
                    "v1": all_metrics(y[te][~hm], p[~hm])})
        print("B fold", f, flush=True)
    np.savez(RESULTS / "v2_oof.npz", **{k.replace("|", "__"): v for k, v in oof.items()}, y=y, source=df.source.values)

    # C. external, trained on all v1+v2 (grouped 10% validation for early stopping)
    ext = {k: pd.read_csv(DATA / "external" / f"{k}.csv") for k in EXT}
    E = {k: dict(np.load(CACHE / f"ext_{k}.npz")) for k in ENCS}
    ext_sets = {n: (ext[n].text.astype(str), {k: E[k][n] for k in ENCS}) for n in EXT}
    _, vi = next(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(df, strat, df.cluster))
    tr = np.setdiff1d(np.arange(len(df)), vi)
    ext_preds = {}
    for m in MODELS:
        seeds = range(3) if m.split("_", 1)[1] in HEADS else [0]
        ps = [fit_predict(m, df, X, tr, vi, ext_sets, s) for s in seeds]
        for n in EXT:
            runs = [ext_metrics(ext[n].label.values, p[n]) for p in ps]
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            agg.update({k + "_sd": float(np.std([r[k] for r in runs])) for k in runs[0] if k != "n"})
            res.setdefault("C_external", {}).setdefault(m, {})[n] = agg
            ext_preds[f"{m}__{n}"] = np.mean([p[n] for p in ps], 0)
        print("C", m, {n: round(res["C_external"][m][n].get("f1", res["C_external"][m][n].get("recall", 100 - res["C_external"][m][n].get("fpr", 0))), 1) for n in EXT}, flush=True)
    np.savez(RESULTS / "v2_ext_preds.npz", **ext_preds)
    json.dump(res, open(RESULTS / "v2_results.json", "w"), indent=1, default=float)
