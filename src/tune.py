"""Equal-budget hyperparameter search for all neural heads on SentinelAgent-Bench v2.

Search space (same for every head): lr in {1e-4, 3e-4, 1e-3} x weight decay in {0, 1e-4} x class-balanced loss {no, yes}.
Selection: validation F1 on an inner split of the TRAINING part of CV fold 0 (test folds never used).
Then: 5-fold template-disjoint CV x 3 seeds with each head's selected configuration.
"""
import copy
import itertools
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, DATA, RESULTS
from metrics import all_metrics
from models import Head, predict

torch.set_num_threads(8)
GRID = list(itertools.product([1e-4, 3e-4, 1e-3], [0.0, 1e-4], [False, True]))
HEADS = ["mlp", "gate_mlp", "trust", "sentinel"]


def train(Xtr, ytr, Xva, yva, variant, seed, lr, wd, balanced, bs=64, max_epochs=40, patience=5):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], variant)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) if wd else torch.optim.Adam(m.parameters(), lr=lr)
    w = None
    if balanced:
        c = np.bincount(ytr, minlength=2)
        w = torch.tensor(len(ytr) / (2 * c), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva)
    g = torch.Generator().manual_seed(seed)
    best, bl, bad = None, 1e9, 0
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); lossf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(m(Xv), yv).item()
        if vl < bl - 1e-5:
            bl, best, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best); m.eval()
    return m


if __name__ == "__main__":
    df = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    y = df.label.values
    X = {k: np.concatenate([np.load(CACHE / f"raw_{k}.npy"), np.load(CACHE / f"v2_{k}.npy")]) for k in ("minilm", "mpnet")}
    strat = df.label.astype(str) + "_" + df.source
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, strat, df.cluster))

    # 1) selection on fold-0 training data only
    trv, _ = folds[0]
    _, vi = next(StratifiedGroupKFold(9, shuffle=True, random_state=0).split(trv, strat.values[trv], df.cluster.values[trv]))
    va = trv[vi]; tr = np.setdiff1d(trv, va)
    best = {}
    for enc in X:
        for v in HEADS:
            scores = []
            for lr, wd, bal in GRID:
                m = train(X[enc][tr], y[tr], X[enc][va], y[va], v, 0, lr, wd, bal)
                f = f1_score(y[va], predict(m, X[enc][va])[0] >= 0.5) * 100
                scores.append(((lr, wd, bal), f))
            cfg, f = max(scores, key=lambda t: t[1])
            best[f"{enc}_{v}"] = {"lr": cfg[0], "wd": cfg[1], "balanced": cfg[2], "val_f1": f,
                                  "grid": [[list(c), s] for c, s in scores]}
            print("select", enc, v, cfg, f"val_f1={f:.2f}", flush=True)
    json.dump(best, open(RESULTS / "tuning_selected.json", "w"), indent=1)

    # 2) CV with selected configs
    res, oof = {}, {}
    for f, (trv, te) in enumerate(folds):
        _, vi = next(StratifiedGroupKFold(9, shuffle=True, random_state=f).split(trv, strat.values[trv], df.cluster.values[trv]))
        va = trv[vi]; tr = np.setdiff1d(trv, va)
        hm = df.source.values[te] == "v2"
        for key, c in best.items():
            enc, v = key.split("_", 1)
            for s in range(3):
                m = train(X[enc][tr], y[tr], X[enc][va], y[va], v, s, c["lr"], c["wd"], c["balanced"])
                p = predict(m, X[enc][te])[0]
                oof.setdefault(f"{key}__{s}", np.zeros(len(df)))[te] = p
                res.setdefault(key, []).append({"fold": f, "seed": s, "all": all_metrics(y[te], p),
                                                "hard": all_metrics(y[te][hm], p[hm]), "v1": all_metrics(y[te][~hm], p[~hm])})
        print("cv fold", f, flush=True)
    json.dump(res, open(RESULTS / "tuned_cv.json", "w"), indent=1, default=float)
    np.savez(RESULTS / "tuned_oof.npz", **oof)
