"""Nested template-disjoint CV: hyperparameters are selected INSIDE each outer fold's training data.

usage: python nested_cv.py <fold 0-4>      -> results/nested_fold{f}.json + nested_oof_fold{f}.npz
       python nested_cv.py aggregate       -> results/nested_cv.json, results/nested_oof.npz

Outer folds: identical to tune.py / run_v2.py (StratifiedGroupKFold(5), seed 0).
Inner selection: a template-disjoint validation split of the outer training data (seed = fold),
same grid and criterion (validation F1) for every head; then 3 training seeds with the selected config.
"""
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, DATA, RESULTS
from metrics import all_metrics
from models import predict
from tune import GRID, HEADS, train

torch.set_num_threads(2)


def setup():
    df = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    strat = df.label.astype(str) + "_" + df.source
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, strat, df.cluster))
    return df, strat, folds


if __name__ == "__main__":
    df, strat, folds = setup()
    y = df.label.values
    if sys.argv[1] == "aggregate":
        res, oof, cfg = {}, {}, {}
        for f in range(5):
            r = json.load(open(RESULTS / f"nested_fold{f}.json"))
            for k, v in r["runs"].items():
                res.setdefault(k, []).extend(v)
            cfg[f] = r["selected"]
            z = dict(np.load(RESULTS / f"nested_oof_fold{f}.npz"))
            te = folds[f][1]
            for k, p in z.items():
                oof.setdefault(k, np.zeros(len(df)))[te] = p
        json.dump({"runs": res, "selected_per_fold": cfg}, open(RESULTS / "nested_cv.json", "w"), indent=1, default=float)
        np.savez(RESULTS / "nested_oof.npz", **oof)
        print("aggregated", len(res), "models")
        sys.exit()

    f = int(sys.argv[1])
    trv, te = folds[f]
    _, vi = next(StratifiedGroupKFold(9, shuffle=True, random_state=f).split(trv, strat.values[trv], df.cluster.values[trv]))
    va = trv[vi]; tr = np.setdiff1d(trv, va)
    hm = df.source.values[te] == "v2"
    X = {k: np.concatenate([np.load(CACHE / f"raw_{k}.npy"), np.load(CACHE / f"v2_{k}.npy")]) for k in ("minilm", "mpnet")}
    out = {"selected": {}, "runs": {}}
    oof = {}
    for enc in X:
        for v in HEADS:
            best, bf = None, -1
            for lr, wd, bal in GRID:  # inner selection on this fold's own training data only
                m = train(X[enc][tr], y[tr], X[enc][va], y[va], v, 0, lr, wd, bal)
                fv = f1_score(y[va], predict(m, X[enc][va])[0] >= 0.5) * 100
                if fv > bf:
                    best, bf = (lr, wd, bal), fv
            out["selected"][f"{enc}_{v}"] = {"lr": best[0], "wd": best[1], "balanced": best[2], "val_f1": bf}
            for s in range(3):
                m = train(X[enc][tr], y[tr], X[enc][va], y[va], v, s, *best)
                p = predict(m, X[enc][te])[0]
                oof[f"{enc}_{v}__{s}"] = p
                out["runs"].setdefault(f"{enc}_{v}", []).append({"fold": f, "seed": s, "all": all_metrics(y[te], p),
                                                                  "hard": all_metrics(y[te][hm], p[hm])})
            print(f"fold {f} {enc} {v} selected {best} val_f1={bf:.2f}", flush=True)
    json.dump(out, open(RESULTS / f"nested_fold{f}.json", "w"), indent=1, default=float)
    np.savez(RESULTS / f"nested_oof_fold{f}.npz", **oof)
    print("done fold", f, flush=True)
