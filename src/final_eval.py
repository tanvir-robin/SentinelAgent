"""Final-model evaluations for the paper (tuned configurations, same training split as agent_rescore.py).

(a) external zero-shot evaluation of tuned heads (3 seeds)
(b) leave-one-attack-group-out on SentinelAgent-Bench v2 (held-out group + held-out benign clusters)
"""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, DATA, RESULTS
from models import predict
from run_external import EXT, ext_metrics
from tune import train

torch.set_num_threads(8)

if __name__ == "__main__":
    cfg = json.load(open(RESULTS / "tuning_selected.json"))
    bench = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    y = bench.label.values
    strat = bench.label.astype(str) + "_" + bench.source
    idx = np.arange(len(bench))
    _, va = next(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(idx, strat.values, bench.cluster.values))
    tr = np.setdiff1d(idx, va)
    ext = {k: pd.read_csv(DATA / "external" / f"{k}.csv") for k in EXT}
    out = {"external": {}, "lofo": {}}
    for enc in ("minilm", "mpnet"):
        X = np.concatenate([np.load(CACHE / f"raw_{enc}.npy"), np.load(CACHE / f"v2_{enc}.npy")])
        E = dict(np.load(CACHE / f"ext_{enc}.npz"))
        for v in ("mlp", "sentinel"):
            c = cfg[f"{enc}_{v}"]
            runs = {k: [] for k in EXT}
            for s in range(3):
                m = train(X[tr], y[tr], X[va], y[va], v, s, c["lr"], c["wd"], c["balanced"])
                for k in EXT:
                    runs[k].append(ext_metrics(ext[k].label.values, predict(m, E[k].astype(np.float32))[0]))
            out["external"][f"{enc}_{v}"] = {k: {mk: float(np.mean([r[mk] for r in rr])) for mk in rr[0]} for k, rr in runs.items()}
            print("ext", enc, v, {k: round(d.get("acc"), 1) for k, d in out["external"][f"{enc}_{v}"].items()}, flush=True)

        # (b) leave-one-attack-group-out
        groups = sorted(g for g in bench.group.unique() if not g.startswith("Benign"))
        ben = np.where(y == 0)[0]
        uc = np.unique(bench.cluster.values[ben])
        rng = np.random.RandomState(0)
        held_c = set(rng.choice(uc, len(uc) // 5, replace=False))
        bte = ben[np.isin(bench.cluster.values[ben], list(held_c))]
        for v in ("mlp", "sentinel"):
            c = cfg[f"{enc}_{v}"]
            for g in groups:
                gte = np.where(bench.group.values == g)[0]
                te = np.concatenate([gte, bte])
                rest = np.setdiff1d(idx, te)
                _, vi = next(StratifiedGroupKFold(9, shuffle=True, random_state=0).split(rest, strat.values[rest], bench.cluster.values[rest]))
                vv = rest[vi]; tt = np.setdiff1d(rest, vv)
                m = train(X[tt], y[tt], X[vv], y[vv], v, 0, c["lr"], c["wd"], c["balanced"])
                p = predict(m, X[te])[0]
                out["lofo"].setdefault(f"{enc}_{v}", {})[g] = {"n": int(len(gte)), "recall": float((p[:len(gte)] >= .5).mean() * 100),
                                                               "benign_fpr": float((p[len(gte):] >= .5).mean() * 100)}
            print("lofo", enc, v, {g: round(d["recall"], 1) for g, d in out["lofo"][f"{enc}_{v}"].items()}, flush=True)
    json.dump(out, open(RESULTS / "final_eval.json", "w"), indent=1)
