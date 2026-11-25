"""Re-score agent scenarios with the tuned heads (same training split as agent_eval.py) and save the trained models."""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, DATA, RESULTS
from encode import get_encoder
from models import predict
from tune import train

if __name__ == "__main__":
    cfg = json.load(open(RESULTS / "tuning_selected.json"))
    scen = json.load(open(DATA / "agent_scenarios.json"))
    texts = [s["tool_output"] for s in scen]
    bench = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    y = bench.label.values
    strat = bench.label.astype(str) + "_" + bench.source
    idx = np.arange(len(bench))
    _, vi = next(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(idx, strat.values, bench.cluster.values))
    va = idx[vi]; tr = np.setdiff1d(idx, va)
    gs = json.load(open(RESULTS / "agent_gate_scores.json"))
    for enc in ("minilm", "mpnet"):
        X = np.concatenate([np.load(CACHE / f"raw_{enc}.npy"), np.load(CACHE / f"v2_{enc}.npy")])
        Es = get_encoder(enc, "cpu").encode(texts, batch_size=32).astype(np.float32)
        for v in ("mlp", "gate_mlp", "trust", "sentinel"):
            c = cfg[f"{enc}_{v}"]
            key = f"{v}_{enc}_v2"
            if key in gs and f"{key}_untuned" not in gs:
                gs[f"{key}_untuned"] = gs[key]
            gs[key] = []
            for s in range(3):
                m = train(X[tr], y[tr], X[va], y[va], v, s, c["lr"], c["wd"], c["balanced"])
                gs[key].append(predict(m, Es)[0].tolist())
                if s == 0:
                    torch.save(m.state_dict(), CACHE / f"final_{v}_{enc}.pt")
            print("scored", key, flush=True)
    json.dump(gs, open(RESULTS / "agent_gate_scores.json", "w"))
