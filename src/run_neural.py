"""Neural heads on frozen embeddings: SBERT+MLP, +gate, trust projection, SentinelAgent.

Protocols: random held-out and template-disjoint held-out (5 training seeds each),
template-disjoint 5-fold CV x 3 training seeds.
"""
import json
import sys
import time

import numpy as np
import torch

from common import CACHE, RESULTS, cv_splits, group_heldout_split, heldout_split, load_corpus
from metrics import all_metrics
from models import n_params, predict, train_head

VARIANTS = ["mlp", "gate_mlp", "trust", "sentinel"]
torch.set_num_threads(8)

if __name__ == "__main__":
    encoders = sys.argv[1:] or ["mpnet", "minilm"]
    df = load_corpus()
    y = df.label.values
    out, preds = {}, {}
    for enc in encoders:
        X = np.load(CACHE / f"raw_{enc}.npy")
        for proto, (tr, va, te) in [("random_heldout", heldout_split(df)), ("grouped_heldout", group_heldout_split(df))]:
            preds[f"{proto}_y"] = y[te]
            for v in VARIANTS:
                runs = []
                for seed in range(5):
                    t0 = time.time()
                    m = train_head(X[tr], y[tr], X[va], y[va], v, seed)
                    p, _ = predict(m, X[te])
                    preds[f"{proto}_{enc}_{v}_s{seed}"] = p
                    runs.append({"seed": seed, "epochs": m.epochs_run, "params": n_params(m), **all_metrics(y[te], p)})
                    if v == "sentinel" and seed == 0:
                        torch.save(m.state_dict(), CACHE / f"sentinel_{enc}_{proto}.pt")
                out.setdefault(proto, {})[f"{enc}_{v}"] = runs
                acc = np.mean([r["acc"] for r in runs]); f1 = np.mean([r["f1"] for r in runs])
                print(proto, enc, v, f"acc={acc:.2f} f1={f1:.2f} sd={np.std([r['f1'] for r in runs]):.2f}", f"{time.time()-t0:.0f}s/run", flush=True)
        for s, f, tr, va, te in cv_splits(df, grouped=True):
            for v in VARIANTS:
                m = train_head(X[tr], y[tr], X[va], y[va], v, s)
                p, _ = predict(m, X[te])
                out.setdefault("grouped_cv", {}).setdefault(f"{enc}_{v}", []).append({"seed": s, "fold": f, **all_metrics(y[te], p)})
            print("cv", enc, s, f, flush=True)
        json.dump(out, open(RESULTS / f"neural_{'_'.join(encoders)}.json", "w"), indent=1, default=float)
        np.savez(RESULTS / f"preds_neural_{'_'.join(encoders)}.npz", **preds)
