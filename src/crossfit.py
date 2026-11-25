"""Cross-fitted calibration and decision-band evaluation on nested-CV out-of-fold predictions.

For each outer fold k: fit temperature + band on OOF predictions of the other four folds, apply to fold k.
Also prints the nested-CV summary table and Wilcoxon tests used in the paper.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DATA, RESULTS
from metrics import ece
from nested_cv import setup
from policy import calibrate, choose_band, fit_temperature, zone_metrics

if __name__ == "__main__":
    df, strat, folds = setup()
    y = df.label.values
    N = json.load(open(RESULTS / "nested_cv.json"))
    R = N["runs"]
    oof = dict(np.load(RESULTS / "nested_oof.npz"))
    C = json.load(open(RESULTS / "v2_results.json"))["B_cv"]

    print("== nested CV (mean±sd over 15 runs) ==")
    for m, runs in list(C.items())[:4] + list(R.items()):
        if m in C and m not in ("tfidf_lr", "minilm_lr", "mpnet_lr", "distilbert_lr"):
            continue
        g = lambda p, k: (np.mean([r[p][k] for r in runs]), np.std([r[p][k] for r in runs]))
        print(f"{m:16s} F1 {g('all','f1')[0]:.2f}±{g('all','f1')[1]:.2f} AUC {g('all','auc')[0]:.2f} "
              f"hardF1 {g('hard','f1')[0]:.2f} hardFPR {g('hard','fpr')[0]:.2f} hardFNR {g('hard','fnr')[0]:.2f}")
    print("== Wilcoxon (F1, all / hard) ==")
    out = {"tests": {}}
    for enc in ("minilm", "mpnet"):
        for b in ("mlp", "gate_mlp", "trust"):
            for part in ("all", "hard"):
                a = [r[part]["f1"] for r in R[f"{enc}_sentinel"]]; c = [r[part]["f1"] for r in R[f"{enc}_{b}"]]
                d = np.array(a) - np.array(c)
                p = wilcoxon(a, c).pvalue if np.any(d) else 1.0
                out["tests"][f"{enc}|{b}|{part}"] = {"dF1": float(d.mean()), "wins": int((d > 0).sum()), "p": float(p)}
                print(f"{enc} sentinel-{b} {part}: dF1={d.mean():+.2f} wins={(d>0).sum()}/15 p={p:.3f}")

    print("== cross-fitted calibration / band ==")
    out["crossfit"] = {}
    for m in ("minilm_sentinel", "minilm_mlp", "mpnet_sentinel", "mpnet_mlp"):
        p = np.mean([oof[f"{m}__{s}"] for s in range(3)], 0)
        raw, cal, zm = [], [], []
        Ts, bands = [], []
        for k, (_, te) in enumerate(folds):
            fit = np.setdiff1d(np.arange(len(y)), te)
            T = fit_temperature(p[fit], y[fit]); band = choose_band(y[fit], calibrate(p[fit], T))
            pc = calibrate(p[te], T)
            raw.append(ece(y[te], p[te]) * 100); cal.append(ece(y[te], pc) * 100)
            zm.append(zone_metrics(y[te], pc, *band)); Ts.append(T); bands.append(band)
        agg = {k: float(np.mean([z[k] for z in zm])) for k in zm[0]}
        out["crossfit"][m] = {"ece_raw": float(np.mean(raw)), "ece_cal": float(np.mean(cal)), "zones": agg,
                              "temperatures": [float(t) for t in Ts], "bands": bands}
        print(m, f"ECE {np.mean(raw):.2f}->{np.mean(cal):.2f}", {k: round(v, 2) for k, v in agg.items()}, "bands", bands)
    json.dump(out, open(RESULTS / "nested_stats.json", "w"), indent=1, default=float)
