"""Calibrated Trust Score and three-zone deployment policy (block / ask user / allow).

Trust Score T = 100 * P(benign) after temperature scaling.
Zones: T < lo -> block automatically; lo <= T <= hi -> escalate to the user; T > hi -> allow automatically.
Policies: fixed 30/70 band, and a validated band chosen on out-of-fold bench predictions so that the
automatic-decision error is <= 1% with the smallest escalation rate.
Agent: escalated content goes to the user; we assume the user rejects injected content and approves benign
content (an attentive-user assumption, stated in the paper) and report how often the user is asked.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import DATA, RESULTS
from metrics import ece

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    z = logit(p)
    nll = lambda T: -np.mean(y * np.log(1 / (1 + np.exp(-z / T)) + EPS) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z / T)) + EPS))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def calibrate(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def zones(p_mal, lo, hi):
    t = 100 * (1 - p_mal)
    return np.where(t < lo, "block", np.where(t > hi, "allow", "ask"))


def zone_metrics(y, p_mal, lo, hi):
    z = zones(p_mal, lo, hi)
    auto = z != "ask"
    wrong = ((z == "block") & (y == 0)) | ((z == "allow") & (y == 1))
    return {"escalation_rate": float((~auto).mean() * 100),
            "auto_error": float(wrong[auto].mean() * 100) if auto.any() else 0.0,
            "missed_attacks_auto_allowed": float(((z == "allow") & (y == 1)).sum() / max((y == 1).sum(), 1) * 100),
            "benign_auto_blocked": float(((z == "block") & (y == 0)).sum() / max((y == 0).sum(), 1) * 100),
            "attacks_escalated": float(((z == "ask") & (y == 1)).sum() / max((y == 1).sum(), 1) * 100)}


def choose_band(y, p_mal, max_err=1.0):
    best = (30, 70)
    best_esc = 101
    for lo in range(1, 51):
        for hi in range(50, 100):
            m = zone_metrics(y, p_mal, lo, hi)
            if m["auto_error"] <= max_err and m["escalation_rate"] < best_esc:
                best, best_esc = (lo, hi), m["escalation_rate"]
    return best


if __name__ == "__main__":
    import sys
    oof_file = sys.argv[1] if len(sys.argv) > 1 else "v2_oof.npz"
    oof = dict(np.load(RESULTS / oof_file, allow_pickle=True))
    bench = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    y = bench.label.values
    out = {"oof_file": oof_file}
    models = {"minilm_sentinel": "sentinel_minilm_v2", "minilm_mlp": "mlp_minilm_v2",
              "mpnet_sentinel": "sentinel_mpnet_v2", "mpnet_mlp": "mlp_mpnet_v2"}
    temps, bands = {}, {}
    for m in models:
        p = np.mean([v for k, v in oof.items() if k.startswith(m + "__")], 0)
        T = fit_temperature(p, y)
        pc = calibrate(p, T)
        temps[m] = T
        bands[m] = choose_band(y, pc)
        out[m] = {"temperature": T, "ece_raw": ece(y, p) * 100, "ece_cal": ece(y, pc) * 100,
                  "band_validated": bands[m], "bench_fixed_30_70": zone_metrics(y, pc, 30, 70),
                  "bench_validated": zone_metrics(y, pc, *bands[m])}
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items() if not isinstance(v, dict)})
        print("   fixed 30/70:", {k: round(v, 2) for k, v in out[m]["bench_fixed_30_70"].items()})
        print("   validated  :", bands[m], {k: round(v, 2) for k, v in out[m]["bench_validated"].items()})

    # agent simulation with the three-zone policy
    scen = json.load(open(DATA / "agent_scenarios.json"))
    eps = json.load(open(RESULTS / "agent_episodes.json"))
    gs = json.load(open(RESULTS / "agent_gate_scores.json"))
    atk = np.array([s["attack"] for s in scen])
    out["agent"] = {}
    for m, g in models.items():
        for pol, band in (("fixed_30_70", (30, 70)), ("validated", bands[m])):
            rows = []
            for p in gs[g]:
                z = zones(calibrate(np.array(p), temps[m]), *band)
                # user rejects escalated attacks (content withheld), approves escalated benign content
                blocked = (z == "block") | ((z == "ask") & atk)
                e = [eps[f"{s['id']}|{bool(b)}"] for s, b in zip(scen, blocked)]
                rows.append({"ASR": 100 * np.mean([x["attack_success"] for x, a in zip(e, atk) if a]),
                             "benign_utility": 100 * np.mean([x["utility"] for x, a in zip(e, atk) if not a]),
                             "user_asked_rate": 100 * np.mean(z == "ask"),
                             "benign_auto_blocked": 100 * np.mean((z == "block")[~atk])})
            out["agent"][f"{m}|{pol}"] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
            print("agent", m, pol, band, {k: round(v[0], 1) for k, v in out["agent"][f"{m}|{pol}"].items()})
    json.dump(out, open(RESULTS / "policy_results.json", "w"), indent=1, default=float)
