"""Aggregate results into tables + significance tests (results/summary.md, results/summary.json)."""
import json

import numpy as np

from common import RESULTS, group_heldout_split, load_corpus
from metrics import all_metrics, bootstrap_ci, delong, mcnemar
from sklearn.metrics import f1_score, roc_auc_score

KEYS = ["acc", "prec", "rec", "f1", "auc", "fnr", "fpr", "tpr@1%fpr", "ece"]
NAMES = {"tfidf_lr": "TF-IDF + LR", "tfidf_rf": "TF-IDF + RF", "tfidf_xgb": "TF-IDF + XGBoost",
         "minilm_lr": "MiniLM (frozen) + LR", "mpnet_lr": "MPNet (frozen) + LR",
         "distilbert_lr": "DistilBERT (frozen) + LR"}


def ms(runs, k):
    v = [r[k] for r in runs]
    return np.mean(v), np.std(v)


def table(rows, title, lines):
    lines.append(f"\n### {title}\n")
    lines.append("| Model | " + " | ".join(KEYS) + " |")
    lines.append("|---" * (len(KEYS) + 1) + "|")
    for name, runs in rows:
        runs = runs if isinstance(runs, list) else [runs]
        cells = []
        for k in KEYS:
            m, s = ms(runs, k)
            cells.append(f"{m:.2f}" + (f" ± {s:.2f}" if len(runs) > 1 else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    C = json.load(open(RESULTS / "classical.json"))
    N = json.load(open(RESULTS / "neural_mpnet_minilm.json"))
    P = dict(np.load(RESULTS / "preds_neural_mpnet_minilm.npz"))
    lines, summ = ["# Revision results (malicious = positive class)"], {}
    for proto in ("random_heldout", "grouped_heldout", "grouped_cv"):
        rows = [(NAMES[k], v) for k, v in C[proto].items()]
        rows += [(k, v) for k, v in N[proto].items()]
        table(rows, proto, lines)

    # significance: SentinelAgent vs matched SBERT+MLP, per encoder & held-out protocol
    lines.append("\n### Significance: sentinel vs mlp (held-out, per training seed)\n")
    for enc in ("mpnet", "minilm"):
        for proto in ("random_heldout", "grouped_heldout"):
            y = P[f"{proto}_y"]
            res = []
            for s in range(5):
                a, b = P[f"{proto}_{enc}_sentinel_s{s}"], P[f"{proto}_{enc}_mlp_s{s}"]
                res.append({"mcnemar": mcnemar(y, a, b), "delong": delong(y, a, b)})
            # seed-averaged probabilities (ensemble view) with bootstrap CI of the F1 / AUC difference
            a = np.mean([P[f"{proto}_{enc}_sentinel_s{s}"] for s in range(5)], 0)
            b = np.mean([P[f"{proto}_{enc}_mlp_s{s}"] for s in range(5)], 0)
            joint = np.stack([a, b], 1)
            dF1 = bootstrap_ci(y, joint, lambda yy, pp: (f1_score(yy, pp[:, 0] >= .5) - f1_score(yy, pp[:, 1] >= .5)) * 100)
            dAUC = bootstrap_ci(y, joint, lambda yy, pp: (roc_auc_score(yy, pp[:, 0]) - roc_auc_score(yy, pp[:, 1])) * 100)
            summ[f"{enc}_{proto}"] = {"per_seed": res, "dF1_ci": dF1, "dAUC_ci": dAUC}
            mp = [r["mcnemar"]["p"] for r in res]; dp = [r["delong"]["p"] for r in res]
            lines.append(f"- {enc} / {proto}: McNemar p per seed = {[round(x, 4) for x in mp]}; "
                         f"DeLong p = {[round(x, 4) for x in dp]}; ΔF1 95% CI = [{dF1[0]:.2f}, {dF1[1]:.2f}]; "
                         f"ΔAUC 95% CI = [{dAUC[0]:.3f}, {dAUC[1]:.3f}]")

    # paired comparison across CV folds
    from scipy.stats import wilcoxon
    lines.append("\n### Paired CV comparison (15 fold-seed pairs, Wilcoxon signed-rank)\n")
    for enc in ("mpnet", "minilm"):
        for base in ("mlp", "gate_mlp", "trust"):
            a = [r["f1"] for r in N["grouped_cv"][f"{enc}_sentinel"]]
            b = [r["f1"] for r in N["grouped_cv"][f"{enc}_{base}"]]
            d = np.array(a) - np.array(b)
            p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
            lines.append(f"- {enc}: sentinel − {base}: mean ΔF1 = {d.mean():.2f} (sd {d.std():.2f}), wins {int((d>0).sum())}/15, p = {p:.4f}")

    # per attack group recall on grouped held-out test (seed-averaged sentinel, mpnet)
    df = load_corpus(); _, _, te = group_heldout_split(df)
    g = df.group.values[te]; y = P["grouped_heldout_y"]
    lines.append("\n### Per-group detection on template-disjoint test (mpnet, seed-avg)\n")
    for v in ("mlp", "sentinel"):
        p = np.mean([P[f"grouped_heldout_mpnet_{v}_s{s}"] for s in range(5)], 0)
        parts = []
        for grp in sorted(set(g)):
            m = g == grp
            rate = ((p[m] >= .5) == (y[m] == 1)).mean() * 100
            parts.append(f"{grp}: {rate:.1f} (n={m.sum()})")
        lines.append(f"- {v}: " + "; ".join(parts))
    open(RESULTS / "summary.md", "w").write("\n".join(lines))
    json.dump(summ, open(RESULTS / "summary.json", "w"), indent=1, default=float)
    print("\n".join(lines))
