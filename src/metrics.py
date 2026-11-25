"""Metrics with malicious = positive class, plus statistical tests."""
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def tpr_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def ece(y, p, bins=15):
    """Expected calibration error of the malicious-class probability."""
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def all_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y, yhat) * 100,
        "prec": precision_score(y, yhat, zero_division=0) * 100,
        "rec": recall_score(y, yhat) * 100,
        "f1": f1_score(y, yhat) * 100,
        "auc": roc_auc_score(y, p) * 100,
        "fnr": fn / max(fn + tp, 1) * 100,   # missed attacks
        "fpr": fp / max(fp + tn, 1) * 100,   # blocked benign
        "tpr@1%fpr": tpr_at_fpr(y, p, 0.01) * 100,
        "tpr@0.1%fpr": tpr_at_fpr(y, p, 0.001) * 100,
        "ece": ece(y, p) * 100,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pa, pb, thr=0.5):
    """Exact McNemar test on discordant predictions of two classifiers."""
    ca = (pa >= thr).astype(int) == y
    cb = (pb >= thr).astype(int) == y
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    pval = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p": float(pval)}


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def delong(y, pa, pb):
    """DeLong test for two correlated ROC AUCs (Sun & Xu fast implementation)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (pa, pb):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    sx = np.cov(np.vstack(v01)); sy = np.cov(np.vstack(v10))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}
