"""Gate analysis (final SentinelAgent-S/B) + paper figures.

Uses the held-out 10% split of SentinelAgent-Bench v2 that the final models were NOT trained on.
"""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, DATA, FIGURES, RESULTS
from models import Head

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#2b2b29"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.edgecolor": "#b5b5ae", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def fisher_ratio(Z, y):
    """Fisher discriminant ratio along the LDA direction (between / within class variance)."""
    m0, m1 = Z[y == 0].mean(0), Z[y == 1].mean(0)
    Sw = np.cov(Z[y == 0].T) + np.cov(Z[y == 1].T) + 1e-3 * np.eye(Z.shape[1])
    w = np.linalg.solve(Sw, m1 - m0)
    p0, p1 = Z[y == 0] @ w, Z[y == 1] @ w
    return float((p1.mean() - p0.mean()) ** 2 / (p0.var() + p1.var()))


if __name__ == "__main__":
    bench = pd.read_csv(DATA / "sentinelagent_bench_v2.csv")
    y_all = bench.label.values
    strat = bench.label.astype(str) + "_" + bench.source
    _, va = next(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(bench, strat, bench.cluster))
    y = y_all[va]
    out = {}
    for enc in ("minilm", "mpnet"):
        X = np.concatenate([np.load(CACHE / f"raw_{enc}.npy"), np.load(CACHE / f"v2_{enc}.npy")])[va]
        m = Head(X.shape[1], "sentinel"); m.load_state_dict(torch.load(CACHE / f"final_sentinel_{enc}.pt")); m.eval()
        with torch.no_grad():
            e = torch.tensor(X)
            g = torch.sigmoid(m.gate.lin(e)).numpy()
            u = (torch.tensor(g) * e).numpy()
            t = m.body(torch.tensor(u)).numpy()
        rng = np.random.RandomState(0); s = rng.choice(len(y), min(3000, len(y)), replace=False)
        dim_mean = g.mean(0)
        out[enc] = {
            "gate_mean": float(g.mean()), "gate_frac_below_0.1": float((g < 0.1).mean()), "gate_frac_above_0.9": float((g > 0.9).mean()),
            "dims_mostly_suppressed(mean<0.2)": int((dim_mean < 0.2).sum()), "dims_mostly_open(mean>0.8)": int((dim_mean > 0.8).sum()),
            "dims_total": int(g.shape[1]),
            "gate_class_gap_mean_abs": float(np.abs(g[y == 1].mean(0) - g[y == 0].mean(0)).mean()),
            "silhouette_raw": float(silhouette_score(X[s], y[s], metric="cosine")),
            "silhouette_gated": float(silhouette_score(u[s], y[s], metric="cosine")),
            "silhouette_trust": float(silhouette_score(t[s], y[s], metric="cosine")),
            "fisher_raw": fisher_ratio(X, y), "fisher_gated": fisher_ratio(u, y), "fisher_trust": fisher_ratio(t, y),
        }
        print(enc, {k: round(v, 3) for k, v in out[enc].items()}, flush=True)
        if enc == "minilm":
            G, Y, T, S = g, y, t, s
    json.dump(out, open(RESULTS / "gate_analysis.json", "w"), indent=1)

    # Figure: gate distribution + t-SNE of Trust Embeddings (SentinelAgent-S)
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.65))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(G[Y == 0].ravel(), bins=bins, color=BLUE, alpha=0.75, label="benign", density=True, rwidth=0.9)
    ax[0].hist(G[Y == 1].ravel(), bins=bins, color=ORANGE, alpha=0.6, label="malicious", density=True, rwidth=0.9)
    ax[0].set_xlabel("gate value $g_i$"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=6, loc="upper right")
    ax[0].set_title("(a) Gate activations", fontsize=7)
    Z = TSNE(2, random_state=0, init="pca", perplexity=40).fit_transform(T[S])
    for lab, c, n in ((0, BLUE, "benign"), (1, ORANGE, "malicious")):
        mm = Y[S] == lab
        ax[1].scatter(Z[mm, 0], Z[mm, 1], s=2, c=c, alpha=0.6, label=n, linewidths=0)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("(b) Trust Embedding t-SNE", fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "gate_analysis.pdf", bbox_inches="tight"); fig.savefig(FIGURES / "gate_analysis.png", dpi=200, bbox_inches="tight")

    # Figure: protection vs CPU cost (agent simulation, three-zone policy where applicable)
    lat = json.load(open(RESULTS / "gate_latency.json"))
    pol = json.load(open(RESULTS / "policy_results.json"))["agent"]
    agent = json.load(open(RESULTS / "agent_results.json"))["table"]
    cpu = lambda k: lat[k]["encoder_ms"][0] + lat[k]["head_ms"][0]
    pts = [
        ("SentinelAgent-S (ours)", cpu("sentinel_minilm_cpu"), pol["minilm_sentinel|validated"]["ASR"][0], BLUE, True),
        ("SentinelAgent-B (ours)", cpu("sentinel_mpnet_cpu"), pol["mpnet_sentinel|validated"]["ASR"][0], BLUE, True),
        ("SBERT+MLP-S", cpu("mlp_minilm_cpu"), pol["minilm_mlp|validated"]["ASR"][0], GRAY, False),
        ("DeBERTa detector", lat["protectai_cpu"][0], agent["protectai"]["ASR"][0], GRAY, False),
        ("LLM judge (API)", lat["llm_judge_api"][0], agent["llm_judge"]["ASR"][0], GRAY, False),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    ax.axhline(agent["none"]["ASR"][0], color=GRAY, lw=1, ls="--")
    ax.text(5.5, agent["none"]["ASR"][0] + 0.6, "no defense", color=GRAY, fontsize=7)
    offs = {"SentinelAgent-S (ours)": (-4, 7), "SentinelAgent-B (ours)": (7, -3), "SBERT+MLP-S": (5, -10),
            "DeBERTa detector": (6, 3), "LLM judge (API)": (-22, 7)}
    for name, x, yv, c, ours in pts:
        ax.scatter(x, yv, s=36 if ours else 28, c=c, edgecolors="white", linewidths=1, zorder=3)
        ax.annotate(name, (x, yv), xytext=offs[name], textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log"); ax.set_xlim(5, 4000); ax.set_ylim(0, 33)
    ax.set_xlabel("screening latency per input, Apple M2 CPU (ms, log)"); ax.set_ylabel("attack success rate (%)")
    ax.grid(axis="y", color="#e6e6e1", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIGURES / "protection_vs_cost.pdf", bbox_inches="tight"); fig.savefig(FIGURES / "protection_vs_cost.png", dpi=200, bbox_inches="tight")
    print("figures written")
