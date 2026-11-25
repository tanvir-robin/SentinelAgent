"""Union-find clustering of near-duplicate prompts (cosine >= thr on MiniLM embeddings)."""
import sys
import numpy as np, torch
from common import CACHE, load_raw

def components(E, thr, bs=2048):
    E = torch.nn.functional.normalize(torch.tensor(E), dim=1)
    n = len(E); parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(0, n, bs):
        S = E[i:i+bs] @ E.T
        r, c = torch.nonzero(S >= thr, as_tuple=True)
        for a, b in zip((r + i).tolist(), c.tolist()):
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
    return np.array([find(i) for i in range(n)])

if __name__ == "__main__":
    df = load_raw(); E = np.load(CACHE / "raw_minilm.npy")
    for thr in map(float, sys.argv[1:]):
        comp = components(E, thr)
        _, inv, cnt = np.unique(comp, return_inverse=True, return_counts=True)
        sizes = np.sort(cnt)[::-1]
        print(f"thr={thr}: {len(cnt)} clusters, largest={sizes[:5].tolist()}, singletons={(cnt==1).sum()}")
        for lab in (0, 1):
            c = np.unique(inv[df.label.values == lab])
            print("  label", lab, "clusters", len(c), "largest cluster share", np.bincount(inv[df.label.values==lab]).max() / (df.label.values==lab).sum())
        np.save(CACHE / f"clusters_{thr}.npy", inv)
