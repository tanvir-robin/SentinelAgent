"""Classification heads trained on frozen sentence embeddings."""
import copy

import numpy as np
import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    """Security-Aware Feature Gate: u = sigmoid(W e + b) * e."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, e):
        g = torch.sigmoid(self.lin(e))
        self.last_gate = g.detach()
        return g * e


def block(i, o, p):
    return [nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(p)]


class Head(nn.Module):
    """variant:
    mlp        - SBERT+MLP baseline: d -> 256 -> 2
    gate_mlp   - Feature gate, then d -> 256 -> 2
    trust      - Trust-embedding projection without gate: d -> 512 -> 256 -> 128 -> 2
    sentinel   - SentinelAgent: gate + d -> 512 -> 256 -> 128 (Trust Embedding) -> 2
    """

    def __init__(self, d, variant):
        super().__init__()
        self.variant = variant
        self.gate = FeatureGate(d) if variant in ("gate_mlp", "sentinel") else nn.Identity()
        if variant in ("mlp", "gate_mlp"):
            self.body = nn.Sequential(*block(d, 256, 0.3))
            self.cls = nn.Linear(256, 2)
        else:
            self.body = nn.Sequential(*block(d, 512, 0.3), *block(512, 256, 0.3), *block(256, 128, 0.2))
            self.cls = nn.Linear(128, 2)

    def embed(self, e):
        return self.body(self.gate(e))

    def forward(self, e):
        return self.cls(self.embed(e))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_head(Xtr, ytr, Xva, yva, variant, seed, lr=1e-4, bs=64, max_epochs=40, patience=5, device="cpu"):
    """Adam + cross-entropy; keep the checkpoint with lowest validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Head(Xtr.shape[1], variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device); yva_t = torch.tensor(yva, device=device)
    g = torch.Generator().manual_seed(seed)
    best, best_loss, bad, epochs = None, float("inf"), 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_t), yva_t).item()
        epochs = ep + 1
        if vl < best_loss - 1e-5:
            best_loss, best, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    model.eval()
    model.epochs_run = epochs
    return model


@torch.no_grad()
def predict(model, X, device="cpu"):
    logits = model(torch.tensor(X, device=device))
    return torch.softmax(logits, 1)[:, 1].cpu().numpy(), logits.cpu().numpy()
