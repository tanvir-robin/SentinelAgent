"""Shared data loading, family grouping, and split utilities for the SentinelAgent revision."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for p in (DATA, CACHE, RESULTS, FIGURES):
    p.mkdir(exist_ok=True)

# label convention used everywhere in the revision: 1 = malicious (positive class), 0 = benign
FAMILY_GROUPS = {
    "Benign": "Benign",
    "PromptLeak": "Prompt leakage",
    "InstructionOverride": "Override & impersonation",
    "RoleplayJailbreak": "Override & impersonation",
    "SocialEngineering": "Override & impersonation",
    "AuthorityImpersonation": "Override & impersonation",
    "FakeSystemMessage": "Override & impersonation",
    "FakeDeveloperMessage": "Override & impersonation",
    "IndirectPromptInjection": "Indirect content injection",
    "BrowserContentInjection": "Indirect content injection",
    "SearchResultInjection": "Indirect content injection",
    "EmailInjection": "Indirect content injection",
    "PDFInjection": "Indirect content injection",
    "READMEInjection": "Indirect content injection",
    "DocumentationInjection": "Indirect content injection",
    "LogFileInjection": "Indirect content injection",
    "ToolOutputPoisoning": "Tool & agent manipulation",
    "FunctionCallingManipulation": "Tool & agent manipulation",
    "ToolInvocationManipulation": "Tool & agent manipulation",
    "AgentPlanningManipulation": "Tool & agent manipulation",
    "ReflectionManipulation": "Tool & agent manipulation",
    "SelfCorrectionManipulation": "Tool & agent manipulation",
    "RAGPoisoning": "Memory, context & RAG",
    "MemoryPoisoning": "Memory, context & RAG",
    "ContextManipulation": "Memory, context & RAG",
    "MultiTurnInjection": "Memory, context & RAG",
    "ConversationHistoryManipulation": "Memory, context & RAG",
    "PromptObfuscation": "Obfuscation",
    "UnicodeObfuscation": "Obfuscation",
    "Base64Obfuscation": "Obfuscation",
    "HTMLInjection": "Structured-format injection",
    "MarkdownInjection": "Structured-format injection",
    "XMLInjection": "Structured-format injection",
    "JSONInjection": "Structured-format injection",
    "YAMLInjection": "Structured-format injection",
    "CSVInjection": "Structured-format injection",
    "SQLInjection": "Structured-format injection",
    "Other": "Other malicious",
}


def load_raw() -> pd.DataFrame:
    """Original 19,251-record corpus with family metadata."""
    m = pd.read_csv(RAW / "attack_metadata.csv")
    d = json.load(open(RAW / "prompt-injections-combined.json"))
    assert len(m) == len(d) and all(m.text.values == [x["text"] for x in d])
    m["label"] = m["label"].astype(int)
    m["group"] = m["attack_family"].map(FAMILY_GROUPS)
    assert m["group"].notna().all()
    assert (m.loc[m.label == 0, "attack_family"] == "Benign").all()
    return m[["id", "text", "label", "attack_family", "group"]]


def load_clean() -> pd.DataFrame:
    """Corpus after semantic near-duplicate removal (built by audit.py)."""
    return pd.read_csv(DATA / "corpus_clean.csv")


def heldout_split(df: pd.DataFrame, seed: int = 42):
    """Stratified 80/10/10 split, stratified on label x attack family."""
    from sklearn.model_selection import train_test_split

    strat = df["attack_family"].values
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=strat, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=strat[tmp], random_state=seed)
    return tr, va, te


def load_corpus() -> pd.DataFrame:
    """Raw corpus plus its near-duplicate cluster id (cosine >= 0.90, MiniLM; see clusters.py)."""
    df = load_raw()
    df["cluster"] = np.load(CACHE / "clusters_0.9.npy")
    return df


def group_heldout_split(df: pd.DataFrame, seed: int = 42):
    """Template-disjoint held-out split (fold 0 of the grouped 5-fold CV, ~67/11/22):
    no near-duplicate cluster spans two partitions."""
    _, _, tr, va, te = next(cv_splits(df, grouped=True, seeds=(seed,)))
    return tr, va, te


def cv_splits(df: pd.DataFrame, grouped: bool, n_splits=5, seeds=(0, 1, 2)):
    """Yield (seed, fold, train, val, test). Val is carved from train the same way as test."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

    for s in seeds:
        if grouped:
            outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=s).split(df, df.label, df.cluster)
        else:
            outer = StratifiedKFold(n_splits, shuffle=True, random_state=s).split(df, df.attack_family)
        for f, (trv, te) in enumerate(outer):
            if grouped:
                inner = StratifiedGroupKFold(9, shuffle=True, random_state=s)
                _, v = next(inner.split(trv, df.label.values[trv], df.cluster.values[trv]))
                va = trv[v]
            else:
                _, va = train_test_split(trv, test_size=1 / 9, stratify=df.attack_family.values[trv], random_state=s)
            tr = np.setdiff1d(trv, va)
            yield s, f, tr, va, te


def save_json(obj, name):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
