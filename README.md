# SentinelAgent

**Lightweight prompt injection detection for tool-augmented LLM agents**

Code and data for the paper *SentinelAgent: Lightweight Prompt Injection Detection for Tool-Augmented LLM Agents* (Tanvir Islam Robin, Sadia Islam Shefa; ICECE 2026).

> ⚠️ **Content warning.** This repository contains prompt injection and jailbreak attempts collected and generated for research. Use it only for research on detecting and defending against such attacks.

SentinelAgent screens untrusted text (user prompts, retrieved documents, tool outputs) before it enters an agent's context. A frozen sentence encoder (all-MiniLM-L6-v2 for **SentinelAgent-S**, all-mpnet-base-v2 for **SentinelAgent-B**) is followed by a security-aware feature gate, a 128-dimensional Trust Embedding, and a temperature-calibrated **Trust Score** that drives a three-zone **block / ask user / allow** policy.

## Key results

All numbers are from nested template-disjoint 5-fold cross-validation (3 seeds) or the simulated agent, as reported in the paper.

| | SentinelAgent-S |
|---|---|
| F1 / ROC-AUC (SentinelAgent-Bench) | 96.83% / 99.45% (no significant F1 difference from SBERT+MLP) |
| Hard-benign false-positive rate | 11.17% (lowest of the MiniLM heads) |
| Agent attack success rate (three-zone policy) | 29.5% → 9.5%, with 88.4% benign task completion |
| Parameters / CPU latency (Apple M2) | 23.2M / 9.3 ms per input (9.4× faster than a 184M DeBERTa detector) |

## SentinelAgent-Bench

`data/sentinelagent_bench_v2.csv`: 24,434 prompts (10,976 benign, 13,458 malicious).

| Column | Meaning |
|---|---|
| `text` | the prompt or tool output |
| `label` | 0 = benign, 1 = malicious (prompt injection) |
| `attack_family` | one of 37 families (core subset) or one of 16 categories (hard subset) |
| `group` | attack group used for unseen-group evaluation |
| `cluster` | template cluster id (connected components of MiniLM cosine ≥ 0.90); **keep each cluster on one side of any split** |
| `source` | `v1` = core subset (19,251), `v2` = hard subset (5,183) |
| `category` | fine-grained category |

* **Core subset:** property-controlled generation with a Google Gemma model from public seed examples, including `deepset/prompt-injections`. Raw files are in `data/raw/`.
* **Hard subset:** 8 hard-benign and 8 hard-attack categories generated with Gemini 2.5 Flash-Lite and GPT-4.1-nano, labeled blind by GPT-4o-mini (only agreeing samples kept), and de-duplicated against the core subset and all external test sets. See `data/v2_clean.csv`, plus `data/v2_raw.jsonl` and `data/v2_judge_labels.json` for provenance.
* **Agent scenarios:** `data/agent_scenarios.json` (176 benign tasks and 176 injected copies).

> **Use template-disjoint splits.** Random splits of this data overstate accuracy: under a random split, 77% of test prompts have a near-duplicate in training, and TF-IDF reaches 99.5%.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python encode.py raw minilm mpnet distilbert   # embeddings of the core subset -> cache/
python clusters.py 0.9                          # template clusters
python audit.py                                 # leakage analysis
python external_data.py                         # downloads the six public external datasets
python run_external.py                          # caches external embeddings, core-subset external results
python run_classical.py && python run_neural.py # core-subset protocols
python run_v2.py                                # benchmark experiments + external evaluation
for f in 0 1 2 3 4; do python nested_cv.py $f; done && python nested_cv.py aggregate
python crossfit.py                              # Table II statistics, cross-fitted calibration
python tune.py && python final_eval.py          # deployed-model configuration, external + unseen groups
python protectai.py                             # DeBERTa reference detector
python gate_analysis.py                         # gate analysis and figures (needs agent_rescore.py outputs)
```

Regenerating the hard subset (`gen_v2.py`, `validate_v2.py`) and running the agent simulation (`agent_scenarios.py`, `agent_sim.py`, `agent_eval.py`, `agent_rescore.py`, `policy.py nested_oof.npz`) call LLMs through OpenRouter. They need a `.env` file in the repository root containing `OPENROUTER_API_KEY=...`; it is git-ignored. Stored outputs of every experiment are in `results/`.

External datasets are **not redistributed**. `external_data.py` downloads them from Hugging Face: `deepset/prompt-injections`, `jackhhao/jailbreak-classification`, `xTRam1/safe-guard-prompt-injection`, `reshabhs/SPML_Chatbot_Prompt_Injection`, `Lakera/gandalf_ignore_instructions` and `leolee99/NotInject`.

Hardware used: Apple M2 laptop, 8 GB RAM, no discrete GPU.

## License

* Code: MIT (see `LICENSE`).
* Data (`data/`): CC BY 4.0 (see `DATA_LICENSE`). Seed material from public datasets remains under its original licenses.

## Citation

```bibtex
@inproceedings{robin2026sentinelagent,
  title     = {SentinelAgent: Lightweight Prompt Injection Detection for Tool-Augmented LLM Agents},
  author    = {Robin, Tanvir Islam and Shefa, Sadia Islam},
  booktitle = {International Conference on Electrical and Computer Engineering (ICECE)},
  year      = {2026}
}
```
