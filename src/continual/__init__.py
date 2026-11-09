"""Continual graph learning (CGL) experiments, kept separate from the main suite (src/experiments.py).

The HeteroGNN guard is trained on a stream of benchmarks (asb, injecagent, agentdojo; qwen2.5:1.5b traces only) one
after another, and re-tested on every benchmark after each stage. Run everything with scripts/run_continual.py.

    stream.py      traces, orders, leakage-checked splits
    strategies.py  finetune, joint, EWC, MAS, LwF, A-GEM, TWP, ER-GNN, replay
    run.py         training engine, fingerprints, per-cell results (CLI: python -m src.continual.run)
    metrics.py     AP / AF / BWT / FWT / ASR forgetting from the accuracy matrix
    report.py      REPORT.md, continual.json, report.html
"""
