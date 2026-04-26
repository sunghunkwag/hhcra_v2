# Hierarchical Hybrid Causal Reasoning Architecture (HHCRA) v2

A neuro-symbolic, physics-informed World Model architecture enabling full traversal of Pearl's Ladder of Causation (Observation, Intervention, Counterfactuals) within a unified pipeline.

## Overview

Current foundation models heavily rely on statistical correlation (Pattern Recognition) which inherently limits structural reasoning and causes hallucinations. **HHCRA v2** introduces a deeply integrated 3-layer architecture moving beyond traditional deep learning limitations:

1. **C-JEPA (Causal Joint-Embedding Predictive Architecture):** Extracts latent environmental variables directly from sparse observational data without manual labeling.
2. **GNN + Liquid Neural Networks (LNN):** Discovers Graph-theoretic causal structures (DAGs) and temporal dynamics via continuous-time ODEs.
3. **HRM (Hypothetical Reasoning Module):** A deterministic, neuro-symbolic engine conducting rigorous `d-separation` and `do-calculus` bounds.

## Verification & Benchmarks

The self-contained structural engine (`hhcra_v2.py`) automatically passes a mathematically rigorous 12-stage validation suite locally:

- **[L1 Observation]** Identifies conditional probabilities $P(Y|X)$.
- **[L2 Intervention]** Calculates post-intervention distributions $P(Y|do(X))$ using autonomously discovered causal structures.
- **[L3 Counterfactuals]** Solves theoretical queries $P(Y_{x'}|X=x, Y=y)$ through Abduction-Action-Prediction workflows.

## Run Inference Tests

```bash
python hhcra_v2.py
```
*(Dependencies: numpy, scipy)*

## Architecture Implications

By offloading logical inference from stochastic LLMs to a deterministic causal neuro-symbolic pipeline, HHCRA allows for statistically sound Meta-Reinforcement Learning, test-time adaptations, and mathematically guaranteed self-improvement cycles without error accumulation (Plateau).

## Author

Sung Hun Kwag (Independent AI Researcher)
