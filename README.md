# NeuroPlast: A Self-Consolidating RL Agent

*Hybrid Spiking-Neural / Transformer Architecture • Hebbian-STDP + Gradient Descent • Sleep-Phase Memory Consolidation*

## Overview
NeuroPlast is a deep reinforcement learning (RL) agent whose perception-to-action pipeline is modeled on biological brain learning and memory consolidation. Instead of standard dense networks trained end-to-end by backpropagation, it integrates three neuro-inspired mechanisms to tackle catastrophic forgetting, energy inefficiency, and the biological implausibility of backprop:

- **Spiking Neural Network (SNN) Core**: Handles low-level perception using discrete, event-driven spikes.
- **Transformer Working Memory**: Attends over a short window of recent spiking activity to provide context.
- **Hybrid Learning Rule**: Combines standard gradient descent with a local Hebbian/STDP update.
- **Sleep-Phase Consolidation**: Periodically replays sampled past experiences offline to prevent catastrophic forgetting.

## Research Questions
1. **RQ1**: Does adding a local Hebbian/STDP term alongside gradient descent change sample efficiency or final return?
2. **RQ2**: Does a periodic sleep-replay consolidation phase reduce catastrophic forgetting across a sequence of tasks?
3. **RQ3**: Does the SNN encoder offer a better accuracy-vs-estimated-energy trade-off (spike sparsity as a proxy) than a same-capacity dense CNN encoder?
4. **RQ4**: Which architecture variant offers the best continual-learning robustness per unit of compute?

## Architecture Variants
We explore three candidate architectures to balance biological plausibility with training stability:
- **Variant A (Differentiable End-to-End)**: SNN trained via surrogate gradients with an auxiliary STDP regularizer.
- **Variant B (STDP Front-End + Backprop Core)**: Frozen SNN front-end trained purely by local STDP; Transformer core trained by backprop.
- **Variant C (Fully Spiking Transformer)**: Fully local, reward-modulated three-factor learning rules end-to-end.

## Repository Structure
```text
.
├── src/
│   ├── models/         # SNN encoders, Transformer modules, Actor-Critic heads
│   ├── core/           # Hybrid learning algorithms (STDP + backprop mixing)
│   ├── memory/         # Experience buffer, sleep-phase replay mechanisms
│   ├── envs/           # Environment wrappers (MinAtar) and continual task suites
│   └── utils/          # Checkpointing, Google Drive sync, W&B logging
├── scripts/            # Training and evaluation entry points
├── notebooks/          # Exploratory analysis and plotting
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Setup & Dependencies
This project uses PyTorch, snnTorch, SpikingJelly, Gymnasium, and Weights & Biases for experiment tracking. It is designed to be runnable in constrained environments like Google Colab.

```bash
pip install -r requirements.txt
```
