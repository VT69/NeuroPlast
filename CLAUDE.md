# NeuroPlast — Project Memory for Claude Code

## Project Brief
NeuroPlast is a hybrid deep RL agent combining a spiking neural network (SNN)
perception core, a Transformer working-memory layer, and a hybrid Hebbian/STDP
+ gradient-descent learning rule, with periodic "sleep" replay consolidation to
fight catastrophic forgetting. It's a solo capstone project built around a
rigorous empirical answer to the research questions below — a clean negative
result on any of them is a genuinely fine outcome, the goal is a real answer,
not a specific predetermined result. The full written proposal
(`NeuroPlast_Project_Plan.docx`) is included in this repo as background reading
if you want more context than this file gives — but this file, and the research
questions specifically, are what actually govern decisions.

## Core Research Questions — the actual target, everything else is negotiable
- **RQ1**: Does a local Hebbian/STDP term alongside gradient descent change sample efficiency or final return vs. backprop alone?
- **RQ2**: Does sleep-replay consolidation reduce catastrophic forgetting vs. plain experience replay and EWC?
- **RQ3**: Does an SNN encoder beat a dense CNN on accuracy-vs-spike-sparsity?
- **RQ4**: Of the architecture variants tried, which is most robust per unit of compute?
- **RQ5**: Vs. a parameter-isolation baseline (zero forgetting by construction, but linearly growing parameters) — does the shared-weight approach retain more performance per parameter, with transfer isolation structurally can't get? We're not trying to "beat" isolation on raw forgetting — that's impossible by construction. The real test is efficiency + transfer.

## Design Space — a starting point, not a spec to follow rigidly
Everything below is a reasonable hypothesis about how to answer the RQs, worked
out ahead of time so you're not starting from nothing. It is **not** a
contract. If you find a better architecture, a different learning rule, a
different environment, or a shortcut to real evidence on any RQ, take it —
just leave a note in `PROGRESS.md` on what you changed and why. Optimize for
genuine answers to the RQs, not fidelity to this plan.

- **Wake/sleep cycle**: online RL loop (Environment → SNN → Transformer →
  Actor/Critic → Action) alternating with an offline "sleep" phase that
  replays past experience and consolidates via local STDP + a distillation
  loss anchoring old-task outputs.
- **Hybrid learning rule**: `Δw = α·(local STDP term) + β·(global
  backprop/TD-error term)` on the SNN's synapses, α:β swept rather than fixed.
- **Three architecture variants worth trying**, roughly in order of how easy
  they are to get working:
  - **A — Differentiable end-to-end**: surrogate-gradient BPTT through the
    SNN, STDP as an auxiliary regularizer. Good first target — simplest path
    to a working agent.
  - **B — STDP frontend + backprop core**: SNN trained only by local STDP,
    frozen/detached before the Transformer.
  - **C — Fully spiking transformer**: three-factor local rule throughout, no
    backprop. Hardest, most novel, most likely to need real problem-solving.
- **Baselines worth having**: naive sequential fine-tuning, plain experience
  replay, EWC-style regularization, and parameter isolation (one frozen
  module per task) — these exist to give RQ2/RQ5 something concrete to
  compare against, not as busywork.

## Tech Stack — preferences, not hard limits
PyTorch, snnTorch (SNN library — good default, switch to SpikingJelly or
something else if you hit a real wall), Gymnasium + MiniGrid/BabyAI for
environments (grid-world tasks — navigate, pick up an object, open a door,
place it in a goal zone; not Atari/MinAtar), a CleanRL-style single-file
PPO/DQN as the RL base (easier to splice a spiking encoder into than
Stable-Baselines3's abstractions), Weights & Biases for tracking if useful.
Swap any of this out if it's genuinely getting in the way — note the swap and
why in `PROGRESS.md`.

## Repository Structure — rough shape, adjust freely since you're the only one working in it
```
models/snn/       SNN encoder, LIF neurons
learning/stdp.py  Eligibility traces, STDP update rule
models/memory/    Transformer working memory
models/heads/     Actor/critic heads
train.py          Main training loop
configs/          One config per experiment/variant
envs/             MiniGrid/BabyAI task suite
sleep/            Sleep-phase consolidation
baselines/        The baseline arms
eval/             Forgetting/transfer metrics, logging
notebooks/        Scratch/exploration
```

## Working Autonomously, Including Overnight
Sessions may run unattended for hours. Work through `PROGRESS.md` (create it
if it doesn't exist) as a living log, updated continuously as you go — not
just written at the end — so that if the session gets interrupted or runs out
of time, there's a clear record of what's done, what's in progress, what
failed and why, and what to look at first when I'm back.

- **Don't stop and wait for input at decision points.** Make the reasonable
  call yourself, log it in `PROGRESS.md`, and keep moving. There's no one
  here to answer a question overnight.
- **Don't stall on one blocker.** If something is taking too long or hitting
  a wall, note it, move to the next task, and come back to it later if there's
  time.
- **Checkpoint everything periodically** — models, configs, intermediate
  results — so nothing meaningful is lost if the session is interrupted.
- **You have full latitude on implementation details, libraries, and even
  architecture choices.** The research questions above are the fixed target;
  nothing else in this file is.

## Suggested Priority List for an Extended Session
Rough ordering, front-loaded toward de-risking the overall approach before
going deep on any one piece. Adjust as you learn more.

1. Repo scaffold, dependencies, and a smoke test confirming the environment setup actually works end to end.
2. A conventional CNN + PPO/DQN baseline, trained on 1–2 MiniGrid tasks to reasonable performance — this de-risks the RL loop itself before anything spiking gets added.
3. The SNN encoder module (unit-tested), plus a standalone SNN-vs-CNN comparison on accuracy vs. spike-sparsity — first real evidence toward RQ3.
4. The STDP/Hebbian eligibility-trace rule as its own tested module.
5. A first end-to-end hybrid agent (SNN + Transformer + RL heads) training on a single task — the first real test of whether the overall approach works at all.
6. If there's time left: a small continual task sequence (2–3 tasks) plus the naive-fine-tuning baseline, to get a first real forgetting measurement.
7. Keep `PROGRESS.md` current throughout, with a clear "what I'd look at first" note at whatever point the session ends.
