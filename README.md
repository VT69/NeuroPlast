# NeuroPlast: A Self-Consolidating RL Agent

*Hybrid Spiking-Neural / Transformer Architecture • Hebbian-STDP + Gradient Descent • Sleep-Phase Memory Consolidation*

NeuroPlast is a deep RL agent that combines a spiking neural network (SNN)
perception core, a Transformer working-memory layer, a hybrid local
STDP + gradient-descent learning rule, and periodic offline "sleep"
consolidation against catastrophic forgetting. The research questions (RQ1–RQ5)
are listed in `CLAUDE.md`; **`PROGRESS.md` is the lab notebook** (what was run,
what was found, what failed, and what changed from the original plan), and
`results/` holds the aggregated tables and figures.

## Layout
```text
neuroplast/
  envs/            MiniGrid wrappers, one-hot obs, vector env; continual task suites (fetch3, natural3)
  models/snn/      LIF neuron + surrogate gradient
  models/encoders.py  CNN and topology-matched SNN encoders, op/energy accounting (RQ3)
  models/memory/   Transformer working memory over the last K frames
  models/heads/    actor / critic heads (multi-head for task-incremental CL)
  models/agent.py  encoder -> [memory] -> heads
  learning/stdp.py    trace STDP, eligibility traces, three-factor modulation
  learning/hybrid.py  dW = alpha * STDP + beta * backprop hook for PPO (RQ1)
  sleep/           sleep-phase consolidation (RQ2)
  baselines/       naive fine-tuning, EWC, replay (CLEAR-style), parameter isolation
  eval/            continual-learning metrics
train.py           single-file PPO (CleanRL-style); `train()` reused by everything
continual.py       task-sequence runner -> accuracy matrix, forgetting, transfer
configs/           one YAML per experiment
scripts/           experiment drivers (RQ3 sweep, job queue, analysis)
jobs/              job lists for scripts/run_queue.py (resumable sweeps)
runs/              run outputs (metrics.csv, final.pt, eval.json / results.json) — committed
results/           aggregated markdown tables + figures (scripts/analyze.py)
tests/             pytest unit tests (LIF, encoders, STDP, agent)
```

## Quick start
```bash
pip install -r requirements.txt
pytest tests/                                            # ~1-2 min on CPU
python train.py --config configs/cnn_doorkey.yaml        # CNN+PPO baseline, ~3 min
python train.py --config configs/hybrid_doorkey.yaml     # SNN+Transformer agent, ~10 min
python continual.py --suite fetch3 --method naive --config configs/cl_cnn.yaml
python scripts/run_queue.py jobs/cl_cnn.txt --procs 2    # a whole sweep, resumable
python scripts/analyze.py                                # -> results/*.md, results/figs/*.png
```
Override any config key from the command line: `--set seed=2 total_frames=100000`.

## Running on a modest local machine (CPU only, e.g. i3 + 12 GB RAM)
Everything here was developed CPU-only; every model is ~150–330k parameters and
every job uses one thread, so a 2-core/4-thread laptop runs 2–3 jobs at once.
- **Install the CPU-only PyTorch wheel first**: the default `pip install torch`
  on Linux/Windows can pull ~3 GB of CUDA libraries you don't need:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **If C: is full (Windows)**, create the virtualenv and caches on another drive:
  ```bat
  python -m venv D:\venvs\neuroplast
  D:\venvs\neuroplast\Scripts\activate
  set PIP_CACHE_DIR=D:\pip-cache
  set TORCH_HOME=D:\torch-cache
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
  and clone the repo onto D: or E: too (run outputs land in `runs/`).
- Use `--procs 2` (not 4) with `scripts/run_queue.py` on a dual-core CPU.
- RAM: each job uses < 1 GB.
- Google Colab also works (CPU runtime is enough), but it isn't needed.
