# NeuroPlast — Progress Log

Living log, updated as work happens. Newest status at the top; decisions log
and detailed notes below.

## Current status (end of session 1, 2026-09-27 ~02:30)
All six items on the priority list are done, plus first-pass answers on all five RQs.
- [x] 1. Scaffold + deps + MiniGrid smoke test (21 unit tests, `pytest tests/`)
- [x] 2. CNN + PPO baseline: DoorKey-5x5 / LavaGapS5 / DoorKey-6x6, 100% eval success
- [x] 3. SNN encoder (tested vs snnTorch) + RQ3 accuracy-vs-sparsity frontier
- [x] 4. STDP / eligibility-trace / three-factor module (tested)
- [x] 5. Hybrid agent (SNN + Transformer + heads) solves DoorKey-5x5
- [x] 6. fetch3 continual suite: naive, EWC, replay, sleep, isolation; CNN and SNN trunks
- [x] Beyond the list: variant B (local-only encoder), variant C (DFA / e-prop, no backprop),
      stabilised STDP, sleep+STDP, small-buffer RQ2, compute benchmark (RQ4), Colab sweep notebook

## Answers so far (3 seeds unless noted; tables in `results/`, p = Welch t-test)
**RQ1: does a local STDP term alongside backprop help?** *Not detectably; naive STDP clearly hurts.*
- Vanilla trace STDP (two- or three-factor) makes learning worse on DoorKey-5x5 (never
  solves at alpha=1; p≈0.02 vs backprop). It's also worse than a random update of the
  same size. Mechanism: runaway excitation (firing rate 0.15 -> 0.5-0.77), because a
  static frame shown every timestep makes pairings overwhelmingly potentiating.
- Stabilised STDP (covariance-centred + local homeostasis) removes the harm. On
  DoorKey-6x6, where encoder learning matters (frozen encoder ~2x slower), the three-factor
  version at alpha=0.03 has the best mean (AUC 0.49 vs 0.41, 195k vs 254k frames to 0.9;
  5 seeds). But p=0.67, and the random-perturbation control is as good as backprop. So no
  evidence yet that it's STDP-specific. Outcomes on 6x6 are bimodal (a seed either
  solves or never does), so this needs ~10 seeds per arm: queued in the Colab notebook.
- Hebbian (unmodulated) STDP fails even when stabilised; local-only encoders
  (variant B) never solve (0/6 seeds).
**RQ2: does sleep consolidation beat replay / EWC?** *Beats EWC and naive clearly; ties replay.*
- CNN trunk: sleep ACC 0.978 ± 0.001 vs replay 0.970 ± 0.009 (p=0.28), EWC best 0.816,
  naive 0.555 (forgets 63%). Sleep and replay use a similar replay budget (538k vs 591k states).
- Small buffers: 200/task sleep 0.964 vs replay 0.946 (p=0.48); 50/task 0.846 vs 0.885
  (p=0.60). No significant difference either way. Qualitatively, sleep *recovers*
  previously forgotten performance in later phases, which replay never does.
- SNN trunk: sleep 0.889 ± 0.003 vs replay 0.814 ± 0.162 (p=0.51; one replay seed failed
  to learn task 0 at all). Forgetting 0.006 vs 0.023. Sleep is the most *reliable* arm.
- STDP inside sleep: vanilla is catastrophic (ACC 0.16-0.20, wipes every task). Stabilised
  is harmless but adds nothing (0.866-0.881 vs 0.889 without it). The benefit of sleep
  comes from offline distillation + rehearsal, not from the local rule.
- EWC: stability-plasticity trade-off. It never forgets, but it can't learn new tasks
  (lambda 10 -> 1000: ACC 0.82 -> 0.55).
**RQ3: does the SNN encoder beat the CNN on accuracy vs spike sparsity?** *Qualified yes.*
- At matched event-driven op counts above ~33k ops/frame: equal, or the CNN is slightly
  better (value R^2 0.995 vs 0.98 at ~30k ops).
- Below that, only the SNN works: the L1-penalised CNN dies at once past lambda 15. The SNN
  keeps teacher-level closed-loop return down to ~16k ops (~2x fewer).
- The big energy gap (~5-10x) comes from the accumulate-vs-multiply cost assumption
  (0.9 vs 4.6 pJ), i.e. neuromorphic hardware only. On CPU the SNN costs ~4x more to train.
- T=2 dominates T=4 dominates T=8. Caveat: other CNN sparsifiers not tried.
**RQ4: which variant is most robust per unit of compute?** *The plain CNN, by a wide margin;
among spiking variants, A (surrogate-gradient SNN) + sleep.* (`results/rq4.md`)
- Continual ACC per CPU-second of training: CNN 60, A 12.9, A+STDP 6.7.
- Variant B never solves. Variant C (DFA, no backprop) solves DoorKey-5x5 in 2/3 seeds. That
  matches backprop with the same linear-readout architecture (which also collapsed in 1/3),
  at the same cost as backprop.
- The full hybrid (A + Transformer) solves DoorKey-5x5, but none of these tasks needs memory,
  so the Transformer only adds cost here.
**RQ5: shared weights vs parameter isolation.**
- Per parameter, shared weights + sleep/replay win clearly: ACC/100k params 0.50 vs 0.20
  (CNN) and 0.46 vs 0.17 (SNN). Isolation uses 2.5x the parameters for 3 tasks, and that
  grows linearly with more tasks.
- Transfer: on the SNN trunk, shared weights transfer (FWT sleep +0.20 ± 0.08). Shared
  sleep reaches *higher* ACC than isolation (0.889 vs 0.833), because isolation's fresh
  networks sometimes fail to learn in 150k frames.
- On the CNN trunk there's no reliable forward transfer (FWT ≈ 0 ± 0.14). Each task is
  learned from scratch in ~60k frames anyway, so fetch3 is too easy to show transfer there.

## What to look at first when you're back
1. The "Answers so far" section above, then `results/continual.md`, `results/rq1_v2.md`,
   `results/rq3_doorkey6.md`, `results/rq4.md`, and the figures in `results/figs/`.
2. **The biggest open question is RQ1 on DoorKey-6x6, and it's seed-starved.** Run
   `notebooks/final_sweep_colab.ipynb` (CPU runtime, "Run all"; it resumes after
   disconnects) for seeds 4-10. A GPU/TPU won't speed it up: time goes to env stepping
   and tiny networks.
3. **Design issues I'd fix next:**
   - fetch3 is too easy for the CNN to show transfer; a harder/longer sequence
     (e.g. 5 tasks, or DoorKey/KeyCorridor variants) would test RQ5 transfer properly.
   - Memory never matters in these tasks. Use MiniGrid-Memory if the Transformer should
     earn its keep.
   - A CNN sparsifier stronger than L1 (k-WTA) would tighten the RQ3 caveat.
4. Things that went wrong (and fixes) are logged below: PPO collapse, OpenMP thread
   oversubscription, pkill self-kill, the RQ3 value-scale bug, and vanilla-STDP runaway
   excitation.

## Environment / how to run
- Dev box for this session: cloud container, 4 CPU cores, 15 GB RAM, **no GPU**.
  The user's machine (i3-7th gen, 12 GB RAM, C: full) has no GPU either, so
  everything is designed to run on CPU: all models are ~150k params and each
  run takes minutes to tens of minutes on 1 thread.
- `pip install -r requirements.txt` (on the local box, install the CPU-only
  torch wheel first, see README).
- `python train.py --config configs/<x>.yaml [--set key=value ...]`
- `pytest tests/`
- Outputs: `runs/<run_name>/{config.yaml,metrics.csv,final.pt,eval.json}`,
  stdout logs in `runs/logs/`. Run outputs are committed (small) since the
  container is ephemeral.

## Decisions log (what changed vs. CLAUDE.md plan, and why)
1. **Package layout**: removed the empty `src/` scaffold; code lives in the
   `neuroplast/` package with sub-packages mirroring CLAUDE.md's suggested
   layout (`models/snn`, `models/memory`, `models/heads`, `learning`, `envs`,
   `sleep`, `baselines`, `eval`) and top-level `train.py`. `.gitignore` used
   to ignore every `models/` dir (would have hidden `neuroplast/models/`) —
   fixed.
2. **Own LIF implementation instead of snnTorch** in the models: ~40 lines,
   exposes per-timestep pre/post spikes for STDP, no hidden state plumbing in
   a vectorized PPO loop. snnTorch is kept as a test-time cross-check.
   SpikingJelly/MinAtar/W&B dropped from requirements (not needed; CSV logs).
3. **Observations**: MiniGrid's 7x7x3 symbolic egocentric view, one-hot
   encoded to 20x7x7 binary. Binary input = already a spike pattern, so the
   SNN uses direct coding with no rate/latency encoder, and CNN and SNN see
   the exact same tensor (fair RQ3 comparison).
4. **Energy proxy for RQ3**: dense MACs for the CNN, event-driven SynOps
   (nonzero input x fan-out) for the SNN, 45nm energy (E_MAC 4.6 pJ, E_AC 0.9
   pJ). Also a sparsity-aware op count for the CNN so we don't compare a
   sparse SNN against a strawman dense CNN.
5. **Own small sync vector env** (same-step autoreset + frame history for the
   Transformer) instead of gymnasium's vector API (its default next-step
   autoreset complicates GAE bookkeeping).
6. **Variant C implemented as DFA / e-prop-style learning** (`neuroplast/learning/dfa.py`):
   every SNN layer gets the output error broadcast through a fixed random
   matrix (no weight transport, no backprop between layers); within a layer the
   update is presynaptic activity x surrogate derivative x broadcast error, a
   three-factor rule. Linear actor/critic readouts (their gradient is a local
   delta rule). No Transformer in C: I know of no credible local learning rule
   for attention, so "fully spiking transformer" was not attempted; C answers
   "can the SNN pathway learn with no backprop at all?". Verified in
   `tests/test_dfa.py` (with feedback = transposed readout, DFA == backprop).
7. **RQ3 metric**: behaviour cloning of a PPO teacher saturates at ~100% accuracy
   on MiniGrid (teachers are near-deterministic), so the DoorKey-6x6 run also
   regresses the teacher's value function and reports R^2, a graded target.
8. **Continual setup**: task-incremental (task ID given via embedding + per-task
   heads). Replay for PPO = CLEAR-style behaviour cloning on stored old-task states
   (PPO can't replay off-policy transitions through its own loss). Sleep = the
   same replay buffer and similar replay budget, but offline blocks + current-task
   self-distillation (+ STDP for SNN trunks). EWC Fisher normalised to mean 1 so
   lambda is comparable across CNN and SNN trunks.
9. **Continual sweeps use CNN and plain-SNN trunks (no Transformer)**: fetch3 is
   Markov given the frame, so memory adds cost (~4x) and no benefit there; the full
   SNN+Transformer hybrid is kept for single-task runs.


## Experiment log

### Infrastructure bugs hit (so they're not repeated)
- `pkill -f <pattern>` from the agent's shell kills that shell too when the pattern
  appears anywhere in the command line (happened twice). Look up PIDs first, kill by PID.
- OpenMP spin-waiting: jobs that ran torch ops before `torch.set_num_threads(1)`
  used 4 threads each; with ~8 processes on 4 cores they made almost no progress
  (a 5-second evaluation took >4 CPU-minutes). Fixed: `continual.py` / `rq3_encoders.py`
  pin threads at start, and `scripts/run_queue.py` sets `OMP_NUM_THREADS=1`.

### Priority 2 — CNN + PPO baselines (`scripts/run_cnn_baselines.sh`)
PPO: 16 envs x 128 steps, Adam 1e-3 (annealed), 4 epochs, mb 256, target-KL 0.03.
| run | eval return (200 held-out eps) | success | frames to 0.9 train return |
|---|---|---|---|
| cnn_doorkey5_s1 | 0.964 | 1.00 | 55k |
| cnn_doorkey5_s2 | 0.965 | 1.00 | 68k |
| cnn_lavagap5_s1 | 0.946 | 1.00 | ~30k |
| cnn_lavagap5_s2 | 0.946 | 1.00 | ~30k |

**Failure found + fixed**: first LavaGap run (constant lr 1e-3, no KL stop)
solved the task by 30k frames, stayed at 0.95 for 250k frames, then
collapsed to 0 return at ~270k. Cause: entropy fell to ~3e-4 and the value
function became near-perfect, so per-minibatch advantage normalisation
blew noise up to unit-scale updates. Fix: LR annealing + target-KL early
stopping, now defaults in `train.py`. All results above are with the fix.

### Priority 5 — first end-to-end hybrid agent (`configs/hybrid_doorkey.yaml`)
SNN encoder (4 LIF layers, T=4, surrogate-gradient BPTT, learnable leak) ->
1-layer Transformer over the last 4 frames -> actor/critic.
- hybrid_doorkey5_s1: eval return 0.959, success 1.00 at 400k frames.
- Sample efficiency: ~125k frames to reach 0.9 train return vs 55-68k for the CNN
  (≈2x slower), wall-clock ~780 fps vs ~2100 fps (1 CPU thread each).
- Speed trick (logged as a design decision): **feature cache** — older frames in
  the Transformer window reuse encoder features computed at rollout time;
  gradients reach the encoder only via the newest frame (like R2D2's stored
  recurrent state). 116 -> 543 fps. Tested equal to full re-encoding in eval
  mode (`tests/test_agent.py`).

### Priority 6 — continual suite + naive fine-tuning (`continual.py`)
`fetch3`: 8x8 room (6x6 interior) with a red ball, green key and blue box. Task k =
pick up object k; wrong pickup ends the episode with 0. Task ID given
(task embedding + per-task actor/critic heads). 150k frames per task.
First run (CNN, naive, seed 1): R matrix (return on task j after training task i):
```
after t0: [0.977, 0.221, 0.122]
after t1: [0.324, 0.979, 0.148]   <- task 0 catastrophically forgotten
after t2: [0.649, 0.776, 0.945]
ACC 0.790  BWT -0.266
```
Notes: task 1 was learned faster than task 0 (forward transfer), task 2 slower
(0.25 at 51k frames vs 0.93 for task 0) — possible loss of plasticity; worth
checking across seeds.

### RQ3 (in progress) — first observations on DoorKey-5x5 behaviour cloning
- The task saturates: the CNN hits 100% test accuracy even with activation rates
  pushed to ~1-5% by the L1 penalty. So the accuracy axis is uninformative on
  this dataset -> training a DoorKey-6x6 teacher for a harder dataset.
  (DoorKey-8x8 was tried first: vanilla PPO got 0 return after 1.5M frames.)
- Honest early caveat: the SNN's energy advantage over the dense CNN
  (~60-85 nJ vs 1.65 uJ per frame) comes mostly from counting accumulates
  (0.9 pJ) instead of MACs (4.6 pJ). An L1-sparsified CNN executes *fewer* ops
  (20-28k) than the unpenalised SNN (68-94k SynOps); its event-driven energy
  (~100 nJ) is within ~1.5x of the SNN. And the SNN degrades much faster than the
  CNN under a sparsity penalty (lam 0.03: SNN T=2 acc 0.92 vs CNN 1.00).

### RQ3 — bug in my own metric, caught and fixed
First DoorKey-6x6 value-regression run gave the SNN (T=4) *negative* test R^2 at
small sparsity penalties (-0.5 to -3) while action accuracy stayed >99%. Diagnosis:
train R^2 was equally negative (underfitting, not a generalisation gap); the raw
value target has variance ~0.02, so its MSE gradient was dwarfed by the spike
penalty. Fix: z-score the value target. Re-test at lam=0.03: SNN T=4 R^2 0.97 at
29k ops/frame vs CNN (lam 0.01) 0.99 at 198k ops/frame. The whole 6x6 sweep was
re-run with the fix; the discarded run is not used anywhere.

### RQ1 — first seed: vanilla STDP hurts, via runaway excitation (22:50)
SNN encoder + PPO on DoorKey-5x5, 300k frames (`jobs/rq1.txt`, seed 1 so far):
| arm | frames to 0.9 | final eval |
|---|---|---|
| bp (backprop only) | 47k | 0.96 |
| frozen (encoder weights never trained) | 49k | 0.96 |
| rand1 (random update, same RMS as STDP at alpha=1) | 66k | 0.96 |
| tf0.1 (three-factor STDP, alpha=0.1) | 154k | 0.78 |
| tf1 / heb1 / tf1_local | never | 0.2-0.3 |
Two lessons:
1. **DoorKey-5x5 doesn't need a learned encoder**: a frozen random SNN + MLP heads
   learns as fast as full backprop. So on this task RQ1 can only detect *harm*,
   not help. -> probing linear heads (`frozen_lin` vs `bp_linear`), where the
   encoder must do the representational work.
2. **Mechanism of the harm**: encoder firing rate climbs from ~0.15 (bp) to 0.5
   (tf1) and 0.77 (heb1): with a static frame presented every step, pre->post
   pairings are overwhelmingly potentiating; synaptic scaling fixes norms but
   not rates. STDP is *worse than same-size random noise* because its drift is
   consistent in sign (grows ~n) while noise random-walks (~sqrt n).
   -> added covariance-centred STDP + local bias homeostasis (target rate 0.15) to
   `HybridSTDP` (`center`, `homeo`); probing now. The vanilla arms stay in the
   sweep as the "naive STDP" reference.

### Continual (CNN trunk, fetch3), 2 seeds so far (22:45)
| method | ACC | FORGET | FWT | replayed states |
|---|---|---|---|---|
| naive | 0.555 ± 0.019 | 0.630 | -0.08 ± 0.36 | 0 |
| EWC lam=10 | 0.870 ± 0.107 | ~0 | -1.67 | 0 |
| EWC lam=100 / 1000 (1 seed) | 0.64 / 0.54 | ~0 | -2.9 / -3.6 | 0 |
| replay (CLEAR-style) | 0.975 ± 0.002 | 0.003 | -0.11 ± 0.15 | 589k |
| **sleep** | **0.978 ± 0.001** | 0.001 | **+0.06 ± 0.01** | 538k |
| isolation | 0.979 ± 0.001 | 0 | 0 (reference) | 0 (2.5x params) |
Reading so far: replay and sleep both retain everything with a 5000-state
buffer (ceiling), so the RQ2 comparison moves to small buffers
(`jobs/cl_budget.txt`, 50 and 200 states/task). EWC shows the classic
stability/plasticity trade-off: it never forgets but can't learn new tasks
(the Fisher from 3 tasks covers most of a 194k-param net). Sleep's FWT is
positive and replay's negative: consolidating offline doesn't tax learning of
the new task, interleaved replay does. Needs the third seed + small buffers.

### RQ3 — answer so far (DoorKey-6x6 teacher, BC + value regression, 2 seeds per point)
Table: `results/rq3_doorkey6.md`; figure: `results/figs/rq3_doorkey6.png`.
Same topology, same input tensor; CNN sparsified with an L1 activation penalty,
SNN with a spike-rate penalty (lam sweeps), SNN T in {2, 4}.
1. **Ops (hardware-agnostic event-driven count)**: where both work, they're
   equivalent or the CNN is slightly better: ~80k ops: CNN R^2 0.996 / SNN T=2 0.993;
   ~30k ops: CNN (lam 10) R^2 0.995, acc 0.998 vs SNN T=2 (lam .01) 0.986 / 0.998.
2. **Floor**: the L1-penalised CNN can't go below ~33k ops (lam 15 degrades, lam >= 20
   kills the whole net at once), while the SNN keeps working to ~16k ops (acc
   0.97-0.99, closed-loop return 0.97 = teacher level, value R^2 0.91-0.95), i.e.
   **~2x fewer ops at the low end**. Caveat: other CNN sparsifiers (k-WTA,
   learned thresholds) might push the CNN floor lower; not tested.
3. **Energy**: on top of that, the SNN's ~5x per-op advantage is purely the
   accumulate (0.9 pJ) vs multiply-accumulate (4.6 pJ) assumption, which only holds on
   neuromorphic / event-driven hardware. On a CPU/GPU the SNN is *more* expensive:
   T timesteps => ~4x the training time per step at T=4 (`results/compute.md`).
4. T=2 dominates T=4 at every sparsity level (fewer timesteps, same accuracy);
   T=8 (easy-dataset sweep) dominated by both.
5. On an easy dataset (DoorKey-5x5 BC) everything saturates at 100%; the
   comparison only becomes informative with a graded target (value R^2).
**Verdict**: a qualified yes. The SNN reaches a lower-ops regime the (L1) CNN can't,
at ~1-3% accuracy cost, and wins big on energy *if* you grant neuromorphic
per-op costs; at matched op counts above ~33k it has no accuracy advantage.

### RQ2 small buffers + SNN trunk, interim (~01:00)
Small-buffer sweep (CNN trunk, 3 seeds, complete; `runs/continual_budget`):
| buffer/task | sleep ACC | replay ACC | p | sleep FORGET | replay FORGET |
|---|---|---|---|---|---|
| 200 | 0.964 ± 0.009 | 0.946 ± 0.035 | 0.48 | 0.017 | 0.042 |
| 50  | 0.846 ± 0.086 | 0.885 ± 0.081 | ~0.6 | 0.193 | 0.131 |
No significant difference either way. Qualitative: under sleep, task-0 return
dips then *recovers* (0.87 -> 0.95 at buf 200) as later sleep phases re-consolidate
it; replay only ever declines. At 50 states the concentrated sleep blocks likely
over-fit the tiny buffer.

SNN trunk (seed 1): **vanilla STDP inside sleep is catastrophic** (sleep_stdp ACC
0.16, dream-only 0.20 vs sleep without STDP 0.89): the same runaway-excitation
failure as RQ1, now wiping all tasks including the current one. Seeds 2-3 of those
two arms were skipped (locks with pid 1 in runs/continual_snn, jobs file annotated);
stabilised-STDP sleep arms (covariance + homeostasis, alpha 0.01 / 0.003) queued in
`jobs/cl_snn2.txt`. Also on the SNN trunk forward transfer is *positive* (naive/replay
FWT ~ +0.4): shared replay ACC 0.90 vs isolation 0.84 (tasks 2-3 learn faster with
shared weights in 150k frames) — the transfer RQ5 asks about, absent on the CNN trunk.
