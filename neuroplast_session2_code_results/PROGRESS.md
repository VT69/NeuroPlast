# NeuroPlast — Progress Log

Living log, updated as work happens. Newest status at the top; decisions log
and detailed notes below.

## Session 2 (2026-09-27/28 night) — complete (~05:40)
Priorities (from the user): (1) full hybrid SNN+Transformer+sleep on the continual
benchmark; (2) significance of RQ5 sleep-vs-isolation; (3) DFA as the global term of the
hybrid rule; (4) longer task sequences. Colab (run by the user) covers seeds 4-10 of
rq1_v2 / cl_budget / cl_snn / cl_cnn, so those are not duplicated here.
- [x] (1) **RESULT — full hybrid (SNN T=4 + Transformer window 4 + sleep) on fetch3**, 3 seeds,
      lr 3e-4, 400k frames/task (`runs/continual_hybrid`, `results/continual.md`, `results/rq5_stats.md`):
      | method | ACC | FORGET | FWT | params |
      |---|---|---|---|---|
      | naive | 0.452 ± 0.016 | 0.536 | | 328k |
      | replay | 0.834 ± 0.015 | 0.024 | | 328k |
      | **sleep** | **0.873 ± 0.022** | **0.004** | +0.42 ± 0.24 | 328k |
      | isolation | 0.832 ± 0.037 | 0.006 | 0 (ref) | 880k |
      | sleep + wake homeostasis | 0.840 ± 0.051 (3 seeds) | 0.017 | | 328k |
      | sleep + wake stabilised STDP | 0.896 (1 seed) | 0.025 | | 328k |
      * Sleep vs naive: ACC p=2e-5, forgetting p=0.004. Sleep removes catastrophic forgetting
        in the integrated architecture.
      * Sleep vs replay: +0.039, Welch p=0.07. Every sleep seed (0.854-0.897) beats every
        replay seed (0.818-0.844), but with 3 vs 3 the exact permutation p can't go below 0.10.
        This is the only trunk where sleep trends ahead of replay (CNN/SNN trunks: ties).
      * Sleep vs isolation: +0.041 ACC (Welch p=0.19, bootstrap CI [+0.003, +0.083]); ACC per
        100k params 0.266 vs 0.095 (p<0.001); FWT +0.42 (p=0.10). Same story as the SNN trunk:
        clear per-parameter win, raw-ACC edge suggestive only.
      * Adding a wake-time local rule on top of sleep doesn't help in the continual setting:
        homeostasis 0.840 ± 0.051 (3 seeds) and stabilised STDP 0.896 (1 seed) vs 0.873 ± 0.022.
      Protocol history: `jobs/cl_hybrid.txt`: naive / replay / isolation / sleep / sleep+wake-STDP
      (stabilised 3-factor, alpha 0.03) on SNN T=4 + Transformer window 4, fetch3, 3 seeds.
      **First attempt stopped**: at 150k frames/task the full hybrid doesn't learn the
      tasks at all (isolation, i.e. fresh nets with nothing to forget, reached only 0.77 /
      0.34 / 0.25; entropy still ~1.4-1.5). The SNN trunk without the Transformer reaches
      ~0.9 in 150k. Forgetting metrics on unlearned tasks are meaningless, so those two runs
      are archived in `runs/continual_hybrid_150k_undertrained/` and the protocol is being
      fixed. Single-task probes on FetchObj-0, 400k frames (eval return / train return at 150k):
      | probe | eval @400k | train @150k |
      |---|---|---|
      | p1 default (window 4, lr 1e-3) | 0.74 | 0.24 |
      | p2 lr 3e-4 | 0.79 | 0.47 |
      | p3 window 1 (Transformer on 1 token) | 0.90 | 0.44 |
      | p4 no final LayerNorm | 0.80 | 0.38 |
      | p5 no final LN + lr 3e-4 | stopped | 0.21 @100k |
      | p6 window 2, no final LN | stopped | 0.49 |
      Diagnosis so far: the Transformer's final LayerNorm hands the heads unit-variance
      features (vs spike rates in [0,1]); with the heads' orthogonal init, 17% of head tanh
      units start saturated (0% without the LN). New option
      `mem_kwargs={final_norm: false}` (default unchanged).
      But removing the LN did not reliably speed learning (p4/p5), so it is *not* used.
      **Decision**: keep the architecture as designed (window-4 working memory, final LN) and
      change the protocol so the tasks can be learned: lr 3e-4, 400k frames/task
      (`configs/cl_hybrid_v2.yaml`). Relaunched -> `runs/continual_hybrid/` (15 runs).
      Scheduling decisions (~01:20): the sleep + wake-STDP arm (`_wtfs`) runs at ~60 fps under
      load (~3 h/run), and tonight's RQ1 controls showed homeostasis (not STDP) is the active
      ingredient. So `_wtfs` runs for seed 1 only, and seeds 1-3 get `_whomeo` (sleep + wake-time
      homeostasis, alpha 0) via `jobs/cl_hybrid2.txt`. The SNN fetch5 sweep was paused to free
      CPU for the hybrid; it resumes afterwards from the same worker if time allows.
      Takeaway worth keeping regardless of the continual result: **on these Markov tasks
      the Transformer makes PPO ~3x less sample-efficient** (window 1 learns much faster
      than window 4), so it only pays for itself on tasks that actually need memory.
- [x] (2) Done: the ACC advantage is NOT significant; see correction under RQ5 below.
- [~] (3) DFA as the global signal. Implemented + tested (5 DFA tests), sweep running
      (`jobs/rq1_dfa.txt`, `jobs/cl_dfa.txt`). Decisions:
      * DFA now works with MLP heads (random feedback into the heads' hidden layers too,
        per-row error routing for mixed-task replay batches). Removes session 1's
        confound (DFA had linear heads, backprop had MLP heads).
      * Probe (DoorKey-5x5, 150k frames, 1 seed): DFA *everywhere* (`dfa_scope=all`)
        had barely started (return 0.05) while DFA on the SNN synapses only
        (`dfa_scope=encoder`, heads by exact gradients) was at 0.95. **Correction**: with
        the full 300k budget and 3 seeds, DFA-everywhere solves 3/3 (eval 0.954 ± 0.004,
        AUC 0.54 vs backprop 0.66, p=0.30) and DFA-encoder 2/3 (0.68 ± 0.47, AUC 0.40,
        p=0.30). The 150k probe was just too short; DFA is slower, not broken.
      * **DoorKey-6x6 (encoder learning matters), 2-3 seeds so far:** DFA alone on the SNN
        fails 0/3 (eval 0.017; AUC 0.023 vs backprop 0.41, p=0.02), worse than a *frozen*
        encoder (2/3). DFA + stabilised three-factor STDP (alpha 0.03) solves 3/3 (eval
        0.946 ± 0.002, AUC 0.57; backprop 4/5, AUC 0.41). alpha 0.3 fails (0/2).
        Mechanism clue: DFA alone drives SNN firing rates up (0.16 -> 0.28-0.35 and
        rising: runaway again); with the stabilised rule rates stay ~0.15. **So the rescue
        may be the homeostasis, not STDP.** Controls (`jobs/rq1_dfa_ctrl.txt`) — COMPLETE:
        | SNN rule (DoorKey-6x6, 400k) | solved | AUC |
        |---|---|---|
        | backprop | 4/5 | 0.41 ± 0.24 |
        | backprop + homeostasis | 3/3 | 0.67 ± 0.12 (p=0.09 vs backprop) |
        | backprop + homeo + STDP (tfs0.03) | 4/5 | 0.49 ± 0.31 |
        | DFA | 1/5 | 0.08 ± 0.12 |
        | DFA + homeostasis | 3/3 | 0.63 ± 0.03 (p=0.0003 vs DFA) |
        | DFA + homeo + STDP (alpha 0.03) | 4/4 | 0.56 ± 0.08 |
        | DFA + homeo + random updates (same RMS) | 3/3 | 0.58 ± 0.12 |
        | DFA + homeo + STDP (alpha 0.3) | 0/4 | 0.21 ± 0.11 |
        **Conclusion: homeostasis is the active ingredient; STDP adds nothing.** DFA+homeo
        vs DFA+homeo+STDP p=0.14 (homeostasis alone scores higher); STDP vs random updates
        p=0.80; larger STDP (0.3) breaks learning even with homeostasis. With homeostasis,
        DFA (no weight transport) matches backprop on the SNN (0.63 vs 0.67, p=0.68).
        This refines last night's RQ1 answer: "stabilised STDP helps a bit" was really
        "homeostasis helps"; the Hebbian/STDP direction itself is no better than noise.
      * So the hybrid rule is implemented as CLAUDE.md defines it, on the SNN's
        synapses: dW_snn = alpha*STDP + beta*DFA (`dfa=true dfa_scope=encoder` +
        `stdp={...}`). The non-spiking readout heads keep exact gradients.
        `dfa_scope=all` stays as the fully backprop-free variant C.
- RQ4 with tonight's variants (`results/rq4.md`, compute = per-step cost x frames, relative
  to the CNN at 150k frames/task): continual ACC per unit training compute: CNN 0.98, SNN (A)
  + sleep 0.21, SNN + homeostasis + sleep 0.21, SNN + stabilised STDP + sleep 0.11, full hybrid
  + sleep ~0.08 (needs 400k frames/task), DFA + homeostasis + sleep 0.09 (ACC only 0.43 at
  150k/task: it learns each new task slowly — diagonal 0.75 / 0.60 / 0.24 — and still forgets
  0.15). DFA costs the same per step as backprop (73-78 vs 80 ms) but is much less
  sample-efficient in the continual setting, so the plain CNN stays the most robust per unit
  compute by ~4x over any spiking variant.
- [~] (4) `fetch5` suite added (same room, 5 distinct objects, random success 21% vs
      37% on fetch3; fetch3 unchanged). **CNN trunk, 3 seeds, complete** (`runs/continual5_cnn`):
      | method | ACC | FORGET | FWT | params |
      |---|---|---|---|---|
      | naive | 0.416 ± 0.098 | 0.612 | -0.87 | 229k |
      | replay | 0.926 ± 0.049 | 0.006 | -0.67 | 229k |
      | sleep | 0.928 ± 0.065 | 0.003 | -0.53 | 229k |
      | isolation | 0.976 ± 0.000 | 0 | 0 (ref) | 799k |
      * RQ2 holds at 5 tasks: sleep = replay (p=0.97), both ~zero forgetting, naive forgets 61%.
      * RQ5 shifts with length: isolation now has *higher* raw ACC (sleep - isolation = -0.047,
        Welch p=0.33, bootstrap CI [-0.122, -0.001]). Per parameter, shared weights still win
        3.3x (0.406 vs 0.122 ACC/100k params, p=0.003), and that ratio grows with task count.
      * The ACC gap is **plasticity, not forgetting**: return right after learning each task
        (R diagonal) is 0.98 for every isolation task but falls to 0.81-0.88 on task 3 for
        the shared nets (sleep 0.81, replay 0.83, naive 0.88), and FWT is negative
        everywhere. Shared nets don't forget; they learn later tasks less well.
      **SNN trunk** (trimmed for time to sleep / isolation / naive x seeds 1-2; `runs/continual5_snn`):
      | method | ACC | FORGET | FWT | params |
      |---|---|---|---|---|
      | naive | 0.405 ± 0.058 | 0.435 | | 229k |
      | sleep | 0.778 ± 0.007 | 0.034 | +0.34 ± 0.07 | 229k |
      | isolation | 0.475 ± 0.023 | 0.025 | 0 (ref) | 801k |
      Opposite of the CNN: fresh isolated SNNs mostly fail to learn a task in 150k frames
      (e.g. 0.32 / 0.22 / 0.39 on tasks 3-5), while the shared SNN with sleep learns them
      (0.80 / 0.67 / 0.88), so sleep beats isolation by +0.30 ACC (Welch p=0.02, n=2 per arm;
      treat as preliminary) with 3.5x fewer parameters. That is the forward transfer RQ5 asks
      about. Caveat: it's transfer of *sample efficiency*; with a bigger per-task budget,
      isolation would close some of the gap.
- Infra note: run `pytest` with `OMP_NUM_THREADS=1` while training jobs are running
  (4-thread torch + busy cores = spin-wait; 24 tests take 14 s single-threaded).

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

## Answers so far (after session 2; tables in `results/`, p = Welch t-test unless noted)
**RQ1: does a local Hebbian/STDP term alongside the global rule help?** *No. The local
plasticity that helps is homeostasis, not STDP.*
- Vanilla STDP hurts (runaway excitation; p≈0.02 on DoorKey-5x5). Stabilised STDP
  (covariance + homeostasis) seemed to help a little (session 1).
- Session 2 controls on DoorKey-6x6 separate the two parts. Homeostasis alone gives
  backprop 3/3 solves, AUC 0.67 vs 0.41 (p=0.09), and DFA 3/3 vs 1/5, AUC 0.63 vs 0.08
  (p=0.0003). STDP on top adds nothing (p=0.14) and is no better than random updates of
  the same size (p=0.80). Large STDP (alpha 0.3) breaks learning even with homeostasis.
- With the global term = DFA instead of backprop (no weight transport), the SNN matches
  backprop only when homeostasis is on (0.63 vs 0.67 AUC, p=0.68). DFA alone drives
  firing rates up and fails.
**RQ2: does sleep beat replay / EWC?** *Beats EWC and naive everywhere; ties replay on the
CNN/SNN trunks, trends ahead of it on the full hybrid.*
- CNN trunk: sleep 0.978 vs replay 0.970 (p=0.28). SNN trunk: 0.889 vs 0.814 (p=0.51).
- Full hybrid (SNN + Transformer): sleep 0.873 vs replay 0.834 (p=0.07; every sleep seed >
  every replay seed); forgetting 0.004 vs naive 0.54.
- 5 tasks (CNN): sleep 0.928 = replay 0.926 (p=0.97), both ~0 forgetting; naive 0.416.
- Small buffers: no significant difference. EWC: stability-plasticity trade-off.
- STDP inside sleep: vanilla catastrophic, stabilised harmless and useless.
**RQ3: SNN vs CNN on accuracy vs sparsity.** *Qualified yes* (unchanged; `results/rq3_doorkey6.md`):
equal at matched ops above ~33k ops/frame; below that only the SNN works (~2x fewer ops);
the big energy gap assumes neuromorphic per-op costs.
**RQ4: most robust per unit of compute?** *The plain CNN, ~4x ahead of any spiking variant*
(`results/rq4.md`, compute = per-step cost x frames). Among spiking variants: SNN + sleep
(0.21 per unit), then STDP (0.11), the full hybrid (0.08: the Transformer needs 400k
frames/task on these memory-free tasks) and DFA + homeostasis (0.09: same per-step cost as
backprop but much slower to learn each continual task). Local-only encoders (B) never solve.
**RQ5: shared weights vs isolation.** *Clear per-parameter win; raw-ACC edge NOT significant;
transfer depends on the trunk and shrinks with sequence length.*
- ACC per 100k params, shared vs isolation: CNN 0.50 vs 0.20; SNN 0.46 vs 0.17; hybrid 0.27
  vs 0.10; 5 tasks 0.41 vs 0.12 (all p<0.01; the gap grows with task count).
- Raw ACC: SNN 0.889 vs 0.833 (p=0.32; session-1 headline corrected); hybrid 0.873 vs 0.832
  (p=0.19); CNN tie; 5-task CNN isolation ahead (0.976 vs 0.928); 5-task SNN sleep far ahead
  (0.778 vs 0.475).
- Transfer depends on the trunk. It's positive where learning from scratch is slow: SNN +0.20
  (p=0.05), hybrid +0.42 (p=0.10), 5-task SNN +0.34, where shared sleep beats isolation
  outright (0.778 vs 0.475, p=0.02, n=2). It's ~0 on the 3-task CNN, and negative on the
  5-task CNN (the shared net learns later tasks less well: a plasticity cost).

## What to look at first when you're back
1. "Session 2" at the top and "Answers so far" above; then `results/continual.md` (all trunks,
   incl. the full hybrid), `results/rq5_stats.md`, `results/rq1_v2.md` (homeostasis/DFA
   controls), `results/rq4.md`.
2. **Biggest new finding:** homeostasis, not STDP, is the local-plasticity term that matters,
   and it's what lets a weight-transport-free global signal (DFA) match backprop. It's worth a
   dedicated follow-up: more seeds, a homeostasis-target sweep, and backprop+homeo vs
   backprop at 10 seeds (currently p=0.09).
3. **Full hybrid:** it works and sleep protects it (0.873, forgetting 0.004), but the Transformer
   costs ~3x sample efficiency on these Markov tasks. To make it earn its keep, test on a task
   that needs memory (MiniGrid-Memory) or a continual sequence with partial observability.
4. Colab seeds 4-10 (cl_snn / cl_cnn / cl_budget / rq1_v2): merge into `runs/`, then rerun
   `python scripts/analyze.py all` and `python scripts/rq5_stats.py`. The RQ5 SNN-trunk test
   (sleep vs isolation) is the one most likely to change.
5. All session-2 jobs finished. Not run for time: fetch5 SNN replay arm and seed 3 (the
   `jobs/cl5_snn.txt` lines were trimmed; regenerate with `python scripts/make_jobs.py`), and
   the sleep + wake-STDP hybrid arm for seeds 2-3 (skipped; homeostasis replaced it).

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
