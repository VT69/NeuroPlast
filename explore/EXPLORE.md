# EXPLORE: open-ended follow-up to the frozen capstone

Branch `explore`, folder `explore/` only. The capstone results (`runs/`, `results/`, `paper/`, `presentation/`,
`demo/`, `main`) are frozen and are only *read* here. New code lives in `explore/np_explore/` and extends
`neuroplast` without modifying it; new run outputs go to `explore/runs/`. Seeds: pilots use 101-110,
confirmation uses 201-210. Neither range was used by any capstone run, which used seeds 1-10.

Claim labels as in the paper: **[pilot]** is exploratory (1-2 seeds, never a finding); **[confirmed]** is only for
Phase C runs done exactly as pre-registered below; **[lit]** is literature; **[interp]** is interpretation.

## Budget (about 30 core-hours on this 4-core machine, one thread per job)
| phase | planned core-h | what |
|---|---|---|
| A | 0 | this file |
| B pilots | about 9 | round 1 (about 4): PackNet, LwF, matched-memory replay, CueFirst memory benchmark, int8 audit; round 2 (about 5) depends on round 1 |
| C confirmation | about 17 | 1-2 pre-registered comparisons, 8-10 fresh seeds |
| reserve | about 4 | reruns, failures |

Spent so far: see the ledger at the end (computed by `explore/budget.py` from each run's logged wall time).

## Phase A: candidate ideas, ranked by expected information per core-hour

Where the frozen results leave room ([ours], from `PROGRESS.md` / `paper/`):
- At the default buffer (5,000 states/task), isolation is more accuracy-per-MB than shared weights.
- With 200 states/task, shared weights reach a similar mean accuracy in about half the memory (3 seeds, wide CI).
- The paper's own limitation: "isolation uses one full network per task, not PackNet-style masks, which would be
  cheaper and a stronger baseline".
- At 200 states/task the buffer is only 0.11 MB; the network (0.78 MB) dominates. So at small memory the question
  is how close to isolation one network can get with little or no stored data.
- The Transformer question is untested: no benchmark both required memory and was learnable.

### 1. PackNet-style isolation inside one network (stronger isolation baseline)
- **Motivation.** [lit] Mallya & Lazebnik (2018): prune the current task's weights by magnitude, retrain, freeze;
  the freed weights serve the next task; storage is one network plus per-weight task masks. [ours] This is the
  paper's stated missing baseline, and it could overturn the "shared weights win with small buffers" result.
- **Prediction.** CNN fetch3/fetch5: PackNet reaches isolation's ACC within 0.03, at about 1/T of isolation's memory
  (one network + a 2-3 bit owner mask per trunk weight). It then dominates small-buffer replay/sleep at matched
  total memory, and the gap grows with T.
- **Matched control.** Isolation (same frames per task, same seeds); replay with a buffer whose bytes equal
  PackNet's mask overhead (matched total memory). Frames: pruning and retraining happen *inside* each task's
  frame budget (train 75%, prune, retrain 25%).
- **Cheapest pilot.** CNN fetch3 and fetch5, seeds 101-102, 150k frames/task (about 0.2 core-h per run).
- **Kill criterion.** Pilot ACC more than 0.05 below the frozen isolation mean on the same suite (fetch3 0.979,
  fetch5 0.976). If the first capacity split fails, allow one re-split; after that, kill.

### 2. Buffer-free consolidation: distillation on current-task states (LwF-style)
- **Motivation.** [lit] Learning without Forgetting (Li & Hoiem, 2016/2017): distil a frozen copy's outputs on
  *new-task* inputs; no stored data. [ours] In fetch, every task samples the same room distribution, so current
  states may cover old tasks' states. The buffer is what our memory analysis charges shared methods for.
- **Prediction.** ACC between naive (0.555, CNN fetch3) and replay (0.970); I expect 0.85-0.95. It stays below
  replay because the current policy rarely visits the states where an old task's decision happens (next to *its*
  object).
- **Matched control.** CLEAR-style replay with the same loss (policy KL + value MSE), the same 128 distilled
  samples per PPO minibatch (replayed-sample budget matched), and a buffer sized to LwF's *peak* memory (the
  snapshot network's bytes). Persistent memory: LwF = 1 network (equal to naive); peak = 2 networks.
- **Cheapest pilot.** CNN fetch3 (+ fetch5), seeds 101-102.
- **Kill criterion.** Pilot fetch3 ACC below replay@50 states/task (0.885, frozen). A 50-state buffer costs 28 KB,
  so LwF would then have no memory advantage worth the transient snapshot.

### 3. int8 storage audit of the memory frontier (analysis only, no training)
- **Motivation.** [ours] Memory was counted as fp32 parameters. Frozen isolated networks (and a finished shared
  network) could be stored in int8, which divides parameter memory by 4 but barely shrinks the buffer (147 of 188
  B/state are the uint8 observation). This could move the frontier.
- **Prediction.** Post-training symmetric per-tensor int8 weight quantisation costs less than 0.01 ACC for the CNN;
  the SNN is more sensitive (thresholds). Qualitative memory conclusions unchanged, but isolation's advantage at
  the default buffer grows.
- **Control.** The same checkpoint in fp32 (paired).
- **Pilot = the whole thing.** Frozen checkpoints `runs/continual*/…/task*/final.pt`, 100 episodes per task, about
  0.3 core-h.
- **Kill criterion.** None needed (cheap audit); report the drop.

### 4. A memory benchmark that works: cue seen on the way (CueFirst-S11)
- **Motivation.** [ours] MiniGrid-MemoryS11/S13 failed because agents never turned back to look at the cue. In
  MiniGrid's MemoryEnv the agent starts at a random x facing the hallway, so the cue is usually behind it. Starting
  it next to the cue (x = 1, facing the hallway) makes the cue visible at step 0 and invisible at the junction (8
  steps later).
- **Prediction.** Memoryless CNN stays at chance (0.5 success); a 12-frame stack learns (>0.8); then Transformer vs
  GRU vs frame stack over the same 12-frame window (CNN encoder, for speed).
- **Matched control.** Memoryless agent (chance check); frame stack and GRU over the same window as the Transformer;
  same learning rate and frames for all arms.
- **Cheapest pilot.** fs1, fs12, GRU-12, Transformer-12; seed 101; 1M frames.
- **Kill criterion.** fs12 success below 0.8 at 1M frames (benchmark not learnable), or fs1 above 0.65 (does not
  require memory).

### 5. Sleep vs replay dose-response at low replay budgets
- **Motivation.** [ours] Sleep equalled replay at a matched budget of about 0.5-1.4M replayed states. Offline
  scheduling might matter only when replay is scarce.
- **Prediction.** Both degrade as the budget shrinks; no large difference (weak prior).
- **Matched control.** The same exact replayed-sample budget for both (sleep `replay_budget`, replay minibatch
  size).
- **Cheapest pilot.** CNN fetch3, budgets at 10% and 2% of replay's usual count, seed 101 (4 runs).
- **Kill criterion.** |sleep − replay| < 0.03 at both budgets.

### 6. Making DFA + homeostasis reliable
- **Motivation.** [ours] The only surviving positive: homeostasis speeds up DFA (AUC Holm p = 0.017), but only 7/10
  runs solve DoorKey-6x6. PROGRESS.md lists a target-rate sweep as future work.
- **Prediction.** A lower target rate (0.05) or a higher one (0.3) changes the solve rate; uncertain direction.
- **Control.** DFA + homeostasis at the pre-registered target 0.15, same seeds.
- **Cheapest pilot.** 3 targets × 3 seeds (101-103), 400k frames, about 2 core-h.
- **Kill criterion.** No target solves more seeds than 0.15 does.

### 7. Stateful SNN (membrane carried across the window) as working memory vs GRU
- **Motivation.** [ours] The SNN resets its membrane every frame, so it has no memory. [lit]
  §38.5 of the literature doc.
- **Needs** a working memory benchmark (idea 4) first, and SNN runs are about 5× slower. Lowest information per
  core-hour tonight.
- **Kill criterion.** Only pilot if idea 4's benchmark is valid and budget remains.

### Folded in, not separate
- **Minimal-buffer curve** (replay with about 10-60 states/task) appears as the matched-memory control in ideas 1-2.
- **Lossless buffer packing** (9 bits/cell, 147 → 56 B) is arithmetic, reported with idea 3.

**Order of work:** round 1 runs ideas 1, 2, 3 and 4 together (they share the queue); round 2 is chosen from their
outcomes.

## Phase B: pilot log
(filled in as pilots finish; failures stay in)

## Phase C: pre-registration
(written and committed before any confirmation run starts)

## Phase D: report

## Budget ledger
