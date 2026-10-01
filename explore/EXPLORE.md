# EXPLORE: open-ended follow-up to the frozen capstone

Branch `explore`, folder `explore/` only. The capstone results (`runs/`, `results/`, `paper/`, `presentation/`,
`demo/`, `main`) are frozen and are only *read* here. New code lives in `explore/np_explore/` and extends
`neuroplast` without modifying it; new run outputs go to `explore/runs/`. Seeds: pilots use 101-110,
confirmation uses 201-210. Neither range was used by any capstone run, which used seeds 1-10.

Claim labels as in the paper: **[pilot]** is exploratory (1-2 seeds, never a finding); **[confirmed]** is only for
Phase C runs done exactly as pre-registered below; **[lit]** is literature; **[interp]** is interpretation.

## What to look at first (live; updated 04:20)
- **Phase C is complete** (62/62 runs; results under "Phase C results").
  - C3 is confirmed: with scarce replay (~14k samples), sleep forgets much more than interleaved replay.
  - C1: LwF-int8 is not significantly different from replay@973 at half the memory, and it is about 3 points below
    full isolation.
  - C2 (SNN): LwF forgets slightly more than replay.
- **Idea 8 overturned the LwF story.** Isolation shrunk to LwF's parameter count reached 0.972 on 2 pilot seeds, so LwF
  has no memory advantage. C4 (pre-registered) confirms this with 10 seeds; the idea-8b width sweep runs alongside.
- Three container restarts; killed runs were rerun from scratch with the same seeds (see the ledger).

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
(filled in as pilots finish; failures stay in. Frozen references are means of the capstone's seeds 1-3.)

### Idea 3: int8 storage audit (done; 0.03 core-h). [pilot, paired]
27 frozen checkpoints were checked: CNN and SNN fetch3 (isolation, sleep, replay, sleep@200) and CNN fetch5 (isolation,
sleep). Each network was evaluated in fp32 and with every weight matrix in symmetric per-tensor int8, on the same 100
held-out episodes with the same RNG state. The fp32 − int8 ACC difference has mean −0.0003 (SD 0.006, range −0.015 to
+0.012; one-sample t p = 0.78, n = 27). int8 storage costs no detectable accuracy here.
- **Consequence for the memory frontier** [interp, arithmetic]:
  - Parameters shrink 4× for every method. Buffer states barely shrink: 188 → 157 B with int8 targets, or → 66 B if the
    uint8 view is also bit-packed (object 4 bits + colour 3 + state 2 = 9 bits/cell).
  - CNN fetch3 with int8 parameters and packed states:
    | arm | memory | ACC |
    |---|---|---|
    | isolation | 0.48 MB | 0.979 |
    | sleep@200 | 0.23 MB | 0.964 |
    | sleep@5000 | 1.18 MB | 0.978 |
  - The ordering is unchanged. Isolation still wins at the default buffer and loses at 200 states/task. So the paper's
    memory conclusion is robust to storage precision; it just moves every point left.
- Files: `explore/quant/quant_audit.json`, `explore/quant_audit.py`.

### Idea 4: CueFirst-S11 memory benchmark. KILLED (kill criterion met). [pilot, seed 101, 1M frames]
| agent (CNN encoder, lr 3e-4) | eval success | episode length |
|---|---|---|
| memoryless (fs1) | 0.49 | 13.9 |
| 12-frame stack | 0.51 | 16.4 |
| GRU over 12 frames | 0.49 | 14.4 |
| Transformer over 12 frames | 0.485 | 14.0 |
- The benchmark does require memory: the memoryless agent is at chance, and the cue is visible at step 0 and invisible
  at the junction (checked).
- But no memory agent learned to use the cue in 1M frames. All of them learn to walk straight to the junction and guess,
  which already earns 50% of the reward. Seeing the cue on the way removed the exploration problem, but not the
  credit-assignment problem.
- Kill criterion (fs12 < 0.8 at 1M) met, so no further runs. The Transformer question stays open.
- [interp] A next attempt would need a denser signal, e.g. a penalty for the wrong choice instead of 0, or a
  curriculum with a short corridor. That is a benchmark-design project, not a pilot.

### Idea 1: PackNet (one-shot pruning at 75% of each task). [pilot]
- **fetch3 s101: ACC 0.681.** Task 0 learned (return 0.98), then collapsed right after its one-shot prune of 2/3 of the
  encoder. Entropy fell from 0.25 to 0.06 and the near-deterministic policy got stuck in timeouts, with no reward
  signal to recover. Tasks 1 and 2 were fine (0.98, 0.87). Forgetting 0 by construction.
- **fetch5 s101: ACC 0.872, FORGET 0.000.** Tasks 1-4 reached 0.94-0.98, but the last task only 0.51. With a 1/5
  share and frozen earlier weights, late tasks lose plasticity.
- Both are below the kill line (isolation − 0.05). The one allowed change is gradual magnitude pruning (Zhu & Gupta
  2017): 5 steps between 50% and 75% of the task's updates, with the same final split and the same frames. It runs in
  round 2; if it also misses the line, PackNet is killed.

### Idea 2: LwF (buffer-free distillation on current-task states). [pilot]
- **fetch3 s101: ACC 0.977, FORGET 0.0006**, with 595k distilled samples (replay uses about 0.55-0.59M). R rows:
  0.979 / 0.978, 0.979 / 0.978, 0.978, 0.975.
- This beats my prediction (0.85-0.95). It is on par with replay@5000 (0.970) and isolation (0.979), and above
  replay@200 (0.946) and replay@50 (0.885), while storing no states.
- Persistent memory 0.78 MB (the same as naive). Peak memory during training 1.55 MB (student + snapshot).
- **Round 2, s102: ACC 0.970, FORGET 0.000.** Replicates s101.
- **int8 teacher, s101: ACC 0.976** (fp32 teacher 0.977). The int8 snapshot costs nothing detectable, as the audit
  predicted.
- **fetch5 s101: ACC 0.969; s102: ACC 0.812, FORGET 0.021.** In s102, task 2 was never learned (0.33 right after its
  own training), while the other four reached 0.91-0.98. That is a plasticity failure, not forgetting: distilling
  towards the snapshot seems to hold the shared encoder too tightly on some seeds. [interp; one seed] **int8 teacher,
  s101: 0.963.**
- **SNN fetch3 s101: ACC 0.825, FORGET 0.056** vs replay@200 (the pilot control) 0.797, FORGET 0.068. The SNN is
  weaker at 150k frames/task anyway (diagonal 0.84-0.88 for both methods).
- **Matched-memory replay controls (s101).** Buffer = one fp32 network's bytes / (T × 188 B):
  - CNN fetch3 replay@1378: ACC 0.959 (LwF 0.977).
  - CNN fetch5 replay@973: ACC 0.950 (LwF 0.969 / 0.812; int8 0.963).
  - CNN fetch5 replay@57 (bytes of LwF's int8 snapshot): ACC 0.623, FORGET 0.435. At LwF-int8's peak memory, replay
    barely works.
- Distilled samples match replay's (fetch3 595k vs 0.55-0.59M; fetch5 1.19M, the same per-minibatch rule).
- **Decision.** Kill criterion (fetch3 ACC < replay@50's 0.885) not met: the fetch3 pilots are 0.970-0.977. The
  fetch5 s102 failure is the main risk; it went into the pre-registration as a descriptive "never-learned tasks"
  count. → Phase C (C1, C2).

### Idea 1, round 2: PackNet with gradual pruning. KILLED. [pilot]
- 5 pruning steps between 50% and 75% of each task's updates; same final 1/T split and frames. Kept fractions were
  exactly 1/3 (fetch3) and 1/5 (fetch5) per task.
  | run | ACC | diagonal (return right after each task) |
  |---|---|---|
  | fetch3 s101 | 0.960 | 0.98 / 0.98 / 0.92 |
  | fetch3 s102 | 0.857 | 0.98 / 0.98 / 0.62 |
  | fetch5 s101 | 0.878 | 0.97 / 0.97 / 0.97 / 0.84 / 0.59 |
- Gradual pruning fixed the task-0 collapse of the one-shot version, but the last task still fails to learn with its
  1/T share on top of frozen earlier weights. Pilot mean 0.908 on fetch3 and 0.878 on fetch5 are both below the kill line
  (isolation − 0.05 = 0.929 / 0.926), so it is killed. No further tuning.
- [interp] Isolation-inside-one-network pays for its low memory (0.81 MB fetch3) with late-task plasticity. The
  capstone's isolation baseline (separate small networks) does not have this problem, so it stays the right
  reference.

### Idea 5: sleep vs replay dose-response at low replay budgets. [pilot, CNN fetch3 s101] → Phase C3
| replayed samples (≈ % of usual) | sleep | replay (batch per minibatch) |
|---|---|---|
| ~595k (100%, frozen seeds 1-3) | 0.978 | 0.970 |
| ~55-61k (~10%) | 0.841 (FORGET 0.206; rb 55,040) | 0.935 (b=13, 60,606 samples) |
| ~11-14k (~2%) | 0.585 (FORGET 0.591; rb 10,880) | 0.957 (b=3, 13,980 samples; FORGET 0.001) |
- The direction reverses: at the full budget sleep ≥ replay (the capstone's result), but with scarce replay, interleaving
  a few old states into every PPO minibatch protects far better than the same number spent in offline sleep phases.
- Kill criterion (|sleep − replay| < 0.03 at both budgets) not met: the gaps are 0.094 and 0.372.
- This was not a kill/keep idea but a boundary on the capstone's sleep finding, so it became a pre-registered
  confirmation (C3) at the ~2% point, with the budget matched conservatively in sleep's favour.

### Idea 6: DFA + homeostasis target-rate sweep. KILLED (kill criterion met). [pilot, DoorKey-6x6, 400k frames]
Cut to 2 seeds per target (101-102) instead of 3, to save budget for Phase C.
| homeostatic target | solved (final eval > 0.5) | final eval return |
|---|---|---|
| 0.05 | 2/2 | 0.933, 0.876 |
| 0.15 (pre-registered; capstone 7/10 on seeds 1-10) | 2/2 | 0.956, 0.934 |
| 0.3 | 2/2 | 0.687 (success 0.88), 0.882 |
- No target solved more seeds than 0.15, so it is killed. At n = 2 per arm the pilot cannot detect a solve-rate change
  of the size that matters (7/10 → 10/10); it only rules out a dramatic failure of the other targets. 0.3 looks
  slightly worse. [pilot; not a finding]
- Cost 1.2 core-h.

### Idea 7: stateful SNN as working memory. NOT RUN.
Its only test bed was CueFirst-S11, which idea 4 killed: no memory agent learned the task, so a stateful-SNN arm could
not be distinguished from anything. Dropped without spending compute.

### Idea 8 (added after Phase C): isolation at LwF's memory. [pilot]
- **Motivation.** [ours, C1] LwF-int8 reached 0.944 at 0.91 MB of persistent memory. Isolation reached 0.975, but at
  3.20 MB. RQ5 asks about performance per unit of memory, so the fair reference is isolation shrunk to LwF's
  parameter count. [lit] PackNet / HAT-style work treats isolation capacity as the binding constraint.
- **Arm.** Five per-task networks with encoder channels (8, 16, 32), feat_dim 64, hidden 64: 44,928 params each, 224,640
  total (LwF's shared network: 228,632). Same frames, lr, evaluation; fp32 for both (int8 would scale both by 4).
- **Confound.** Narrower networks change optimisation as well as capacity. So this asks what isolation achieves at this
  budget with the default recipe, not what the best small architecture achieves.
- **Prediction.** Uncertain; the fetch tasks are easy, so a quarter-width network per task may still reach ~0.95.
- **Pilot.** CNN fetch5, seeds 101-102 (about 0.3 core-h).
- **Kill / promote.** Pilot mean ≥ 0.944 means isolation matches LwF per unit of memory, so LwF has no memory advantage.
  Report it and stop. Pilot mean < 0.914 (LwF − 0.03) means LwF may win per unit of memory; promote to a pre-registered
  C4 against C1's LwF-int8 runs.

- **Pilot result (seeds 101-102): ACC 0.972 and 0.972, FORGET 0.002**; every task 0.97-0.98 right after training;
  407 s per run. **Kill criterion met:** at LwF's parameter count, isolation matches or beats LwF (0.944 confirmed).
  So on CNN fetch5, LwF has no memory advantage over isolation. This also qualifies the capstone's memory frontier,
  which compared shared methods only against full-width isolation. [pilot]
- Because that qualification matters for RQ5, it gets a confirmation (C4, below) and a cheap width sweep (idea 8b).

### Idea 8b: how far isolation shrinks (width sweep), and whether it holds on fetch3 and on the SNN. [pilot]
Logged before running; seeds 101-102; same frames, lr and evaluation as everything else.
| arm | per-task network | total params | fp32 MB | compare with |
|---|---|---|---|---|
| CNN fetch5, ½ of LwF | channels (6,12,24), feat 48, hidden 32 | 119,210 | 0.48 | LwF-int8 0.944 (0.91 MB) |
| CNN fetch5, ¼ of LwF | channels (4,8,16), feat 32, hidden 32 | 57,940 | 0.23 | LwF-int8 0.944 (0.91 MB) |
| CNN fetch3 | channels (8,16,32), feat 96, hidden 48 | 186,624 | 0.75 | frozen sleep@200 0.964, replay@200 0.946 (0.89 MB) |
| SNN fetch3 | channels (8,16,32), feat 96, hidden 48 | 187,080 | 0.75 | C2 LwF-int8 0.869 (0.78 MB), replay@1380 0.885 (1.56 MB) |
- **Prediction.** CNN isolation holds down to ½ and breaks somewhere by ¼. The SNN is harder (capstone: SNN fetch5
  isolation needed 450k frames/task), so narrow SNN isolation may fall below C2's shared methods.
- **What would change conclusions.** If the SNN narrow isolation is at least C2 replay's 0.885, isolation dominates per
  unit of memory on both substrates at these task difficulties; that would get its own pre-registration (C5).

### Infrastructure incidents
- Container restart 1: 4 pilot jobs killed, rerun from scratch; 0.183 core-h lost.
- Container restart 2 (22:36): the first 4 C2 SNN runs killed about 2 min in, rerun from scratch with the same seeds
  (pre-registered stopping rule); 0.13 core-h lost.
- Commit `16883dd` accidentally added run checkpoints (`*.pt`); they were untracked in the next commit but remain in
  the `explore` branch history.

## Phase C: pre-registration
Written and committed before any confirmation run started (commit that adds this section). Nothing below changes
after the first confirmation run starts, whatever the interim results.

**Candidate.** Idea 2, buffer-free distillation (LwF) with an int8 teacher (`--method lwf --mk teacher_int8=true`,
batch 128, coef 1.0).
- Chosen from pilots because it was the only idea that cleared its kill criterion (CNN fetch3 0.977/0.970, int8
  0.976; CNN fetch5 0.969/0.812, int8 0.963; SNN fetch3 0.825).
- The int8 teacher was picked over the fp32 teacher before confirmation because it needs a quarter of the extra peak
  memory, and the paired pilots and the int8 audit showed no cost.

**Control: replay whose extra storage is one full fp32 network.**
- Same loss (policy KL + 0.5 value MSE), same coefficient, same 128 distilled/replayed samples per PPO minibatch (so
  replayed samples are matched), and the same frames, learning rate and seeds.
- Buffer = ⌊fp32 network bytes / (T × 188 B)⌋ states/task: 973 on CNN fetch5, 1,380 on SNN fetch3.
- This gives replay 4× LwF-int8's extra peak memory (LwF's int8 snapshot is 231 KB on CNN fetch5; the replay buffer
  is 915 KB). After training LwF stores nothing, while replay keeps its buffer. The control is deliberately generous
  to replay.

### C1: CNN fetch5 (primary family)
- **Arms and seeds.** LwF-int8, replay@973, and isolation as a reference. Seeds 201-210 (10 per arm, never used), 30
  runs. `configs/cl_cnn.yaml`: 150k frames/task, lr 1e-3, 100 evaluation episodes/task.
- **H1 (primary).** Final ACC differs between LwF-int8 and replay@973. Two-sided Welch t-test, α = 0.05. Report the
  difference with its 95% Welch CI. Pilot direction: LwF higher. The test is two-sided anyway.
- **H2.** FORGET differs between LwF-int8 and replay@973 (Welch).
- **H3.** LwF-int8 vs isolation, final ACC (Welch). "No large difference" may be claimed only if the 95% CI lies
  within ±0.05. Otherwise the difference is reported as it is.
- **Multiple comparisons.** Holm correction over H1-H3.
- **Descriptive.** Persistent and peak memory; replayed-sample counts (expected to match within about 1%); CPU time
  per run; per-task R rows; number of runs with a never-learned task (a task under 0.5 right after its own training).

### C2: SNN fetch3 (separate family; the capstone's substrate)
- **Arms and seeds.** LwF-int8 vs replay@1380. Seeds 201-208 (8 per arm), 16 runs. `configs/cl_snn.yaml` (150k
  frames/task, lr 1e-3).
- **H4.** Final ACC differs (two-sided Welch, α = 0.05; 95% CI).
- **H5.** FORGET differs (Welch). Holm correction over H4-H5.

### C3: sleep vs replay when replay is scarce (CNN fetch3; separate family; added before any C3 run)
- **Motivation (pilots, seed 101).** At about 10% of the usual budget, replay reached 0.935 and sleep 0.841. At about
  2%, replay reached 0.957 (FORGET 0.001) and sleep 0.585 (FORGET 0.59).
- **Arms and seeds.** Sleep with `replay_budget=14080` (110 sleep steps of 128) vs replay with `batch=3` (3 replayed
  states per PPO minibatch, about 13.7-14.0k in total). Seeds 201-208 (8 per arm), 16 runs.
  `configs/cl_cnn.yaml`, default buffer 5,000/task, same frames, lr and seeds.
- **Budget matching is conservative towards sleep.** 14,080 is at least the largest replay count expected from the
  capstone's replay runs (4,566-4,653 minibatches × 3). The exact counts are reported.
- **H6.** Final ACC differs (two-sided Welch, α = 0.05; 95% CI).
- **H7.** FORGET differs (Welch). Holm correction over H6-H7.

### C4: isolation at LwF's memory (CNN fetch5; added after Phase C1-C3 finished and after the idea-8 pilot, before any C4 run)
- **Motivation.** Idea-8 pilot: quarter-width isolation (224,640 params) reached 0.972 on 2 pilot seeds.
- **Arm.** `isolation` with `--set "enc_kwargs={channels: [8, 16, 32]}" feat_dim=64 hidden=64`, tag `_narrow`. Seeds
  201-210 (10 runs), `configs/cl_cnn.yaml`, outputs in `explore/runs/confirm/`.
- **Comparison arms.** The C1 runs already on disk (same seeds, code, config and machine): LwF-int8, replay@973, and
  full-width isolation. They are not rerun.
- **H8.** ACC, narrow isolation − LwF-int8 (Welch, two-sided, 95% CI).
- **H9.** ACC, narrow isolation − replay@973 (Welch).
- **H10.** ACC, narrow isolation − full isolation (Welch). "No large difference" only if the CI lies within ±0.05.
- Holm over H8-H10. FORGET is reported descriptively (isolation's is 0 by construction, apart from evaluation noise).
- Same stopping rule as above. Analysis: `explore/confirm_stats.py` (C4 family added in the same commit as this
  section).

### Stopping rule
- Run every listed run once; no early stopping, no interim tests.
- A run killed by an infrastructure failure (for example a container restart) is rerun from scratch with the same
  seed. That is the queue's normal resume behaviour, and it is logged. A run that crashes twice is reported as
  missing.
- Analysis: `explore/confirm_stats.py`, committed with this section.
- Budget cap for Phase C: 12 core-h. If exceeded, C2 is cut to the seeds that finished and the cut is reported.

### What would falsify the headline
- If H1's CI includes 0 and its upper bound is below +0.03, the claim becomes "LwF matches replay given 4× its extra
  memory". That is still a memory result, but not an accuracy one.
- If LwF-int8 is significantly below replay, buffer-free distillation is worse, and that gets reported.

## Phase C results (run exactly as pre-registered; 62/62 runs, none missing)
Full tables: `explore/confirm_results.md` (pre-registered tests, `explore/confirm_stats.py`). Exploratory follow-ups
(written after seeing the results): `explore/confirm_secondary.md`. One change to `confirm_stats.py` after the runs:
the descriptive "persistent MB" column counted the default 5,000-state buffer as 0 for C3. It now uses the same default
as `scripts/memory_fair.py`. No test changed.

### C1: CNN fetch5, 10 seeds per arm [confirmed]
| arm | final ACC | FORGET | persistent memory | replayed / distilled samples |
|---|---|---|---|---|
| LwF-int8 | 0.944 ± 0.018 | 0.009 | 0.91 MB (network only) | 1.195M |
| replay@973 | 0.920 ± 0.052 | 0.019 | 1.83 MB | 1.195M |
| isolation | 0.975 ± 0.001 | 0.000 | 3.20 MB | 0 |
- **H1 (ACC, LwF − replay): +0.024 [−0.014, +0.062], Welch p = 0.19, Holm p = 0.20.** Not significant. The CI
  includes 0 and its upper bound is above +0.03, so neither falsification branch applies: the direction is unresolved.
  What the CI does exclude is LwF being more than 0.014 worse than replay, at half the persistent memory and a quarter
  of the extra peak memory.
- **H2 (FORGET): −0.010 [−0.022, +0.002], p = 0.10, Holm p = 0.20.** Not significant.
- **H3 (ACC, LwF − isolation): −0.031 [−0.044, −0.019], p = 0.0003, Holm p = 0.001.** LwF is significantly below
  isolation. The CI lies inside ±0.05, so by the pre-registered rule this is also "no large difference": isolation buys
  about 3 points of ACC for 3.5× the memory.
- No run had a never-learned task. The pilot's fetch5 s102 failure (task 2 at 0.33) did not recur in 10 fresh seeds.
- [exploratory] LwF's spread is smaller (SD 0.018 vs 0.052; worst seed 0.911 vs 0.787; Brown-Forsythe p = 0.13), and
  LwF is above replay on 8/10 seeds (paired difference +0.024 [−0.005, +0.053], p = 0.09).

### C2: SNN fetch3, 8 seeds per arm [confirmed]
| arm | final ACC | FORGET | persistent memory | samples |
|---|---|---|---|---|
| LwF-int8 | 0.869 ± 0.068 | 0.031 | 0.78 MB | 0.532M |
| replay@1380 | 0.885 ± 0.073 | 0.014 | 1.56 MB | 0.527M |
- **H4 (ACC): −0.016 [−0.092, +0.059], p = 0.65.** Not significant, and the CI is too wide to exclude a large effect
  either way.
- **H5 (FORGET): +0.017 [+0.006, +0.027], p = 0.005, Holm p = 0.011.** Significant: on the SNN, LwF forgets more than
  replay. The effect is small in absolute terms (0.031 vs 0.014).
- [exploratory] ACC is almost entirely decided by the seed (cross-arm correlation r = 0.98; seed 204 fails task 0 in
  both arms). Paired by seed, LwF is 0.016 [0.004, 0.029] below replay (paired p = 0.02). So on the SNN, LwF is slightly
  worse than replay with twice the persistent memory, consistently but by little.

### C3: sleep vs replay at ~14k replayed samples (~2% of the usual budget), CNN fetch3, 8 seeds per arm [confirmed]
| arm | final ACC | FORGET | samples |
|---|---|---|---|
| sleep, replay_budget 14,080 | 0.815 ± 0.145 | 0.244 | 14,080 |
| replay, 3 per minibatch | 0.945 ± 0.020 | 0.003 | 13,986 (13,866-14,016) |
- **H6 (ACC, sleep − replay): −0.131 [−0.252, −0.010], p = 0.038.** **H7 (FORGET): +0.241 [+0.058, +0.424], p = 0.017,
  Holm p = 0.034.** Both significant. When replay is scarce, spending it inside the PPO minibatches protects old tasks
  far better than spending it in offline sleep phases. The capstone's "sleep ≥ replay" holds only at the full budget.
- [exploratory] Sleep's failure is bimodal and hits task 0: its final task-0 return is 0.03-0.93 (4/8 runs below 0.5),
  against 0.98 on every replay run (Fisher p = 0.08).
- [exploratory] Interleaved replay has a cost the ACC average hides. Its last task (just trained, no forgetting yet)
  reaches 0.889 ± 0.055 against sleep's 0.973 ± 0.008 (difference 0.084 [0.038, 0.130], p = 0.003). Mixing old states
  into every minibatch slows learning of the current task, while offline sleep leaves plasticity intact. Here the
  protection is worth far more than it costs.

## Phase D: report

## Budget ledger
