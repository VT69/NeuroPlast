# NeuroPlast — Literature & Experimental Context

> **How to use this file.** Read it before designing a new experiment or writing
> any results text. `PROGRESS.md` is the source of truth for results and
> overrides this file wherever they disagree. Section 38 ("Corrections and
> additions") overrides every earlier section of this file.

## 0. What this document is

This document gives the research context for **NeuroPlast**. The project does
not assume that biologically inspired mechanisms are superior. It is an
empirical investigation into whether SNNs, local plasticity, homeostasis and
sleep-like consolidation provide measurable benefits for **continual
reinforcement learning**, under constraints on parameters, memory and compute.

The literature below is background and motivation. **Our experimental results
override the original hypotheses when they contradict them.**

Keep these categories distinct in all reasoning and writing:

- **Literature claim**: what an external paper reports.
- **NeuroPlast result**: what our experiments actually observed.
- **Hypothesis**: something we want to test.
- **Interpretation**: an explanation supported to some degree by evidence.

Never turn a hypothesis or interpretation into an established fact.

---

## 1. Core research problem

The central problem is **continual reinforcement learning**: an agent learns
Task A → Task B → Task C → ... while its parameters keep being updated.

The challenge is the **stability–plasticity dilemma**:

- **Plasticity**: the agent must change enough to learn new tasks.
- **Stability**: the agent must preserve previously learned abilities.

Naive sequential fine-tuning gives high plasticity and poor stability, because
gradient updates for a new task modify parameters that earlier tasks relied on.

- **EWC** (Kirkpatrick et al.) formalises this: parameters important to earlier
  tasks should change less aggressively.
- **CLEAR** (Rolnick et al.) showed that experience replay is an extremely
  strong solution in continual RL: mix new on-policy experience with replayed
  old experience, plus behavioural cloning on replayed states.
- **Continual World** (Wołczyk et al.) argued that continual RL must be judged
  not only by forgetting but also by **forward transfer**: did earlier learning
  help later tasks? Methods can reduce forgetting while hurting plasticity.

---

## 2. NeuroPlast's original hypotheses

The original project combined four ideas.

**A. SNN perception.** Replace a dense CNN with an SNN, motivated by
event-driven computation, temporal dynamics, sparse binary spikes and potential
neuromorphic efficiency. Energy efficiency was always framed as a hypothesis to
measure with a proxy (synaptic operations vs dense MACs), not a claim about
physical power.

**B. Transformer working memory.** Attend over recent SNN representations to
provide working memory where the current observation alone is insufficient.
**Not yet validated**: current tasks are Markov given the current frame.

**C. Local plasticity.** Combine a global gradient with local STDP/Hebbian
learning:

```
ΔW = α · ΔW_local + β · ΔW_gradient
```

with α:β swept experimentally rather than fixed by intuition.

**D. Sleep consolidation.** Periodically stop interacting with the environment
and consolidate offline. The plan proposed a simple baseline first (offline
replay + distillation), then a more biologically inspired interleaving of
novel and familiar experience. That distinction became important
experimentally.

---

## 3. Khetarpal et al. — Towards Continual Reinforcement Learning

The broad conceptual framework. Continual RL is an agent learning from an
evolving stream of tasks, with desiderata including online and incremental
learning, retention, adaptation, transfer, minimising forgetting, and
potentially task-agnostic operation (unknown task boundaries, no task IDs).

**Relevance.** Our benchmark is **task-incremental with task ID given**. That
is fine for a controlled first study but is a limitation. Do not claim
NeuroPlast solves general continual RL. Use: *"We study a controlled
task-incremental continual RL setting."*

---

## 4. EWC — Elastic Weight Consolidation

After learning task A, estimate parameter importance with the Fisher
information, then penalise moving important parameters while learning task B:

```
L(θ) = L_B(θ) + (λ/2) · Σ_i F_i · (θ_i − θ*_A,i)²
```

High Fisher → resist change; low Fisher → allow plasticity. The paper also
showed tasks can share representations rather than occupying separate
subnetworks.

**NeuroPlast lesson.** EWC shows the core trade-off: protecting old knowledge
too strongly reduces the ability to learn new knowledge. Our EWC results showed
exactly this: strong stability, poor acquisition of later tasks, worse as λ
grows. Describe EWC as *a stability-oriented baseline with a plasticity cost*,
not simply "weaker than sleep".

---

## 5. CLEAR — probably the most important baseline for sleep

CLEAR combines new experience (for plasticity), replayed old experience (for
stability), and behavioural cloning on replayed states (policy KL + value L2
toward the historical policy). Replay drastically reduces forgetting; even
replay without behavioural cloning helps.

**Why it matters.** Our sleep phase uses a replay buffer, offline updates and
self-distillation. **Sleep must be compared directly against replay at matched
budgets**, or we can't tell whether any benefit comes from consolidation or
just from replay.

---

## 6. PackNet / parameter isolation

PackNet trains a task, prunes, freezes the important parameters, and uses the
freed capacity for the next task, storing masks rather than whole networks.

Isolation is almost impossible to beat on raw forgetting: frozen weights mean
forgetting on that task is **zero by construction**. So the meaningful question
is not "does NeuroPlast forget less than isolation?" but *"can shared parameters
retain high performance at substantially lower total memory while exploiting
transfer?"* (See section 38.3 on counting replay-buffer memory.)

Note: our isolation baseline is one full network per task, not PackNet-style
masks. PackNet-style isolation would be cheaper and is a stronger baseline if
time allows.

---

## 7. Continual World — evaluation lesson

Continual RL has multiple objectives: retention, capacity, compute, plasticity,
forward transfer. Reducing forgetting alone can hide an inability to learn new
tasks. Benchmarks need meaningful relationships between tasks so transfer can
be measured.

**Implication.** If every task is learnable from scratch in a small number of
frames, FWT ≈ 0 tells us little. That is why our 3-task CNN benchmark shows no
reliable transfer. The SNN sometimes shows positive transfer, but part of that
may be because fresh isolated SNNs were under-trained (section 38.6).

---

## 8. Sleep literature — Tadros et al., Sleep-like unsupervised replay

Their Sleep Replay Consolidation (SRC) roughly: train awake with backprop, stop,
generate spontaneous/noisy activity, replay internal representations, apply
local unsupervised Hebbian plasticity, resume. They report that this can
recover lost task performance, interpreted as reactivating old representations
and reducing representational overlap.

**Difference from NeuroPlast.** Their experiments are mainly ANN classification
settings; ours is continual RL, and our sleep implementation is offline replay
plus distillation, not SRC.

- Do **not** say: "Our sleep implementation reproduces Tadros et al."
- Say: "Our sleep phase is conceptually inspired by sleep replay/consolidation
  literature; the implementation uses offline replay and distillation in a
  continual RL setting."

---

## 9. The key surprise from our sleep experiments

Original hypothesis: sleep + local plasticity produces the benefit.

Result: vanilla STDP inside sleep was catastrophic; stabilised STDP inside sleep
was harmless but gave no measurable improvement over plain sleep.

So current evidence says the benefit of NeuroPlast sleep comes from **offline
rehearsal/distillation**, not from the STDP update. NeuroPlast is therefore not
evidence that "biological STDP during sleep solves forgetting". It is evidence
that *an offline consolidation phase can strongly reduce forgetting, while the
local STDP mechanism tested here is not necessary and can destabilise
learning.* That is a more honest and more interesting result.

---

## 10. STDP and three-factor learning

(Lee et al., 2025 review, *Brain-inspired learning rules for SNN-based
control*.) Ordinary STDP depends only on local pre/post spike timing. RL needs
credit assignment: which synaptic changes contributed to reward? Reward-modulated
STDP adds a global third factor (reward, TD error, neuromodulator):

```
Δw ∝ eligibility trace × modulatory signal
```

This is far more RL-appropriate than unsupervised Hebbian STDP.

**Lesson.** "Add STDP to backprop and maybe performance improves" was too
simplistic. STDP alone does not solve RL credit assignment. And our experiments
currently provide **no evidence that STDP itself improves learning**.

---

## 11. Why our vanilla STDP failed

Observed: firing rate rose from ~0.15 to 0.5–0.77, learning degraded severely,
and STDP did worse than random perturbations of the same magnitude.

Mechanism: the MiniGrid frame is static across the SNN's T timesteps, so the
same pre→post pairings repeat. With a predominantly potentiating rule this
becomes a positive feedback loop:

```
w ↑ → firing rate ↑ → more correlated spikes → Δw ↑ → ...
```

This is **runaway excitation**, not proof that STDP is fundamentally useless.
STDP also did worse than same-size noise because its drift has a consistent
sign (grows ~n) while noise random-walks (~√n).

The stabilised version (covariance-centering + local firing-rate homeostasis)
removed the instability, but the remaining effect was not distinguishable from
controls. Later work showed the rescue of DFA came from the homeostasis
component alone (section 22).

---

## 12. Surrogate gradients (Van den Berghe et al.)

Spikes are non-differentiable (`S_t = H(V_t − V_th)`), so backprop uses a smooth
surrogate derivative. The paper studies the surrogate **slope**: shallower
slopes increase gradient magnitude in deeper layers but reduce alignment with
the true gradient, and the effect is much stronger in RL than supervised
learning. It reports sizeable gains from favourable or adaptive slope schedules
(the specific multipliers are unverified; check the PDF before quoting).

It also identifies an SNN-RL-specific vicious cycle: SNNs need temporal warm-up
for membrane states, but poor early policies end episodes early, giving too
little warm-up, giving bad gradients, giving poor policies.

**Implication.** If SNN training is unstable, consider surrogate slope, slope
scheduling, membrane warm-up, sequence length and homeostatic stabilisation
before concluding the SNN architecture is inadequate.

---

## 13. SNNs are not automatically efficient

SNNs are motivated by event-driven computation and accumulates instead of
multiply-accumulates. But **simulation on a CPU/GPU is not neuromorphic
execution**: T timesteps multiply compute and memory roughly T-fold versus a
static network.

Our proxy:

```
SynOps   = Σ_t (nonzero presynaptic spikes × fan-out)
E_SNN    ≈ N_AC  · E_AC     (E_AC  ≈ 0.9 pJ, 45 nm)
E_CNN    ≈ N_MAC · E_MAC    (E_MAC ≈ 4.6 pJ, 45 nm)
```

This is an **estimate**, not wall-plug energy.

- Too broad: "SNNs are more energy efficient."
- Correct: "Under an event-driven SynOp model, the tested SNN reaches competitive
  performance at lower operation counts in the low-compute regime; translating
  that into physical energy savings requires neuromorphic hardware."

---

## 14. Spikingformer

Shows Transformer-style modelling can be combined with spike-driven
computation. Its key concern: "spiking" architectures often hide non-spike
computation (e.g. residual connections introducing floating-point
multiplications). It proposes a residual design that keeps computation
spike-driven.

**Implication.** If we move toward a truly spiking Transformer, don't count only
spikes. Also account for dense projections, residual paths, attention,
floating-point multiplications, normalisation and memory movement.

---

## 15. Spiking Decision Transformer (SNN-DT)

Combines spiking neurons, Transformer sequence modelling, phase-based spike
positional encoding, local three-factor plasticity in the action head, and
dendritic-style routing. Evaluated on classic control (CartPole, MountainCar,
Acrobot, Pendulum) in **offline** RL, with ablations supporting phase coding and
routing, and very sparse activity (exact spike counts unverified).

**Not a direct baseline.** SNN-DT is offline RL with a Decision Transformer
formulation; NeuroPlast is online PPO, continual learning, MiniGrid, sleep
consolidation. Use it as architectural inspiration only. The paper itself notes
longer-horizon scaling and attention cost as limitations.

---

## 16. Self-Consolidation for Self-Evolving Agents (Yu et al.)

Instead of retrieving ever-growing historical experience, extract useful
information, reflect on failures, and distil experience into compact
parametric knowledge: *non-parametric experience → parametric memory*. It
highlights that accumulating raw experience scales poorly.

**Relevance.** Replay keeps old experience and shows it again; consolidation
converts experience into lasting parameters. NeuroPlast's sleep sits between
the two and still relies heavily on replay. Do not claim NeuroPlast solves the
experience-to-parametric-memory problem. This also motivates counting buffer
memory in RQ5 (section 38.3).

---

## 17. MiniGrid

Cheap, interpretable, goal-oriented, customisable RL environments. Advantages
for NeuroPlast: cheap simulation, small observations, interpretable episodes,
easy task sequencing, visual demos, controllable difficulty.

But MiniGrid is a **controlled testbed**, not a substitute for large-scale
continual RL benchmarks. Claims must stay scoped to it.

---

# NeuroPlast results (snapshot as of 2026-09-28)

> Sections 18–28 summarise results at the time of writing. `PROGRESS.md` is
> authoritative and will be more current.

## 18. RQ1 — Does STDP improve learning?

**Hypothesis:** `ΔW = α·ΔW_STDP + β·ΔW_BP` improves sample efficiency or
representation learning.

**What happened (DoorKey-5x5):** backprop learned quickly; a *frozen* random SNN
encoder learned about as fast (so this task can only detect harm, not help);
vanilla STDP degraded performance and at high strength prevented solving
(p ≈ 0.02 vs backprop); same-size random perturbations outperformed STDP.
Mechanism: runaway excitation (section 11).

**Stabilised STDP (DoorKey-6x6, where encoder learning matters):** removed the
harm; best mean AUC (0.49 vs 0.41, 5 seeds) but p = 0.67, and the random
control matched backprop. Outcomes are bimodal (a seed solves or never does),
so solve rate over ~10 seeds per arm is the right statistic.

**Interpretation:** STDP is not currently supported as a beneficial ingredient.
- Don't write: "STDP failed."
- Write: "The tested STDP formulation did not produce a statistically reliable
  benefit and was unstable without homeostatic stabilisation."

## 19. RQ2 — Sleep consolidation

Early CNN results (fetch3, 3 tasks): naive forgets severely; EWC retains but
can't learn new tasks; replay and sleep both retain strongly and tie (CNN
0.978 vs 0.970, p = 0.28). Small buffers (50 and 200 states/task): no
significant difference. Qualitatively, sleep *recovers* previously dipped
old-task performance in later phases; replay only declines.

**Full hybrid (SNN + Transformer + sleep).** The original 150k frames/task budget
was insufficient for the hybrid to learn at all, so the protocol was changed to
**LR 3e-4 and 400k frames/task** (this must be disclosed, and all arms in the
comparison must use the same protocol). Three seeds:

| Method    | ACC   | Forgetting |
|-----------|-------|------------|
| Sleep     | 0.873 | 0.004      |
| Replay    | 0.834 | 0.02       |
| Isolation | 0.832 | ~0         |
| Naive     | 0.452 | 0.54       |

- Sleep vs naive: p = 2e-5.
- Sleep vs replay: p ≈ 0.07. Every sleep seed beat every replay seed, but this
  is a trend, not a finding.

Say: "Sleep showed a promising advantage over replay, not statistically
conclusive at three seeds." Never: "Sleep significantly outperforms replay."

## 20. Sleep + STDP

Tested plain sleep, sleep + homeostasis, sleep + STDP. Vanilla STDP in sleep
wiped out every task (ACC 0.16–0.20). Stabilised STDP was harmless but no better
than plain sleep. Sleep + homeostasis did not beat plain sleep in the continual
setting.

So: **sleep benefit ≈ offline rehearsal/distillation**, not STDP during sleep.
This separates systems-level consolidation from synaptic-level plasticity.

## 21. RQ3 — SNN vs CNN on operations

- At moderate op counts: CNN and SNN comparable; CNN sometimes slightly better.
- Low end: the L1-sparsified CNN floors at ~33k ops/frame and then collapses;
  the SNN stays functional down to ~16k (~2× lower).
- T=2 dominates T=4 dominates T=8.
- On CPU the SNN costs ~4× more to train.
- Only L1 activation sparsification was tried for the CNN; k-WTA, learned
  thresholds and structured sparsity were not.

Correct claim: "The tested SNN achieved a lower operation-count floor than the
tested L1-sparsified CNN on this task." Not: "SNNs are more efficient than CNNs."

## 22. RQ4 — Architecture variants

- **Plain CNN:** best training efficiency by a wide margin (~4×).
- **SNN + surrogate-gradient backprop (variant A):** more expensive to train;
  sparse event-driven inference. Best spiking variant, especially with sleep.
- **SNN + STDP:** more expensive and not clearly beneficial.
- **Local-only encoder (variant B):** never solved a task (0/6 seeds).
- **DFA / e-prop-style SNN (variant C, no backprop between layers):** alone it
  fails on DoorKey-6x6 (1/5 seeds solve) due to runaway firing. Adding
  homeostasis alone gives 3/3 solves (p = 0.0003 vs DFA alone); STDP on top adds
  nothing and does no better than random updates. With homeostasis, DFA matches
  backprop on that task. In the continual setting it learned too slowly at the
  tested budget (ACC 0.45 vs 0.89 for backprop).
- **SNN + Transformer:** solves the benchmark given enough training, but the
  tasks don't need memory, so the Transformer adds cost with no demonstrated
  benefit. This says the benchmark doesn't exercise the Transformer, not that
  Transformers are useless.

**Homeostasis evidence is DFA-specific so far.** Backprop + homeostasis alone
has not been run (section 38.4).

## 23. RQ5 — Shared weights vs isolation

**Per parameter:** shared weights with sleep or replay give roughly 2.5× more
ACC per 100k parameters (0.50 vs 0.20 CNN; 0.46 vs 0.17 SNN; p < 0.001). This
**ignores replay-buffer memory** (section 38.3).

**Raw accuracy, SNN trunk, 3 tasks:** sleep 0.889 vs isolation 0.833 is **not
significant** (p = 0.32). An earlier claim that it was a win was corrected.

**Five tasks:**

- CNN (3 seeds): sleep 0.928, replay 0.926, naive 0.416, isolation 0.976.
  Isolation has higher raw accuracy because the shared network learns later
  tasks less well; it isn't forgetting them.
- SNN (2 seeds): sleep ~0.778 vs isolation ~0.475. Fresh isolated SNNs often
  failed to learn within 150k frames per task, so part of this gap may be
  under-training rather than transfer (section 38.6). Preliminary only.

## 24. The key conceptual decomposition

"Biological inspiration" is not one package. There are separable mechanisms:

| Mechanism | Question | Current answer |
|---|---|---|
| Spike-based computation | Useful compute/energy trade-off? | Possibly at low op counts; hardware-dependent |
| Local plasticity (STDP) | Improves RL learning? | Not demonstrated; can destabilise |
| Homeostasis | Stabilises spiking learning? | Yes for DFA; untested with backprop |
| Sleep / consolidation | Reduces forgetting? | Strongly vs naive; promising vs replay, not conclusive |
| Shared parameters | Transfer at lower memory than isolation? | Promising per parameter; buffer memory and budget confounds open |
| Transformer memory | Helps on memory tasks? | Not yet tested |

This decomposition is the most important conceptual evolution of the project.

## 25. What literature + experiments suggest together

The evidence does **not** support "SNN + STDP + Transformer + sleep = a
biologically inspired agent better than conventional RL". It suggests:

- Conventional RL solves individual tasks well; continual RL introduces the
  stability–plasticity trade-off.
- Replay is already extremely strong (CLEAR). Any new consolidation method must
  beat or complement replay, not merely beat naive fine-tuning.
- EWC protects knowledge through constraints at a plasticity cost.
- Isolation removes forgetting by adding capacity, which grows with tasks.
- SNNs' strongest potential advantage is sparse event-driven computation under
  appropriate hardware assumptions, not training speed.
- Local plasticity alone doesn't solve deep RL credit assignment; three-factor
  rules are more RL-compatible.
- Sleep-like consolidation introduces a useful temporal separation: acquire
  quickly while awake, consolidate offline during sleep.

## 26. Biggest scientific weakness: benchmark difficulty and mechanism alignment

- If tasks are solvable from the current frame, the Transformer has no job.
- If each task is learnable from scratch in ~50–60k frames, transfer is hard to
  detect.
- If task identity is given, the problem is easier than task-agnostic continual
  learning.

The next benchmarks should introduce a genuine memory requirement, longer task
sequences, meaningful shared structure between tasks, and eventually less
explicit task information (with the caveats in section 38.2).

## 27. Why a memory task matters

Current tasks are effectively `P(a_t | o_t)`. Memory tasks require
`P(a_t | o_{t−k:t})` or `P(a_t | o_{1:t})`, giving temporal state and the
Transformer an actual job (e.g. MiniGrid-MemoryS7 or similar).

Ablation design (all on the same task):

| Architecture | Role |
|---|---|
| CNN, single frame | memoryless baseline |
| CNN + frame stack | explicit short history |
| SNN, single frame | tests whether SNN state carries cross-frame information (see 38.5) |
| SNN + frame stack or small GRU | non-attention memory on the SNN encoder |
| SNN + Transformer | attention-based context |

Questions this answers: does any memory help; does the SNN's own state provide
it; does attention add value beyond simpler memory?

## 28. Literature-to-NeuroPlast mapping

| Literature | Core mechanism | What NeuroPlast learned |
|---|---|---|
| Khetarpal et al. | Continual RL framework | Need retention + transfer + adaptation, not forgetting alone |
| Kirkpatrick / EWC | Protect important parameters | Stability costs plasticity |
| CLEAR | Replay + behavioural cloning | Replay is a very strong baseline |
| PackNet | Parameter isolation | Zero forgetting is easy if capacity grows |
| Continual World | Forward transfer | Tasks must be related and hard enough |
| Tadros / SRC | Sleep + spontaneous replay + Hebbian plasticity | Consolidation plausible; our implementation differs |
| SNN learning-rule review | STDP / R-STDP / three-factor | RL needs reward/error modulation |
| Adaptive surrogate gradients | Surrogate slope, warm-up | SNN-RL optimisation has unique dynamics |
| Spikingformer | Spike-driven Transformer | Energy accounting is subtle |
| SNN-DT | Spiking sequence control + local plasticity | Inspiration, not a continual-RL baseline |
| Self-Consolidation | Experience → parametric memory | Replay dependence and memory growth matter |
| SpikingJelly | SNN infrastructure | Event-driven op counting matters |
| MiniGrid | Cheap goal-oriented RL | Good testbed; needs harder memory/continual tasks |

---

# Guidance for future work

## 29. What NOT to do

- **Don't assume STDP is beneficial.** No statistically reliable benefit so far.
- **Don't remove the STDP failure from the story.** It is a useful result.
- **Don't tune a failed mechanism until it looks good.** Add controls instead.
- **Don't say sleep beats replay conclusively** (p ≈ 0.07 at n = 3).
- **Don't claim biological realism.** This is biologically inspired engineering.
- **Don't claim physical energy efficiency.** We measure an operation/energy proxy.
- **Don't claim the Transformer works.** It hasn't been tested on a memory task.
- **Don't claim isolation is worse.** It has zero forgetting by construction; the
  question is performance per unit of total memory, and transfer.
- **Don't treat 2- or 3-seed results as established.**
- **Don't hide protocol changes** (e.g. the hybrid's LR 3e-4 / 400k frames).
- **Don't compare memory costs using parameters alone** (section 38.3).

## 30. Experimental priorities

1. **Finish RQ1** on a task where encoder learning matters, ~8–10 seeds per arm:
   backprop; backprop + homeostasis; backprop + stabilised STDP; backprop +
   random perturbation of matched size. Report solve rate. Key question: does
   STDP add anything beyond homeostasis and generic perturbation?
2. **Memory-fair RQ5:** sleep and replay at small buffers vs isolation,
   reporting total memory (parameters + buffer at actual storage size).
3. **Sleep vs replay, ~8–10 seeds**, with replay budget, frames, optimisation
   budget, architecture and evaluation matched. Include naive, EWC, isolation.
   Metrics: ACC, forgetting, BWT, FWT, sample efficiency, replay memory, compute.
4. **Isolation with an adequate budget**, so fresh networks reliably learn, to
   check whether the SNN transfer gap survives.
5. **Memory benchmark** with the ablation in section 27.
6. **Longer task sequences** (5 → possibly 10) with meaningful shared structure.
7. **Stretch: task-agnostic benchmark** with observable goals (section 38.2).

## 31. Recommended narrative

Not: "We built a biologically inspired RL agent and it works."

Instead: **"We decomposed several biologically inspired mechanisms for continual
reinforcement learning and determined experimentally which ones contribute."**

1. Naive local STDP is unstable and does not reliably improve RL learning.
2. Homeostatic regulation rescues spiking learning (so far shown for DFA).
3. Sleep-style offline consolidation strongly reduces catastrophic forgetting.
4. Its advantage over replay is promising but not yet conclusive.
5. Shared weights use far fewer parameters than isolation, and shared SNNs may
   transfer under constrained budgets (confounds pending).
6. SNN operation-count advantages appear in the low-operation regime; physical
   energy claims depend on hardware.
7. The Transformer's value is untested because current tasks don't need memory.

## 32. Architecture as separate mechanisms

```text
Environment → SNN perception → Transformer memory → Actor/Critic → Action

Wake:   RL gradient ───────────► all parameters
        local plasticity ──────► SNN parameters (optional arm)

Sleep:  replay old experience → offline rehearsal → distillation → parameters

Evaluate: retention, forgetting, forward/backward transfer, sample efficiency,
          parameter + buffer memory, operation count, compute time
```

SNN, Transformer, STDP, homeostasis and sleep operate at different levels and
must be evaluated separately.

## 33. The deepest conceptual framing: where knowledge is stored

| Strategy | Example | Pro | Con |
|---|---|---|---|
| Separate parameters | PackNet / isolation | Excellent retention | Capacity grows |
| Constrained shared parameters | EWC | Fixed capacity | Plasticity suffers |
| Replay memory | CLEAR | Powerful and simple | Stored data grows |
| Consolidation | Sleep / SRC | Aims to turn experience into stable representations | Hard to separate from replay |
| Local synaptic plasticity | STDP / three-factor | Local, online | Credit assignment, stability, scaling |

NeuroPlast asks: *can a shared neural substrate combine efficient representation,
replay/consolidation and local plasticity to reach a useful point in the
stability–plasticity–capacity trade-off?*

## 34. Evidence hierarchy (as of 2026-09-28)

| Confidence | Claim |
|---|---|
| Strong | Naive sequential learning causes catastrophic forgetting |
| Strong | Replay substantially reduces forgetting |
| Strong | EWC trades plasticity for stability |
| Strong | Vanilla STDP destabilises our SNN (runaway excitation) |
| Strong | Sleep-style consolidation prevents forgetting vs naive (p = 2e-5) |
| Moderate | Homeostasis rescues DFA-trained SNNs (untested with backprop) |
| Moderate | Shared weights are more parameter-efficient than isolation (buffer memory not yet counted) |
| Preliminary | Sleep outperforms replay (p ≈ 0.07, n = 3) |
| Preliminary | Shared SNNs transfer positively (n = 2; isolation possibly under-trained) |
| Conditional | SNN compute/energy advantage (sparsity and hardware assumptions) |
| Unknown | Transformer improves NeuroPlast |
| Unsupported | STDP improves continual RL |

## 35. Rule for any new mechanism

Before adding anything, write down in `PROGRESS.md`:

1. which literature motivates it,
2. which NeuroPlast result or failure it addresses,
3. what confound it introduces,
4. what ablation would establish that it helped.

Hypothesis-to-test examples:

| Mechanism | Hypothesis | Test |
|---|---|---|
| STDP | Improves representation learning | BP vs BP+STDP vs BP+homeostasis vs BP+random |
| Sleep | Benefits beyond ordinary replay | Replay vs sleep at equal replay and compute budgets |
| Transformer | Context helps partially observable tasks | SNN vs SNN+GRU/frame-stack vs SNN+Transformer on a memory task |
| SNN | Better low-operation frontier | CNN vs SNN at matched parameters/ops, with stronger CNN sparsifiers |
| Shared weights | Transfer without linear growth | Shared vs isolation as task count grows, counting total memory |

Prefer the smallest experiment that answers the question. Preserve failed
experiments and negative results in the log.

## 36. Bottom line

Treat NeuroPlast as **a controlled empirical study of mechanisms for continual
RL, using a hybrid SNN agent as the experimental substrate**. Not "an attempt to
build a brain", not "an SNN that should automatically be more efficient", and not
"a system where every component is expected to help". The value comes from
disentangling the mechanisms.

## 37. Literature files (place PDFs in `docs/papers/`)

1. Khetarpal et al. — Towards Continual Reinforcement Learning: A Review and Perspectives
2. Kirkpatrick et al. — Overcoming Catastrophic Forgetting in Neural Networks (EWC)
3. Rolnick et al. — Experience Replay for Continual Learning (CLEAR)
4. Wołczyk et al. — Continual World
5. Mallya & Lazebnik — PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning
6. Tadros et al. — Sleep-like Unsupervised Replay Reduces Catastrophic Forgetting
7. Lee et al. — Brain-inspired Learning Rules for SNN-based Control
8. Van den Berghe et al. — Adaptive Surrogate Gradients for Sequential RL in SNNs
9. Spikingformer — A Key Foundation Model for SNNs
10. Spiking Decision Transformers — Local Plasticity, Phase-Coding and Dendritic Routing
11. Yu et al. — Self-Consolidation for Self-Evolving Agents
12. SpikingJelly — Open-source infrastructure for spike-based intelligence
13. Huo et al. — Research on SNN Learning Algorithms and Networks Based on Biological Plausibility
14. MiniGrid / MiniWorld supplementary material

---

# 38. Corrections and additions (override all earlier sections)

1. **Source of truth for results.** Sections 18–28 and 34 are a snapshot as of
   2026-09-28. `PROGRESS.md` overrides them where they disagree.

2. **Removing task ID from fetch3 makes it unsolvable, not harder.** The goal
   ("pick up object k") is not visible in the observation. A task-agnostic
   benchmark needs a new task sequence where the goal is inferable from the
   observation (e.g. different environment families in sequence, or a visual
   goal cue), plus replacing the per-task heads that sleep's distillation
   currently relies on. Treat this as a benchmark redesign and a stretch goal.

3. **Count replay-buffer memory in every RQ5 claim.** Isolation grows
   parameters; sleep and replay grow stored states. Rough estimate: one
   observation is 20×7×7 binary values (~123 bytes bit-packed, ~3.9 KB as
   float32); 5,000 states/task over 3 tasks is ~1.8 MB even bit-packed, versus
   ~0.6–0.8 MB for one extra ~150–190k-parameter network in float32. Measure
   the actual storage format rather than trusting this estimate. Report total
   memory (parameters + buffer) alongside per-parameter numbers. The
   small-buffer arms (50 and 200 states/task) are the fair comparison against
   isolation; make them the headline if the result survives.

4. **Homeostasis evidence is DFA-specific.** It rescued DFA-trained SNNs.
   Backprop + homeostasis alone has not been run. Until it has, claim only
   "homeostasis rescues DFA-trained SNNs."

5. **Memory benchmark design.**
   - Before interpreting the plain SNN as having "implicit temporal dynamics",
     check whether membrane state persists across environment steps or resets
     each frame. If it resets, the plain SNN has no cross-frame memory.
   - Include a non-attention memory baseline (frame stacking or a small GRU)
     on the same SNN encoder. Otherwise a Transformer win shows only that
     memory helps, not that attention helps.

6. **Isolation budget.** In the 5-task SNN run, isolated networks often failed
   to learn in 150k frames. Rerun isolation with a per-task budget at which a
   fresh network reliably learns, and check whether the transfer gap survives.

7. **Paper-derived numbers.** Figures attributed to papers in this document
   were produced by automated summarisation and are unverified. Check them
   against the PDFs in `docs/papers/` before quoting them in any write-up.

8. **Correction note, 2026-09-28 (session 3).** Overrides 38.3, 38.4, and the
   RQ5 statements in sections 22, 23 and 34.
   - **38.4 was wrong.** Backprop + homeostasis *had* been run, in session 2:
     `runs/rq1_v2/rq1v2_homeo_s1-3`, 3/3 solved on DoorKey-6x6, AUC 0.67 vs 0.41
     for backprop (p = 0.09). Session 3 extended it to 10 seeds alongside
     backprop, backprop + stabilised STDP, and a matched random-perturbation
     control. The pre-registered results are in `PROGRESS.md` and
     `results/rq1_stats.md`; those, not the "DFA-specific" wording in 22, 24, 31,
     34 and 38.4, are authoritative.
   - **38.3's buffer estimate is replaced by a measurement.** The replay buffer
     stores the raw uint8 symbolic view (7×7×3 = 147 B), not the one-hot tensor:
     **188 B per stored state** on the CNN/SNN trunks (observation + mask + task
     id + teacher logits + value) and **632 B** for the hybrid's 4-frame
     windows. The implementation keeps a second concatenated copy (2× RAM), which
     is an implementation artefact reported separately.
   - **The RQ5 claims in sections 23 and 34 are superseded by the memory-fair
     analysis** (`results/rq5_memory.md`, `scripts/memory_fair.py`). Shared
     weights win *per parameter* (unchanged), but *per total memory* (parameters
     + replay buffer) at the default 5,000 states/task, isolation uses less
     memory and gets about twice the ACC per MB: CNN fetch3 1.92 MB / 0.979 vs
     sleep 3.60 MB / 0.978; hybrid 3.52 MB vs 10.79 MB; CNN fetch5 3.20 MB vs
     5.62 MB. Shared weights win per total memory **only at small buffers**
     (CNN fetch3, 200 states/task: 0.964 ACC at 0.89 MB, vs isolation 0.979 at
     1.92 MB). "Shared weights are more memory-efficient than isolation" may only
     be claimed with the buffer size stated.
