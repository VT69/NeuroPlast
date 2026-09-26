# NeuroPlast — 3-Person Team Division Plan

*A research/engineering work-split for the NeuroPlast capstone (hybrid SNN-Transformer RL agent with Hebbian/STDP + sleep-phase consolidation), optimized for parallel work, minimal Git conflicts, and clean integration.*

---

# 1. Project Understanding

**Objective.** Build and rigorously evaluate NeuroPlast, an RL agent that combines a spiking-neural-network (SNN) perceptual front end, a Transformer working-memory layer, a hybrid learning rule (surrogate-gradient backprop + local Hebbian/STDP, mixed by a swept weight α:β), and a periodic "sleep" consolidation phase that replays past experience offline to fight catastrophic forgetting. It is compared against four baselines — naive sequential fine-tuning, plain experience replay, EWC-style regularization, and a parameter-isolation upper bound — on a 3–5-task continual MiniGrid/BabyAI benchmark, answering five pre-registered research questions (RQ1–RQ5).

**Key architectural facts that drive the division:**

- Three candidate architectures (A/B/C, Section 4.2 of the plan) share one input/output contract, so they can be prototyped and swapped behind a common harness — this is what makes parallel work possible at all.
- The system alternates two modes that never run simultaneously (wake/sleep), hinged on a shared experience buffer — a natural interface boundary.
- Compute is Colab-Pro-constrained (no fixed GPU, session timeouts) — checkpointing/resumability is a Day-1 infra requirement, not a late addition.
- Two non-code deliverables (interactive dashboard, paper-shaped write-up) are first-class, not afterthoughts, and need their own owner.
- The plan already provides a real 34-week, 8-phase timeline (P0–P7) with exit criteria — this plan reuses it rather than inventing a new one.

**Scope boundaries the whole team should hold to (from the plan's own Section 2.3 — restated here so nobody accidentally scope-creeps):**

- **In scope, explicitly:** single-GPU, Colab-Pro-feasible MiniGrid/BabyAI grid-world tasks (e.g., navigate to a location, pick up an object, open a door, place an object in a goal zone); all three architecture variants at least on the cheap proxy benchmark; the hybrid learning rule; sleep-phase consolidation vs. the four other baselines; an **energy proxy** (spike counts / estimated synaptic operations vs. dense-network FLOPs — **not** physical power measurement on real hardware); the dashboard; and the write-up.
- **Out of scope, explicitly:** deployment on physical neuromorphic hardware (Loihi 2, SpiNNaker, BrainScaleS) or real robots; multi-GPU or large-scale training; full Atari-57-style suites or high-dimensional continuous-control humanoids; and any claim of biological realism beyond algorithmic analogy — this is a bio-inspired engineering study, not a neuroscience model-validation study. If a task board item starts drifting toward any of these, that's a scope-creep signal, not a sign of thoroughness.
- **Optional, time-permitting:** a generalization probe — procedurally-varied MiniGrid room layouts, to check whether conclusions hold when the environment is randomized rather than fixed (plan Section 3.3). Treated the same way as Variant C and RQ4 below: real work, but conditional on time remaining after the core deliverables land, and never allowed to block a phase exit criterion.

**Ambiguities in the plan and assumptions made to resolve them:**

| Ambiguity | Assumption made |
|---|---|
| The plan doesn't state team size or member skill levels. | Division follows the user's request for exactly 3 members, split by **technical responsibility**, not by skill — the plan gives no basis to assume anyone is stronger at SNNs vs. infra vs. RL. |
| The plan doesn't say whether Colab/Drive/W&B are shared accounts or per-member. | Assumed each member has their **own** Colab Pro session and Drive folder (to allow real parallel training) unified by a **single shared W&B project** as the source of truth. Flagged again in Section 24. |
| Variant C ("fully spiking transformer") is explicitly a stretch goal, contingent on time. | Treated as **conditional, jointly-owned** work between Member 1 (SNN side) and Member 2 (Transformer side), not a hard commitment baked into the base workload — this keeps the three workloads balanced regardless of whether it happens. |
| The plan doesn't specify tooling for the capstone report / paper draft. | Assumed the team drafts prose in **Overleaf or Google Docs** (real-time collaborative, no Git merge pain on paragraphs) and only commits the final exported PDF/`.tex`/`.md` to the repo — flagged as a recommendation, not invented as fact. |

---

# 2. Proposed Architecture

```text
neuroplast/
│
├── data/
│   ├── checkpoints/            # gitignored — model/optimizer/buffer state, synced to Google Drive
│   └── logged_runs/            # gitignored — local JSON exports; W&B is source of truth
│
├── notebooks/
│   ├── member1_snn/            # M1's scratch notebooks — never edited by others
│   ├── member2_agent/          # M2's scratch notebooks
│   └── member3_infra/          # M3's scratch notebooks
│
├── src/
│   ├── envs/                            # OWNER: Member 3
│   │   ├── minigrid_wrappers.py
│   │   ├── task_sequence.py             # continual 3–5 task ordering + task-boundary signal
│   │   └── task_configs/
│   │
│   ├── models/
│   │   ├── snn/                         # OWNER: Member 1
│   │   │   ├── lif_encoder.py
│   │   │   ├── stdp.py                  # local Hebbian/STDP update
│   │   │   └── variants/
│   │   │       ├── variant_a.py         # differentiable end-to-end
│   │   │       ├── variant_b.py         # STDP front-end + backprop core
│   │   │       └── variant_c.py         # stretch goal — JOINT M1+M2
│   │   ├── memory/                      # OWNER: Member 2
│   │   │   └── transformer_memory.py
│   │   └── policy_heads.py              # OWNER: Member 2
│   │
│   ├── agent/                           # OWNER: Member 2
│   │   ├── ppo.py / dqn.py
│   │   ├── buffer.py                    # experience buffer — the wake/sleep hinge
│   │   └── sleep.py                     # baseline distillation + neuro-inspired interleaved replay
│   │
│   ├── baselines/                       # OWNER: Member 2
│   │   ├── naive_finetune.py
│   │   ├── plain_replay.py
│   │   ├── ewc.py
│   │   └── isolation.py                 # parameter-isolation arm
│   │
│   ├── infra/                           # OWNER: Member 3
│   │   ├── checkpointing.py             # Drive-backed, resumable
│   │   ├── wandb_logger.py              # shared logging schema (Section 8)
│   │   └── config.py
│   │
│   ├── eval/                            # OWNER: Member 3
│   │   ├── forgetting.py
│   │   ├── transfer.py
│   │   ├── sample_efficiency.py         # env-steps-to-threshold — the RQ1 metric
│   │   ├── efficiency.py                # params/spike-ops vs. retained performance
│   │   └── energy_proxy.py              # spike counts vs. dense FLOPs
│   │
│   └── utils/                           # SHARED — low-churn, PR-only
│       └── seeding.py
│
├── dashboard/                           # OWNER: Member 3
│   ├── index.html
│   ├── src/                             # Recharts components
│   └── data/                            # generated JSON exports + a committed sample fixture
│
├── tests/
│   ├── test_snn.py, test_learning_rule.py            # Member 1
│   ├── test_agent.py, test_sleep.py, test_baselines.py  # Member 2
│   └── test_envs.py, test_eval_metrics.py             # Member 3
│
├── docs/
│   ├── capstone_report/     # drafted in Overleaf/Google Docs; final export committed here
│   └── paper_draft/
│
├── .github/workflows/       # OWNER: Member 3 — PR-only for others
├── requirements.txt         # SHARED — coordinate-only, single-line additions
├── .env.example
├── .gitignore
└── README.md                # OWNER: Member 3, PR-only, each member fills their own subsection
```

This gives each member their own top-level directories under `src/` — the SNN core, the agent/sleep/baselines, and envs/infra/eval/dashboard never share a file in normal operation. The only genuinely shared surfaces are `requirements.txt`, `README.md`, `.github/workflows/`, and the W&B logging schema — all handled explicitly in Section 16.

---

# 3. Module Breakdown

| Module | What it does / why required | Depends on | Depended on by | Complexity | Independent? | Git conflict risk |
|---|---|---|---|---|---|---|
| Env & continual task suite | Gymnasium/MiniGrid/BabyAI wrappers; defines the 3–5-task sequence and task-boundary signal | — | Everything | Medium | Yes — can start Day 1 | Low (own dir) |
| SNN perception encoder | LIF-neuron spiking encoder, surrogate-gradient trainable (snnTorch) | Env's observation shape (interface only) | Transformer memory, energy proxy | High | Yes, on a toy task | Low (own dir) |
| Hybrid learning rule (STDP + backprop mix) | Local Hebbian/STDP update combined with global gradient via swept α:β | SNN encoder | Architecture variants A/B/C | Highest | No — needs a working SNN encoder first | Low (own dir) |
| Architecture variants A/B/C | Three plasticity/backprop splits sharing one I/O contract | Learning rule, (Variant C also needs Transformer) | Phase-3 decision point | High (A/B), Highest (C) | A/B yes; C is joint | Low, except Variant C (shared file, coordinate via PR) |
| Transformer working-memory layer | Attends over a window of recent spikes for context | SNN encoder **output shape only** (can mock) | Full agent, policy head | Medium-High | Yes, against mock spike tensors | Low (own dir) |
| RL training loop (PPO/DQN) | CleanRL-style loop; wake-phase acting + learning | Env, encoder+memory+policy interface | Sleep phase, baselines, eval | Medium-High | Yes, with placeholder encoder | Low (own dir) |
| Experience buffer | Stores transitions; the wake/sleep hinge | — | Sleep phase, baselines | Medium | Yes | Low |
| Sleep-phase consolidation | Offline replay + distillation, then interleaved novel/familiar version | Buffer, trainable agent | Continual benchmark results, RQ2 | High | Partially — can build against mock buffer data first | Low (own dir) |
| Baseline arms (naive/replay/EWC/isolation) | Four comparison conditions reusing the training harness | RL training loop | Continual benchmark, RQ5 | Medium | Yes, once loop skeleton exists | Low (own dir) |
| Experiment tracking & checkpointing infra | W&B logging schema, Drive checkpoint/resume | — | Every other module (they all log/checkpoint) | Medium | Yes — Day 1 | Medium (schema is a cross-cutting **interface**, not a file — see Section 8) |
| Evaluation & metrics harness | Forgetting, transfer, **sample efficiency (env-steps-to-threshold — the RQ1 metric)**, param/compute efficiency, energy proxy, seed-variance stats | Logged data (can use mock JSON early) | Dashboard, write-up | Medium-High | Yes, against fixtures | Low (own dir) |
| Interactive dashboard (HTML + Recharts) | Renders pre-recorded results — task-switch timeline, forgetting curves, before/after-sleep replay | Logged JSON schema (interface only) | Demo deliverable | Medium | Yes, against a committed sample fixture | Low (own dir) |
| Capstone report / paper draft | Write-up around the pre-registered RQs | Results from all modules | — | Medium (writing, not code) | No — needs real results for final pass, but drafting can start early | High if done as one shared file → mitigated in Section 16 |

---

# 4. Member 1 — Complete Responsibility

**Primary responsibility:** The spiking perceptual core and the hybrid learning rule — the piece of NeuroPlast that answers RQ1 (does STDP help?) and RQ3 (energy/accuracy trade-off).

**Modules owned:** SNN perception encoder · hybrid Hebbian/STDP + gradient learning rule · Architecture Variants A and B (Variant C jointly with Member 2, conditional).

**Files owned:** `src/models/snn/*`, `src/models/snn/variants/variant_a.py`, `variant_b.py` (co-owns `variant_c.py` with M2), `tests/test_snn.py`, `tests/test_learning_rule.py`, `notebooks/member1_snn/`.

**Tasks:**
- P0: Stand up snnTorch (primary) + SpikingJelly (for later profiling only if needed); reproduce a known SNN-RL result on a toy task as the plan's own sanity check; begin the STDP/plasticity literature review (Section 6 of the plan is a starting point, not final).
- P1–P2: Build the LIF-neuron SNN encoder against the agreed observation-shape interface (Section 8); benchmark it standalone vs. a same-capacity dense CNN encoder on accuracy + spike sparsity (first pass at RQ3).
- P3: Implement the hybrid update rule `Δw = α·(local STDP term) + β·(global gradient term)`; prototype Variants A and B on the cheap single-task proxy benchmark; participate in the joint Phase-3 decision-point review (Section 19) that selects which variant(s) go forward.
- P4: Support integration issues as Member 2 attaches the Transformer memory layer on top of the selected variant.
- P5–P6: Tune STDP time constants and encoder stability as continual-task issues surface; run α:β and STDP-time-constant ablations; finalize the energy-proxy analysis (spike counts / estimated synaptic ops vs. dense FLOPs) for RQ3.
- P7: Write the SNN/learning-rule methods section and RQ1/RQ3 results for the capstone report and paper draft.

**Dependencies:** Needs the observation tensor shape and env action space from Member 3 early (Day 1 interface, not code). Needs nothing from Member 2 until P4 integration.

**What other members depend on:** A stable `encode(obs, T) -> spikes` function with a fixed output shape (Section 8, Interface 1) that Member 2's Transformer layer consumes; spike-count telemetry that Member 3's energy-proxy module reads.

**Deliverables:** Working SNN encoder + STDP module; Variant A/B implementations; RQ1 results (sample efficiency + final return, computed via Member 3's `eval/sample_efficiency.py`) and RQ3 results with ablations; energy-proxy numbers.

**Testing responsibility:** Unit tests on LIF neuron dynamics, surrogate-gradient correctness, STDP trace accumulation math (can be tested on CPU, per the plan's own Section 3.1 guidance — no GPU needed for this).

**Integration responsibility:** Verify the encoder's output shape/dtype never silently changes once Member 2 depends on it; flag any breaking change via PR, never a direct push.

---

# 5. Member 2 — Complete Responsibility

**Primary responsibility:** The agent itself — working memory, the RL training loop, the experience buffer, sleep-phase consolidation, and all four baseline comparison arms. This is the largest sustained engineering surface and directly answers RQ2 and RQ5.

**Modules owned:** Transformer working-memory layer · policy heads · PPO/DQN training loop · experience buffer · sleep-phase consolidation (both versions) · all four baseline arms · Variant C (jointly with M1, conditional).

**Files owned:** `src/models/memory/*`, `src/models/policy_heads.py`, `src/agent/*`, `src/baselines/*`, `src/models/snn/variants/variant_c.py` (co-owned), `tests/test_agent.py`, `tests/test_sleep.py`, `tests/test_baselines.py`, `notebooks/member2_agent/`.

**Tasks:**
- P0: Fork CleanRL's single-file PPO/DQN; build the agent skeleton against a **placeholder encoder** (simple CNN or random features) so this doesn't wait on Member 1's SNN; draft the experience-buffer interface.
- P1: Full conventional CNN+PPO/DQN baseline on the single-task benchmark using Member 3's env wrappers — this is the plan's own P1 exit criterion.
- P2: Continue refining the agent loop; start the simpler baseline arms (naive fine-tuning, plain replay) using the CNN encoder as a stand-in while the SNN stabilizes.
- P3: Support Variant A/B integration into the agent harness; help evaluate variant stability/performance on the proxy task at the joint decision-point review.
- P4: Integrate the Transformer working-memory layer on top of the selected variant(s) — the single biggest integration event with Member 1; full agent trains on the single-task benchmark (plan's P4 exit criterion).
- P5: Build sleep-phase consolidation (baseline distillation first, then the neuro-inspired interleaved-replay version); implement the EWC and parameter-isolation baselines; run the full continual multi-task benchmark across all 5 conditions — this is where RQ2 and RQ5 numbers come from.
- P6: Sleep-frequency ablations; seed-variance runs (≥3–5 seeds per condition, per the plan's own reliability bar) across all conditions.
- P7: Write the agent/sleep/baselines methods section and RQ2/RQ5 results.

**Dependencies:** Needs the env interface from Member 3 by Day 1. Needs the SNN encoder's stable output interface from Member 1 by P4 (can build and test everything before that against a placeholder). Needs the W&B/checkpoint schema from Member 3 early so training runs log correctly from the start.

**What other members depend on:** The `buffer.add()/sample()` interface (Section 8, Interface 4) that sleep and baselines both consume; the trained-run outputs (returns, forgetting curves, checkpoints) that Member 3's eval harness and dashboard are built on.

**Deliverables:** Full trained agent; 4 baseline implementations; sleep-phase module (both versions); RQ2/RQ5 results with forgetting/transfer numbers across all 5 conditions.

**Testing responsibility:** Agent forward/backward pass correctness, buffer sampling logic (uniform vs. interleaved), sleep-phase distillation loss correctness, baseline-arm isolation (e.g., isolation baseline truly freezes prior-task weights).

**Integration responsibility:** Verify the agent still trains correctly every time Member 1 changes the encoder interface; verify sleep/baseline runs log to the agreed W&B schema so Member 3's dashboard doesn't silently break.

---

# 6. Member 3 — Complete Responsibility

**Primary responsibility:** Everything that makes the other two members' work runnable, comparable, and presentable — environments, infra (checkpointing/tracking), the evaluation-metrics harness, and the results dashboard. This role is broad but low-conflict, and it's the one best positioned to start immediately and finish the project (dashboard + repro README).

**Modules owned:** Env wrappers & continual task sequence · experiment tracking (W&B) & checkpointing infra · evaluation/metrics harness · interactive results dashboard · CI config · README/reproducibility.

**Files owned:** `src/envs/*`, `src/infra/*`, `src/eval/*`, `dashboard/*`, `.github/workflows/*`, `tests/test_envs.py`, `tests/test_eval_metrics.py`, `notebooks/member3_infra/`, primary editor of `README.md`.

**Tasks:**
- P0: Set up Colab/Drive/W&B for the whole team; build MiniGrid/BabyAI env wrappers and the continual 3–5-task sequence + task-boundary signal; **drive the Day-1 joint meeting that locks the W&B logging schema** (Section 8, Interface 7) — the most important cross-cutting artifact in the whole plan.
- P1: Implement Drive-backed checkpoint/resume (model + optimizer + buffer state, saved every few minutes per the plan's own risk mitigation) and validate it against Member 2's baseline training loop.
- P2: Build the energy-proxy measurement module against Member 1's SNN encoder outputs; start the dashboard skeleton against a **committed mock/sample JSON fixture** so it doesn't wait on real results.
- P3: Sweep-config tooling in W&B so the team can run 3 variants × sweeps without manual bookkeeping.
- P4: Stress-test checkpointing on longer full-agent runs; begin forgetting/transfer/**sample-efficiency** metric code against synthetic data — sample efficiency (env-steps-to-threshold) is the concrete metric RQ1 is measured on, so it needs its own module (`eval/sample_efficiency.py`), not a footnote inside `efficiency.py`.
- P5: Real integration — compute forgetting, transfer, sample-efficiency, and parameter/compute efficiency metrics (RQ1, RQ2, RQ5) from Member 2's actual logged runs across all 5 conditions.
- P6: Assemble the interactive dashboard from real logged data: task-switching timeline, per-condition forgetting curves, parameter-count-vs-performance chart, and the before-sleep/after-sleep replay of the same evaluation episode. If time remains after the core RQ1/RQ2/RQ3/RQ5 numbers are in, support the stretch RQ4 comparison (Section 7's RQ mapping) and the optional generalization probe (Section 1) — neither is allowed to block the P6 exit criterion.
- P7: Finalize the public repo README (reproducing key results) and the dashboard link; assemble the infra/eval/reproducibility section of the write-up.

**Dependencies:** Needs nothing to start (Day 1 owner of env + infra). Needs real logged data from Member 2 by P5 for the real (non-mock) eval/dashboard pass; needs spike telemetry from Member 1 for the energy proxy.

**What other members depend on:** The env observation/action interface (Section 8, Interface 1) that Member 1's encoder and Member 2's agent both consume; the checkpoint format and W&B schema that every training run must comply with; the JSON export contract the dashboard reads.

**Deliverables:** Working env suite + task sequence; resumable checkpointing; W&B logging in place from Day 1; forgetting/transfer/efficiency/energy metrics for all 5 conditions; functional dashboard; reproducible README.

**Testing responsibility:** Env wrapper correctness (obs/action shapes, task-boundary signal firing correctly), checkpoint save/resume round-trip integrity, metric-computation correctness against known synthetic forgetting curves.

**Integration responsibility:** Own the CI config that runs all three members' test suites on every PR; verify a schema change to logging doesn't silently break the dashboard or the eval code (breaking changes go through PR review, never a silent edit).

---

# 7. Ownership Matrix

| Component | Member 1 | Member 2 | Member 3 |
|---|---:|---:|---:|
| Env & task suite | – | Consumer | **OWNER** |
| SNN encoder | **OWNER** | Consumer | Consumer (energy proxy) |
| Hybrid learning rule (STDP+backprop) | **OWNER** | – | – |
| Variants A / B | **OWNER** | Consumer | – |
| Variant C (stretch) | Contributor | Contributor | – |
| Transformer memory / policy head | Consumer | **OWNER** | – |
| RL training loop (PPO/DQN) | – | **OWNER** | Consumer (logs) |
| Experience buffer | – | **OWNER** | – |
| Sleep-phase consolidation | – | **OWNER** | Consumer (metrics) |
| Baseline arms (naive/replay/EWC/isolation) | – | **OWNER** | Consumer (metrics) |
| Checkpointing & W&B infra | Consumer | Consumer | **OWNER** |
| Eval/metrics harness | Contributor (energy data) | Contributor (run data) | **OWNER** |
| Dashboard | – | – | **OWNER** |
| Capstone report / paper draft | Shared | Shared | Shared |
| `requirements.txt` | Coordinate | Coordinate | Coordinate |
| `README.md` | Contributor | Contributor | **OWNER** |
| `.github/workflows/` | – | – | **OWNER** |
| Tests | Own module's tests | Own module's tests | Own module's tests |

**Research Question → Owner Mapping.** The plan's five pre-registered RQs (Section 2.2) map onto the ownership above as follows — spelled out explicitly so no RQ is silently dropped:

| RQ | Question (short form) | Primary owner(s) | Key module(s) | Status |
|---|---|---|---|---|
| RQ1 | Does STDP alongside gradient descent change sample efficiency or final return, vs. backprop alone? | **M1** | Hybrid learning rule; `eval/sample_efficiency.py` (M3) | Core |
| RQ2 | Does sleep-replay consolidation reduce forgetting vs. plain replay and EWC? | **M2** | Sleep phase; `eval/forgetting.py` (M3) | Core |
| RQ3 | Does the SNN encoder beat a same-capacity CNN on accuracy-vs-energy, with and without STDP? | **M1** (+ M3 for the energy-proxy computation) | SNN encoder; `eval/energy_proxy.py` | Core |
| RQ4 | Of Variants A/B/C, which offers the best continual-learning robustness per unit of compute — and does the ranking change across task types? | **M1 + M3, jointly** | Architecture variants; `eval/efficiency.py` | **Stretch, paper-facing** (the plan's own label) — pursued only if the Phase-3 decision point leaves enough remaining budget to run more than the selected variant(s) through the full continual pipeline. Not a hard commitment; kept off the critical path (see Task Board VAR-04). |
| RQ5 | Does NeuroPlast retain more performance per parameter than parameter isolation as task count grows, with positive transfer isolation can't achieve? | **M2** (+ M3 for the efficiency metric) | Baseline arms; `eval/efficiency.py` | Core. Concrete success bar (plan Section 4.3): at matched-or-lower total parameter count than isolation's running total, NeuroPlast retains *most* (not necessarily all) earlier-task performance and shows measurable positive transfer isolation cannot achieve by construction — isolation always wins on raw forgetting, so raw forgetting alone is not the metric to chase here. |

---

# 8. Interface Contracts

**1. Env → SNN encoder** *(Owner: M3 · Consumer: M1)*
```python
obs: torch.Tensor[B, C, H, W]  # MiniGrid partial-observation image
spikes = encode(obs, T=cfg.snn_timesteps)  # -> torch.Tensor[T, B, D]
```

**2. SNN encoder → Transformer memory** *(Owner: M1 · Consumer: M2)*
```python
context = TransformerMemory.forward(spike_window: Tensor[T, B, D]) -> Tensor[B, D_ctx]
```

**3. Transformer memory → Policy head** *(Owner: M2 internal)*
```python
action_logits, value = policy_head(context: Tensor[B, D_ctx])
```

**4. Agent ↔ Experience buffer** *(Owner: M2)*
```python
buffer.add(transition)
batch = buffer.sample(batch_size, mode="uniform" | "interleaved")
```

**5. Buffer → Sleep phase** *(Owner: M2)*
```python
updated_model, sleep_metrics = sleep_consolidate(model, buffer, config)
```

**6. Training loop ↔ Checkpointing** *(Owner: M3 · Consumer: M1, M2)*
```python
save_checkpoint(model_state, optimizer_state, buffer_state, step)
state = load_checkpoint(path)  # fully resumable, per the plan's Day-1 requirement
```

**7. All training/eval code → W&B logger** *(Owner: M3 · Consumer: everyone — locked Day 1)*
```python
log({"task_id": int, "condition": str, "return": float,
     "forgetting_task_i": float, "spike_count": int,
     "n_params": int, "seed": int}, step=global_step)
```
This schema is the single most important shared artifact in the project — a change here silently breaks the dashboard and eval code for everyone, so it's locked in a joint P0 meeting and changed only via PR.

**8. Logged data → Dashboard** *(Owner: M3)*
```python
export_dashboard_json(run_id) -> {
  "tasks": [...], "conditions": [...],
  "forgetting_curve": [...],
  "episodes": [{"before_sleep": [frames], "after_sleep": [frames]}]
}
```

**9. Task sequence → Sleep/baseline task-boundary signal** *(Owner: M3 · Consumer: M2)*
```python
task_sequence.current_task_id: int
task_sequence.on_task_switch(callback)
```

---

# 9. Dependency Graph

```text
Env & Task Suite (M3) ───────────────┐
                                      │
SNN Encoder (M1) ─────────┐          │
                           ▼          ▼
Hybrid Learning Rule (M1)  →  Transformer Memory + Policy Head (M2)
                                      │
                                      ▼
                          RL Training Loop / PPO-DQN (M2)
                                      │
                                      ▼
                          Experience Buffer (M2)
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
          Sleep Consolidation   Baseline Arms   Checkpointing/W&B (M3)
                (M2)                (M2)
                        └─────────────┬─────────────┘
                                      ▼
                     Continual Benchmark Results (all 5 conditions)
                                      │
                                      ▼
              Eval/Metrics Harness — forgetting, transfer, efficiency, energy (M3)
                                      │
                                      ▼
                Dashboard (M3)  +  Capstone Report / Paper (Shared)
```

**What can start immediately (Day 1, fully parallel):**
- M3: env wrappers, task-sequence scaffolding, W&B/Drive setup, dashboard skeleton against a mock JSON fixture.
- M1: SNN encoder prototyping on a toy task, snnTorch sanity-check reproduction.
- M2: agent/buffer skeleton against a **placeholder encoder** (plain CNN or random features) — not waiting on M1.

**What must wait:** Full hybrid-agent integration (P4) — mediated entirely through Interface 2, so it's a defined handoff, not shared-file editing. Real (non-mock) eval/dashboard numbers wait on Member 2's actual continual-benchmark runs (P5).

**What uses mock/sample data:** Sleep-phase development (mock buffer), eval harness and dashboard (fabricated JSON matching the Section 8 schema) — both built well before real results exist.

**What integrates late by design:** Variant C (stretch goal) and the paper-shaped write-up's final results section.

---

# 10. Parallel Development Plan

```text
                 ┌── Member 1: SNN encoder + hybrid STDP/backprop learning rule + Variants A/B
                 │
NeuroPlast ──────┼── Member 2: Transformer memory + PPO/DQN agent + buffer + sleep phase + baselines
                 │
                 └── Member 3: Envs + checkpointing/W&B infra + eval metrics + dashboard
```

All three tracks run genuinely in parallel from P0 through P3, converging at two explicit, scheduled sync points rather than an ad-hoc merge: the **P0 W&B-schema lock** (all three) and the **P3 architecture decision point** (M1 + M2, M3 supporting with sweep infra). The P4 encoder↔memory integration is the one moment M1 and M2 work in close contact — worth a scheduled pairing session, not just async PRs.

---

# 11. Git Branch Strategy

Simple trunk-based flow — no GitFlow, no long-lived `develop` branch, this is a 3-person team.

```text
main
│
├── feature/m1-snn-encoder
├── feature/m1-stdp-rule
├── feature/m1-variant-a
├── feature/m2-agent-loop
├── feature/m2-sleep-phase
├── feature/m2-baselines
├── feature/m3-env-wrappers
├── feature/m3-infra-checkpoint
└── feature/m3-dashboard
```

- Each member branches per **task**, not per person — a branch lives for days, not weeks, and is deleted after merge.
- `main` is always in a runnable state — the plan's own phase exit criteria (Section 4.6 of the plan) double as the bar for what's allowed onto `main`.
- No one pushes directly to `main`, ever — see Section 15.

---

# 12. Git Commands / Daily Workflow

```bash
# Start a new task
git checkout main
git pull origin main
git checkout -b feature/m1-stdp-rule

# Work in small commits
git add src/models/snn/stdp.py tests/test_learning_rule.py
git commit -m "feat(snn): add STDP eligibility trace accumulation"
git push origin feature/m1-stdp-rule

# Before opening a PR, sync with main
git checkout main
git pull origin main
git checkout feature/m1-stdp-rule
git merge main          # see Section 13 for merge-vs-rebase policy
# resolve any conflicts, re-run tests, then push and open the PR
```

Recommended cadence: commit at least once per working session, push at least daily — long-lived uncommitted work is the single easiest way to create a painful merge later.

---

# 13. Merge vs Rebase Strategy

**Team rule: merge `main` into your feature branch; never rebase a branch another person has also pulled.**

- Use `git merge main` (not rebase) to bring your branch up to date — it's safer for a team where not everyone is a Git expert, and it preserves an honest history of when integration happened.
- `git rebase` is fine **only** for cleaning up your own commit history on a branch nobody else has checked out, before opening a PR (e.g., squashing "wip wip wip" commits).
- Never rebase a shared/pushed branch that a teammate might have pulled — it rewrites history out from under them.
- Never force-push to `main` or to a branch someone else is also working on. If a force-push feels necessary, stop and ask in the team channel first.
- PR merges into `main` use a standard merge commit (or "squash and merge" if the team prefers a cleaner `main` history) — pick one and stay consistent.

---

# 14. Commit Convention

Small, logical, conventional-commit-style messages:

```text
feat(snn): add LIF encoder forward pass
feat(agent): implement PPO rollout buffer
feat(sleep): add distillation-based consolidation loss
feat(infra): add resumable checkpoint save/load
fix(envs): correct task-boundary signal off-by-one
test(snn): add STDP trace accumulation unit test
docs(readme): add reproduction instructions
```

**Avoid:** `finished project`, `changes`, `final code`, `updated everything`, `fix stuff`. If a commit message needs "and" three times, it should be two commits.

---

# 15. Pull Request Rules

1. Never push directly to `main`.
2. Finish work on a `feature/*` branch scoped to one task.
3. Merge the latest `main` into your branch before opening the PR (Section 13).
4. Run the relevant test suite locally (`pytest tests/test_<your_module>.py`) before opening.
5. Open the PR with a short description of what changed and which interface (Section 8) it touches, if any.
6. **Cross-review, not self-review:** M1 reviews M2's PRs, M2 reviews M3's PRs, M3 reviews M1's PRs — this rotation means everyone reviews and is reviewed, and no one only ever reviews their closest collaborator.
7. Any PR that changes a Section-8 interface or the W&B schema needs sign-off from every member who consumes it, not just the assigned reviewer.
8. Resolve conflicts on your own branch before merging, not by force-pushing over the reviewer's comments.
9. Merge only when CI (Member 3's workflow) is green.
10. Delete the feature branch after a successful merge.

---

# 16. Merge Conflict Prevention

| File / Directory | Owner | Others allowed to modify? | Reason / strategy |
|---|---|---|---|
| `src/models/snn/**` | M1 | No — PR only | Isolated module, single owner |
| `src/models/memory/**`, `src/agent/**`, `src/baselines/**` | M2 | No — PR only | Isolated module, single owner |
| `src/envs/**`, `src/infra/**`, `src/eval/**`, `dashboard/**` | M3 | No — PR only | Isolated module, single owner |
| `src/models/snn/variants/variant_c.py` | M1 + M2 (joint) | Yes, by both named owners | Genuinely shared logic (spiking attention) — coordinate directly, small commits, PR review by the third member |
| `requirements.txt` | Shared | Yes, but one dependency per commit, never a bulk rewrite | Classic multi-line conflict file — append-only edits, never reformat |
| `README.md` | M3 | Others via PR, editing only their own subsection | One person keeps structure consistent; others add content, not restructure |
| `.github/workflows/*` | M3 | No — PR only | CI misconfiguration affects everyone; single point of control |
| W&B logging schema (`src/infra/wandb_logger.py`) | M3 | Changes require sign-off from M1 and M2 | Breaking this silently breaks eval + dashboard for everyone (Section 8, Interface 7) |
| `notebooks/*` | Per-member subfolder | No cross-editing | Notebooks are near-impossible to text-merge (JSON with cell outputs) — each member has their own subfolder, and nothing in `notebooks/` is ever imported by `src/` |
| `docs/capstone_report/`, `docs/paper_draft/` | Shared | Draft in Overleaf/Google Docs, not Git | Prose merge conflicts are painful and low-value to resolve in Git; commit only the final export |
| `.gitignore`, `.env.example` | M3 | PR only | Small, rarely-changed, but affects everyone's environment |

---

# 17. ML-Specific Risks

**Version-control strategy for ML artifacts** (what's committed vs. ignored vs. external):

| Artifact | Treatment |
|---|---|
| Raw/processed MiniGrid configs, task-sequence definitions | Commit to Git (small, text-based) |
| Trained model checkpoints (`.pt`) | **Not** committed — stored in Google Drive, referenced by path/run-ID in W&B |
| Replay buffer state | Not committed — Drive, same as checkpoints |
| W&B logs / metrics | Not committed as raw logs — W&B is the source of truth; only small JSON exports for the dashboard are committed |
| Jupyter notebooks | Committed, but only in per-member scratch folders — never used as the source of truth for reusable code (that lives in `src/`) |
| Dashboard sample fixture | Committed (small, needed so the dashboard builds without real results) |

Git LFS or DVC is **not** recommended here — the project explicitly avoids committing large binaries at all (Drive + W&B cover that need), so there's nothing LFS/DVC would actually manage. Introducing them would be unnecessary tooling for a 3-person capstone.

**Problem → Why it happens → Prevention → Recovery**

- **Dataset/task-config version mismatch** (M1's local proxy-task config drifts from M2's) → because each member can edit `task_configs/` locally to iterate faster → Prevention: `task_configs/` is owned by M3, changes only via PR → Recovery: pin a config hash in W&B run metadata so any mismatch is caught immediately.
- **Different preprocessing/observation normalization between M1's encoder tests and M2's full-agent runs** → because M1 prototypes standalone before Transformer integration → Prevention: normalization lives in Member 3's env wrapper, not duplicated in `src/models/snn/` → Recovery: a shared `preprocess()` call, single source of truth.
- **Different random seeds silently changing SNN spike patterns** → because SNNs are timestep-stochastic → Prevention: seed logged as a required W&B field (Interface 7) → Recovery: re-run with the logged seed to reproduce.
- **Incompatible serialized models across snnTorch/SpikingJelly if Variant C mixes libraries** → Prevention: standardize on snnTorch by default, only port to SpikingJelly if profiling shows a real bottleneck (per the plan's own recommendation) → Recovery: keep both encoders behind the same Interface-1 contract so swapping libraries doesn't touch consumer code.
- **Notebook conflicts** → covered structurally in Section 16 (per-member folders, notebooks never imported).
- **Data leakage across the continual task boundary** (e.g., sleep-phase replay accidentally sampling "future" task data) → because the buffer holds all history → Prevention: `task_sequence.current_task_id` (Interface 9) gates what's eligible for replay at each point → Recovery: an eval-harness sanity check (M3) that asserts no post-boundary data appears in a pre-boundary evaluation.
- **Core hypothesis turns out negative** (STDP doesn't help) → not a Git/process risk, but a real project risk called out in the plan itself (Section 4.8) → Prevention: RQs are pre-registered (plan Section 2.2) precisely so a clean negative result is still a valid, reportable outcome, not a failure to hide.

---

# 18. Testing Strategy

| Test area | Owner | What's tested | Merge-blocking? |
|---|---|---|---|
| LIF neuron dynamics, surrogate gradient, STDP trace math | M1 | Unit tests, CPU-only (per plan Section 3.1 — no GPU needed) | Yes |
| Variant A/B forward pass shape/contract compliance | M1 | Unit tests against Interface 1/2 | Yes |
| Transformer memory forward pass, policy head output shapes | M2 | Unit tests | Yes |
| Buffer add/sample (uniform + interleaved) | M2 | Unit tests | Yes |
| Sleep-phase distillation loss correctness | M2 | Unit tests, small synthetic batch | Yes |
| Baseline-arm isolation (frozen weights truly frozen, EWC penalty computed correctly) | M2 | Unit tests | Yes |
| Env wrapper obs/action shapes, task-boundary signal | M3 | Unit tests | Yes |
| Checkpoint save/resume round-trip | M3 | Integration test — save, kill process, reload, confirm identical state | Yes |
| Forgetting/transfer/sample-efficiency/param-efficiency metric computation | M3 | Unit tests against known synthetic forgetting curves and a synthetic return trajectory with a known steps-to-threshold | Yes |
| Full encoder→memory→policy pipeline | M1 + M2 jointly | Integration test, small proxy task | Yes, at P4 integration |
| Full wake→sleep→wake continual-task cycle | M2 + M3 jointly | End-to-end integration test on the proxy benchmark | Yes, at P5 integration |
| Dashboard renders correctly from the sample JSON fixture | M3 | Manual/visual check + a schema-validation test | Yes |

**Minimum bar before merging to `main`:** the owning member's unit tests pass, and CI (Member 3's workflow) runs the full `tests/` suite green — a PR that breaks another member's tests does not merge until fixed, regardless of whose module caused it.

---

# 19. Integration Plan

Integration is scheduled at the plan's own phase boundaries, not left to the end:

- **Phase 0 (Interface lock):** All three sign off on the observation/action interface (1), the W&B schema (7), and the checkpoint format (6). Nothing downstream should need to change these later.
- **Phase 1–2 (Independent build):** M1 builds the SNN encoder standalone; M2 builds the full agent loop against a placeholder encoder; M3 builds env + infra + starts eval/dashboard against mock data. No cross-file dependencies yet.
- **Phase 3 (Decision-point integration):** Joint M1+M2 review of Variants A/B (+C if pursued) on the cheap proxy task; M3 provides the sweep/logging infra to make the comparison fair. One (or two) variants selected — documented in a PR description, not just a chat message.
- **Phase 4 (Encoder ↔ Agent integration):** M2 swaps the placeholder encoder for M1's real SNN encoder behind Interface 2. This is the first real cross-owner integration — scheduled as a paired session, tested against the single-task benchmark exit criterion.
- **Phase 5 (Full continual-system integration):** M2's sleep phase + baseline arms run against M3's real continual task suite; M3's eval harness switches from mock to real logged data.
- **Phase 6 (Results integration):** M3 assembles the dashboard from real data across all 5 conditions; M1 and M2 feed final ablation numbers into it.
- **Phase 7 (Write-up integration):** each member's methods/results section merges into one report; M3 finalizes the reproducible README.

---

# 20. Development Timeline

Reusing the plan's own 34-week, 8-phase schedule (Section 4.6) rather than inventing a new one:

| Phase (weeks) | Member 1 | Member 2 | Member 3 | Integration checkpoint |
|---|---|---|---|---|
| **P0** (1–3) | snnTorch setup; toy SNN-RL sanity check; STDP lit review | CleanRL fork; agent skeleton vs. placeholder encoder; buffer interface draft | Colab/Drive/W&B setup; env wrappers; **drives the W&B schema lock meeting** | Tooling runs end-to-end on a toy task |
| **P1** (3–7) | SNN encoder vs. CNN, standalone (RQ3 first pass) | Conventional CNN+PPO/DQN baseline on single-task benchmark | Checkpoint/resume implemented & validated against M2's runs | Baseline reaches documented score |
| **P2** (7–12) | SNN encoder finalized, benchmarked | Continue agent loop; start naive/plain-replay baselines | Energy-proxy module; dashboard skeleton vs. mock JSON | SNN trains stably, efficiency measured |
| **P3** (12–17) | Hybrid STDP+backprop rule; prototype Variants A/B(/C) | Support variant integration into harness | Sweep-config tooling for 3-variant comparison | **Joint decision point** — variant(s) selected, documented |
| **P4** (17–22) | Support encoder integration issues | Integrate Transformer memory atop selected variant(s); full agent on single-task benchmark | Stress-test checkpointing on longer runs; start forgetting/transfer code vs. synthetic data | Full agent matches/approaches baseline |
| **P5** (22–28) | Support STDP tuning for continual tasks; ablation prep | Sleep-phase (both versions); EWC + isolation baselines; full continual benchmark, all 5 conditions | Real forgetting/transfer/efficiency metrics from M2's runs | Forgetting + efficiency metrics computed (RQ2, RQ5) |
| **P6** (28–31) | α:β and STDP-time-constant ablations; finalize energy proxy (RQ3) | Sleep-frequency ablations; seed-variance runs | Assemble interactive dashboard from real data | All RQs answered; dashboard functional |
| **P7** (31–34) | Write SNN/learning-rule methods + RQ1/RQ3 results | Write agent/sleep/baselines methods + RQ2/RQ5 results | Finalize README + repro instructions + dashboard link | Report submitted; repo public with working dashboard |

---

# 21. Task Board

| ID | Task | Owner | Priority | Dependency | Deliverable | Status |
|---|---|---|---|---|---|---|
| ENV-01 | MiniGrid/BabyAI wrappers | M3 | High | — | `src/envs/minigrid_wrappers.py` | Todo |
| ENV-02 | Continual 3–5 task sequence + boundary signal | M3 | High | ENV-01 | `task_sequence.py` | Todo |
| INFRA-01 | W&B project + logging schema (locked jointly) | M3 | Critical | — | Interface 7 signed off | Todo |
| INFRA-02 | Drive-backed resumable checkpointing | M3 | High | — | `checkpointing.py` + round-trip test | Todo |
| SNN-01 | Reproduce toy SNN-RL sanity check | M1 | High | — | P0 exit criterion | Todo |
| SNN-02 | LIF encoder against Interface 1 | M1 | Critical | ENV-01 | `lif_encoder.py` | Todo |
| SNN-03 | SNN vs. CNN encoder benchmark (RQ3 pass 1) | M1 | High | SNN-02 | RQ3 partial results | Todo |
| LEARN-01 | STDP eligibility trace module | M1 | Critical | SNN-02 | `stdp.py` | Todo |
| LEARN-02 | Hybrid Δw = α·local + β·global rule | M1 | Critical | LEARN-01 | mixing-weight sweep config | Todo |
| VAR-01 | Variant A implementation | M1 | High | LEARN-02 | `variant_a.py` | Todo |
| VAR-02 | Variant B implementation | M1 | High | LEARN-02 | `variant_b.py` | Todo |
| VAR-03 | Variant C (stretch, joint) | M1+M2 | Low | VAR-01/02, MEM-01 | `variant_c.py` | Todo |
| VAR-04 | RQ4 multi-variant continual-robustness comparison (stretch, paper-facing) | M1+M3 | Low | VAR-01/02(/03), EVAL-03 | RQ4 results, if time permits | Todo |
| ENV-03 | Generalization probe: procedurally-varied MiniGrid room layouts (optional) | M3 | Low | ENV-02 | Generalization results, if time permits | Todo |
| AGENT-01 | PPO/DQN loop vs. placeholder encoder | M2 | Critical | — | agent skeleton | Todo |
| AGENT-02 | Experience buffer (uniform + interleaved sample) | M2 | High | — | `buffer.py` + Interface 4 | Todo |
| MEM-01 | Transformer working-memory layer vs. mock spikes | M2 | High | Interface 2 spec | `transformer_memory.py` | Todo |
| BASE-01 | CNN+PPO/DQN conventional baseline | M2 | Critical | AGENT-01, ENV-01 | P1 exit criterion | Todo |
| BASE-02 | Naive sequential fine-tuning arm | M2 | Medium | AGENT-01 | `naive_finetune.py` | Todo |
| BASE-03 | Plain experience replay arm | M2 | Medium | AGENT-02 | `plain_replay.py` | Todo |
| BASE-04 | EWC-style regularization arm | M2 | Medium | AGENT-01 | `ewc.py` | Todo |
| BASE-05 | Parameter-isolation arm | M2 | Medium | AGENT-01 | `isolation.py` | Todo |
| SLEEP-01 | Baseline distillation sleep phase | M2 | Critical | AGENT-02 | `sleep.py` v1 | Todo |
| SLEEP-02 | Neuro-inspired interleaved replay (RQ2) | M2 | High | SLEEP-01 | `sleep.py` v2 | Todo |
| EVAL-01 | Forgetting metric | M3 | High | Interface 7 | `forgetting.py` | Todo |
| EVAL-02 | Backward/forward transfer metric | M3 | High | Interface 7 | `transfer.py` | Todo |
| EVAL-03 | Parameter/compute efficiency metric | M3 | Medium | Interface 7 | `efficiency.py` | Todo |
| EVAL-04 | Energy proxy (spike counts vs. FLOPs) | M3 | Medium | SNN telemetry | `energy_proxy.py` | Todo |
| EVAL-05 | Sample-efficiency metric — env-steps-to-threshold (RQ1) | M3 | High | Interface 7 | `sample_efficiency.py` | Todo |
| DASH-01 | Dashboard skeleton vs. mock JSON fixture | M3 | Medium | Interface 8 spec | `dashboard/` runs | Todo |
| DASH-02 | Dashboard wired to real logged data | M3 | High | EVAL-01..04, real runs | Functional dashboard | Todo |
| TEST-01..09 | Unit/integration tests per module | Each owner | High | Respective module | `tests/*` | Todo |
| DOC-01 | Capstone report draft (per-section) | Shared | Medium | Real results | `docs/capstone_report/` | Todo |
| DOC-02 | Paper-shaped draft for CoLLAs/workshop | Shared | Medium | DOC-01 | `docs/paper_draft/` | Todo |
| DOC-03 | Reproducible README | M3 | High | Working repo | `README.md` | Todo |

---

# 22. Risk Register

| Risk | Probability | Impact | Prevention | Owner |
|---|---|---|---|---|
| Surrogate-gradient SNN training unstable / spikes vanish | Medium-High | High | Well-tested LIF neurons with adaptive thresholds; start shallow (plan's own mitigation); if instability persists, investigate CaRe-BN-style normalization for SNN-RL stabilization (plan Section 6 lit review) | M1 |
| Exactly reproducing sleep-replay literature setups (e.g., interleaved-replay, REM-like replay papers cited in plan Section 6) is nontrivial | Medium | Medium | Ship the simple distillation-based sleep phase first (SLEEP-01); add the neuro-inspired interleaved novel/familiar version only once that baseline works (SLEEP-02) — plan's own mitigation | M2 |
| Colab disconnects lose training progress | High | High | Resumable checkpointing to Drive from Day 1 (INFRA-02), short individual runs | M3 (infra), M2 (uses it) |
| Three architecture variants triple engineering load | High | Medium | Variant A as shared "spine," reuse >80% of code for B; Phase-3 decision point cuts scope early | M1 (+ M2 for the joint call) |
| Hybrid-rule hyperparameter search space (α:β, STDP constants) explodes | Medium | Medium | Small grid/random search on the cheap proxy task first, before scaling to full benchmark | M1 |
| Core hypothesis (STDP helps) turns out negative | Medium | Low (still reportable) | RQs pre-registered; negative result is a valid capstone outcome | M1 + M2 |
| W&B schema changes silently break dashboard/eval | Medium | High | Schema locked jointly at P0; changes require sign-off from all consumers (Section 15, rule 7) | M3 |
| Two members unnecessarily edit the same file | Low (by design) | Medium | Per-module directory ownership (Section 16); PR-only on shared files | All |
| Encoder↔Transformer integration (P4) surfaces interface mismatches | Medium | Medium | Interface 2 fixed in Section 8 before either side writes real code; scheduled pairing session at P4 | M1 + M2 |
| One member becomes a bottleneck (e.g., M2's agent loop needed by everyone) | Medium | High | M2's loop is buildable against a placeholder encoder from Day 1 (Section 9) so no one waits on it | M2 (mitigated structurally) |
| Notebook merge conflicts | Low (by design) | Low | Per-member notebook subfolders; notebooks never imported by `src/` | All |
| Live demo fails in front of an audience | Low | Medium | Dashboard replays pre-recorded results, no live inference (plan's own mitigation) | M3 |
| Report/paper drafted as one shared Git file causes prose merge conflicts | Medium | Low | Draft in Overleaf/Google Docs, commit only final export | All |
| Last-minute integration crunch | Low (by design) | High | Integration scheduled at every phase boundary (Section 19), not deferred to P7 | All |

---

# 23. Final Team Rules

1. Never push directly to `main`.
2. Every task gets its own short-lived feature branch.
3. Every module has exactly one owner (Section 7) — don't edit another member's files without coordinating first.
4. Keep commits small and use conventional messages (Section 14).
5. Pull the latest `main` into your branch regularly, not just before a PR.
6. Interfaces (Section 8) are agreed *before* implementation, especially the W&B schema — it's locked Day 1.
7. Never commit secrets, API keys, or `.env` values — use `.env.example`.
8. Never commit trained checkpoints or raw logs — Drive and W&B are the source of truth (Section 17).
9. Never force-push to `main` or a shared branch.
10. Every PR gets reviewed by the rotation partner (Section 15), never self-merged.
11. Run tests locally before opening a PR; CI must be green before merging.
12. Integrate at every phase boundary — don't let it pile up for P7 (Section 19).
13. Keep `main` in a runnable state at all times.
14. If you're blocked on another member's module, work against a mock/placeholder first — don't sit idle (Section 9).
15. A negative or null research result is a legitimate outcome — document it as rigorously as a positive one.

---

# 24. Recommended First-Day Setup

**Environment (reproducible across all three machines):**
- Python 3.11 (matches current PyTorch/snnTorch/SpikingJelly support).
- One shared `requirements.txt` pinning: `torch`, `snntorch`, `spikingjelly` (only imported once actually needed), `gymnasium`, `minigrid`, `wandb`, plus test tooling (`pytest`).
- Each member creates their own virtual environment (`python -m venv .venv`) from the same `requirements.txt` — never a personally-modified copy.
- `.env.example` covers the W&B API key and any Drive-mount path variables; each member copies it to a local `.env` (gitignored) with their own credentials.

**Day 1 checklist, in order:**
1. All three create the GitHub repo (or fork), set branch protection on `main` (no direct pushes, PR required).
2. Create the folder structure from Section 2; commit it empty (with `.gitkeep` where needed) as the very first PR, reviewed by all three.
3. **Joint 30–60 minute meeting to lock the W&B logging schema (Interface 7) and the observation/action interface (Interface 1)** — this is the single highest-leverage thing the team can do on Day 1, since two of the three members build against these interfaces before touching each other's code.
4. Each member sets up their own Colab Pro account + Google Drive folder, connected to the **shared** W&B project.
5. M3 stands up the env wrappers and a minimal W&B "hello world" log so everyone can confirm their credentials work.
6. M1 reproduces the plan's own P0 sanity check (a known SNN-RL result on a toy task) to confirm the SNN toolchain works locally.
7. M2 forks CleanRL and gets a placeholder-encoder PPO loop running end-to-end on one MiniGrid task, confirming the full training→logging→checkpoint path works before any real modeling begins.
8. Agree the PR review rotation (Section 15, rule 6) and the commit-message convention (Section 14) explicitly, in writing, in the README.
9. Commit this plan itself into the repo at `docs/TEAM_PLAN.md` — the three agent prompts in Section 25 below assume it lives there, since each prompt tells the coding agent to read it before writing code.

---

# 25. AI Coding Agent Prompts — One Per Member

Each block below is a **complete, standalone prompt** one member pastes as the first message to their own AI coding agent (e.g., Claude Code) in their own checkout of the repo. They are written so a fresh agent session — with no other context — can pick up exactly the right scope, respect every other member's ownership boundaries, and self-check against the plan's own exit criteria.

**Before using these:** commit this document into the repo at `docs/TEAM_PLAN.md` (Section 24, Day-1 checklist item 2) so the "read the plan first" instruction in each prompt actually resolves to something. Each member should run their prompt in a **separate** agent session against their own branch — never paste more than one member's prompt into the same session, since that reintroduces exactly the cross-file editing this plan is designed to prevent.

## 25.1 Member 1 Prompt — SNN Perception + Hybrid Learning Rule

````
You are acting as an AI coding agent for Member 1 on the NeuroPlast capstone project: a hybrid
spiking-neural-network (SNN) / Transformer reinforcement-learning agent that combines
Hebbian/STDP local learning with gradient-descent backprop, plus a periodic "sleep" phase that
consolidates memory to fight catastrophic forgetting. It is benchmarked on a continual
MiniGrid/BabyAI task sequence against four baselines.

BEFORE WRITING ANY CODE: read docs/TEAM_PLAN.md in this repository in full. If it is not
present, stop and ask the user to add it — do not invent scope, file layout, or interfaces from
scratch. Everything below assumes that document as the source of truth; if anything here
conflicts with it, the plan document wins and you should flag the conflict to the user.

## Your role
You own the SNN perceptual core and the hybrid Hebbian/STDP + gradient-descent learning rule.
Your work answers RQ1 (does adding local STDP alongside gradient descent change sample
efficiency or final return, vs. backprop alone?) and RQ3 (does the SNN encoder offer a better
accuracy-vs-estimated-energy trade-off than a same-capacity dense CNN encoder, and does that
trade-off survive once STDP is added?).

## Files you own — edit freely, no need to ask
- src/models/snn/lif_encoder.py
- src/models/snn/stdp.py
- src/models/snn/variants/variant_a.py
- src/models/snn/variants/variant_b.py
- tests/test_snn.py
- tests/test_learning_rule.py
- notebooks/member1_snn/  (scratch only — nothing here is ever imported by src/)

## Files you may touch only jointly, coordinated via PR
- src/models/snn/variants/variant_c.py — shared with Member 2 (fully spiking transformer,
  stretch goal, conditional on time remaining). Do not start this until Members 1 and 2 have
  both agreed to pursue it. Small commits, request review from Member 3.

## Files you must NEVER edit directly — if you need a change here, open a PR and tag the owner
- Anything under src/envs/, src/agent/, src/models/memory/, src/baselines/, src/infra/,
  src/eval/, dashboard/ (owned by Member 2 or Member 3)
- requirements.txt — append-only, one new dependency per commit, never reformat the whole file
- README.md, .github/workflows/* — owned by Member 3
- The W&B logging schema in src/infra/wandb_logger.py — you may call it, but you must not
  change what fields it expects without sign-off from Member 2 and Member 3 first (per the
  plan's PR rule 7)

## Interface contract you must implement exactly (do not silently change the shape)
Input (owned by Member 3 — do not assume a different shape without checking):
    obs: torch.Tensor[B, C, H, W]   # MiniGrid partial-observation image

Output (consumed by Member 2's Transformer memory layer):
    def encode(obs: torch.Tensor, T: int) -> torch.Tensor:
        """Returns spikes: torch.Tensor[T, B, D]"""

Every training run must log these fields via Member 3's W&B logger: spike_count, seed,
n_params, alongside whatever standard RL fields Member 2's loop already logs.

## Tasks, in the plan's own phase order — do not get ahead of what Members 2/3 have ready
1. (P0) Set up snnTorch (the primary SNN library per the plan). Reproduce a known, simple
   SNN-RL result on a toy task as a sanity check — this is a hard P0 exit criterion.
2. (P1-P2) Build lif_encoder.py: a LIF-neuron SNN encoder trained via surrogate-gradient
   backprop, matching encode(obs, T) above. Use T (spike-simulation timesteps) in the 4-16
   range — do not default higher; it multiplies compute/memory roughly T x and is the single
   biggest hidden cost in this project. Benchmark standalone against a same-capacity dense CNN
   encoder on accuracy and spike sparsity (RQ3, first pass). Your later RQ1 numbers (sample
   efficiency and final return, STDP-on vs. STDP-off) will be computed by Member 3's
   eval/sample_efficiency.py against your logged runs — make sure your training loop logs
   return per step under the agreed W&B schema so that module has something to consume.
3. (P3) Build stdp.py: a local Hebbian/STDP eligibility-trace update from pre/post-synaptic
   spike timing only, with no dependence on any downstream loss. Combine with the gradient path
   as Δw = α·(local STDP term) + β·(global gradient term), where α:β is a swept hyperparameter
   exposed via config, never a hardcoded constant. Implement variant_a.py (differentiable
   end-to-end) and variant_b.py (STDP front-end + backprop core), sharing the encode() contract
   so they're swappable behind Member 2's training harness. Run both on the cheap single-task
   proxy benchmark and bring stability/performance numbers to the joint Phase-3 decision-point
   review with Member 2 — do not unilaterally pick a variant; document the joint decision in a
   PR description.
4. (P4) Support Member 2 integrating the Transformer memory layer on top of your encoder. If
   integration reveals a shape/dtype mismatch, fix it on your side and flag the change loudly —
   don't leave Member 2 to work around it silently.
5. (P5-P6) Tune STDP time constants and encoder stability as continual-task issues surface.
   Run ablations across α:β and STDP time constants. Finalize the energy-proxy analysis (spike
   counts / estimated synaptic operations vs. dense FLOPs at matched parameter count) — make
   sure spike-count telemetry is actually logged, since Member 3's eval/energy_proxy.py depends
   on it existing in W&B, not just being computed and discarded.
6. (P7) Write the SNN/learning-rule methods section and RQ1/RQ3 results for the report (drafted
   in Overleaf/Google Docs per the plan, not as a Git-tracked prose file).

## Cautions specific to this module
- Colab has no guaranteed GPU model and sessions time out (~12-24h, often less). Use
  torch.cuda.amp mixed precision, keep batch size and T modest, and checkpoint/resume through
  Member 3's checkpointing.py rather than writing your own ad hoc save/load logic.
- Surrogate-gradient SNN training is prone to instability (vanishing spikes). Use well-tested
  LIF neurons with adaptive thresholds; start shallow before scaling depth.
- Always log the random seed — SNNs are timestep-stochastic, and an unlogged seed makes a
  result impossible to reproduce or debug later. The plan requires >=3-5 seeds per condition.
- Do unit tests (LIF dynamics, surrogate-gradient correctness, STDP trace math) on CPU where
  possible — reserve Colab GPU-hours for real training runs.
- Start every new capability in snnTorch first; only port a module to SpikingJelly if profiling
  shows it's an actual bottleneck.
- Never change the encode() output shape without a PR that Member 2 explicitly signs off on.

## Git workflow
- Branch per task off main: feature/m1-snn-encoder, feature/m1-stdp-rule, feature/m1-variant-a.
- Small, conventional commits: "feat(snn): add LIF encoder forward pass",
  "fix(snn): correct surrogate gradient scale", "test(snn): add STDP trace accumulation test".
  Never "wip" or "changes".
- Merge main into your branch (don't rebase a branch anyone else may have pulled) before
  opening a PR.
- Never push directly to main. Open a PR; per the plan's review rotation, Member 3 reviews your
  PRs, you review Member 2's.
- Run `pytest tests/test_snn.py tests/test_learning_rule.py` locally and confirm it's green
  before opening any PR.
- Never commit trained checkpoints, .pt files, or raw logs — those live in Google Drive / W&B.

## Before you write any production code
Confirm you understand the encode(obs, T) -> spikes[T, B, D] contract, and ask the user to
confirm: the actual value of T, the real observation shape from Member 3's env wrapper, and
whether the W&B schema has already been locked (per the plan's Day-1 requirement). If the
schema isn't locked yet, say so and suggest that happen before you write logging code against it.
````

## 25.2 Member 2 Prompt — Agent, Sleep Consolidation, and Baselines

````
You are acting as an AI coding agent for Member 2 on the NeuroPlast capstone project: a hybrid
spiking-neural-network (SNN) / Transformer reinforcement-learning agent that combines
Hebbian/STDP local learning with gradient-descent backprop, plus a periodic "sleep" phase that
consolidates memory to fight catastrophic forgetting. It is benchmarked on a continual
MiniGrid/BabyAI task sequence against four baselines.

BEFORE WRITING ANY CODE: read docs/TEAM_PLAN.md in this repository in full. If it is not
present, stop and ask the user to add it — do not invent scope, file layout, or interfaces from
scratch. Everything below assumes that document as the source of truth; if anything here
conflicts with it, the plan document wins and you should flag the conflict to the user.

## Your role
You own the agent itself: the Transformer working-memory layer, the RL training loop
(PPO/DQN), the experience buffer, sleep-phase consolidation, and all four baseline comparison
arms (naive fine-tuning, plain replay, EWC, parameter isolation). This is the largest sustained
engineering surface in the project and directly answers RQ2 (does sleep-replay consolidation
reduce catastrophic forgetting vs. plain replay and EWC?) and RQ5 (does NeuroPlast retain more
performance per parameter than a parameter-isolation baseline as task count grows, and does it
show positive transfer that isolation cannot?).

## Files you own — edit freely, no need to ask
- src/models/memory/transformer_memory.py
- src/models/policy_heads.py
- src/agent/ppo.py, src/agent/dqn.py, src/agent/buffer.py, src/agent/sleep.py
- src/baselines/naive_finetune.py, plain_replay.py, ewc.py, isolation.py
- tests/test_agent.py, tests/test_sleep.py, tests/test_baselines.py
- notebooks/member2_agent/  (scratch only — nothing here is ever imported by src/)

## Files you may touch only jointly, coordinated via PR
- src/models/snn/variants/variant_c.py — shared with Member 1 (fully spiking transformer,
  stretch goal, conditional on time remaining). Do not start this until Members 1 and 2 have
  both agreed to pursue it. Small commits, request review from Member 3.

## Files you must NEVER edit directly — if you need a change here, open a PR and tag the owner
- Anything under src/envs/, src/models/snn/ (except variant_c.py above), src/infra/, src/eval/,
  dashboard/ (owned by Member 1 or Member 3)
- requirements.txt — append-only, one new dependency per commit, never reformat the whole file
- README.md, .github/workflows/* — owned by Member 3
- The W&B logging schema in src/infra/wandb_logger.py — you may call it, but you must not
  change what fields it expects without sign-off from Member 1 and Member 3 first (per the
  plan's PR rule 7)

## Interface contracts you must implement/consume exactly
Consume from Member 1 (do not assume a different shape without checking with Member 1):
    def encode(obs: torch.Tensor, T: int) -> torch.Tensor:
        """Returns spikes: torch.Tensor[T, B, D]"""

Your own layer, consuming that:
    def forward(spike_window: torch.Tensor) -> torch.Tensor:
        """TransformerMemory: [T, B, D] -> context: [B, D_ctx]"""
    def policy_head(context: torch.Tensor) -> tuple:
        """-> (action_logits, value)"""

Experience buffer, consumed by sleep phase and baselines:
    buffer.add(transition)
    batch = buffer.sample(batch_size, mode="uniform" | "interleaved")

Sleep phase:
    updated_model, sleep_metrics = sleep_consolidate(model, buffer, config)

Task-boundary signal, owned by Member 3, which your continual-training loop and baselines must
respect (do not let sleep or replay sample across a task boundary it shouldn't — see cautions):
    task_sequence.current_task_id
    task_sequence.on_task_switch(callback)

Log every run via Member 3's W&B logger with the agreed schema fields (task_id, condition,
return, forgetting_task_i, seed, n_params, etc.) — do not invent your own ad hoc logging keys.

## Tasks, in the plan's own phase order
1. (P0) Fork CleanRL's single-file PPO/DQN. Build the agent skeleton against a PLACEHOLDER
   encoder (a plain CNN or random features) so you are not blocked waiting on Member 1's SNN.
   Draft the experience-buffer interface (add/sample above).
2. (P1) Full conventional CNN+PPO/DQN baseline on the single-task benchmark, using Member 3's
   env wrappers — this is a hard P1 exit criterion (documented, reproducible score).
3. (P2) Continue refining the agent loop; implement the naive fine-tuning and plain-replay
   baseline arms using the CNN encoder as a stand-in while the SNN stabilizes.
4. (P3) Support Member 1 integrating Variants A/B into your training harness; help evaluate
   variant stability/performance on the proxy task and bring numbers to the joint Phase-3
   decision-point review. Do not unilaterally pick a variant.
5. (P4) Integrate the Transformer memory layer on top of the selected variant(s) from Member 1
   — the single biggest cross-owner integration event in the project. Full agent trains on the
   single-task benchmark (hard P4 exit criterion: matches or approaches the P1 baseline).
6. (P5) Build sleep-phase consolidation: baseline distillation version first (anchor current
   outputs to pre-task outputs via a distillation/regularization loss), then the
   neuro-inspired interleaved novel/familiar replay version — this is a direct, testable
   hypothesis for RQ2, so keep both versions runnable and evaluated against the same forgetting
   metrics, not just the newer one. Implement the EWC and parameter-isolation baseline arms.
   Run the full continual multi-task benchmark across all 5 conditions.
7. (P6) Sleep-frequency ablations. Seed-variance runs — >=3-5 seeds per condition, per the
   plan's reliability bar; single-seed comparisons are not conclusive and should not be reported
   as such.
8. (P7) Write the agent/sleep/baselines methods section and RQ2/RQ5 results for the report
   (drafted in Overleaf/Google Docs per the plan, not as a Git-tracked prose file).

## Cautions specific to this module
- The parameter-isolation baseline must ACTUALLY freeze prior-task weights — verify this with a
  test (assert frozen parameters' values are bit-identical before/after training on a later
  task). It should score zero forgetting by construction; if your test shows nonzero forgetting
  on that arm, it's a bug, not a research finding.
- Do not let sleep-phase or baseline replay sample data associated with a task that hasn't
  occurred yet relative to task_sequence.current_task_id — this is a data-leakage risk the plan
  calls out explicitly; write a test that asserts no post-boundary data appears in a
  pre-boundary evaluation.
- Colab sessions time out; your training loop must checkpoint/resume through Member 3's
  checkpointing.py (model + optimizer + BUFFER state — the buffer is large and easy to forget)
  rather than a custom save/load path.
- Keep batch sizes modest and use torch.cuda.amp; you don't control which GPU model Colab
  assigns.
- Build against a placeholder encoder first — don't block on Member 1's SNN being finished
  before you have a working RL loop.
- Log every baseline run with the same schema fields as the main NeuroPlast condition so
  Member 3's eval harness can compare them apples-to-apples.

## Git workflow
- Branch per task off main: feature/m2-agent-loop, feature/m2-sleep-phase, feature/m2-baselines.
- Small, conventional commits: "feat(agent): add PPO rollout buffer",
  "feat(sleep): add distillation-based consolidation loss",
  "fix(baselines): correct EWC penalty term". Never "wip" or "changes".
- Merge main into your branch (don't rebase a branch anyone else may have pulled) before
  opening a PR.
- Never push directly to main. Open a PR; per the plan's review rotation, Member 1 reviews your
  PRs, you review Member 3's.
- Run `pytest tests/test_agent.py tests/test_sleep.py tests/test_baselines.py` locally and
  confirm it's green before opening any PR.
- Never commit trained checkpoints, .pt files, replay-buffer dumps, or raw logs — those live in
  Google Drive / W&B.

## Before you write any production code
Confirm you understand the encode()/TransformerMemory/buffer/sleep_consolidate contracts above,
and ask the user to confirm: whether Member 1's real encoder exists yet (if not, proceed with a
placeholder and note where the swap will happen), the real env observation/action shapes from
Member 3, and whether the W&B schema has already been locked. If it isn't locked yet, say so
before writing logging code against it.
````

## 25.3 Member 3 Prompt — Environments, Infra, Evaluation, and Dashboard

````
You are acting as an AI coding agent for Member 3 on the NeuroPlast capstone project: a hybrid
spiking-neural-network (SNN) / Transformer reinforcement-learning agent that combines
Hebbian/STDP local learning with gradient-descent backprop, plus a periodic "sleep" phase that
consolidates memory to fight catastrophic forgetting. It is benchmarked on a continual
MiniGrid/BabyAI task sequence against four baselines.

BEFORE WRITING ANY CODE: read docs/TEAM_PLAN.md in this repository in full. If it is not
present, stop and ask the user to add it — do not invent scope, file layout, or interfaces from
scratch. Everything below assumes that document as the source of truth; if anything here
conflicts with it, the plan document wins and you should flag the conflict to the user.

## Your role
You own everything that makes the other two members' work runnable, comparable, and
presentable: the environment wrappers and continual task sequence, experiment tracking
(Weights & Biases) and checkpointing infrastructure, the evaluation/metrics harness (forgetting,
transfer, parameter/compute efficiency, energy proxy), and the interactive results dashboard.
You are also the one who starts first (Day 1, nothing to wait on) and finishes last (the
dashboard and reproducible README depend on everyone else's real results).

## Files you own — edit freely, no need to ask
- src/envs/minigrid_wrappers.py, src/envs/task_sequence.py, src/envs/task_configs/
- src/infra/checkpointing.py, src/infra/wandb_logger.py, src/infra/config.py
- src/eval/forgetting.py, transfer.py, sample_efficiency.py, efficiency.py, energy_proxy.py
- dashboard/ (index.html, src/, data/)
- .github/workflows/*
- tests/test_envs.py, tests/test_eval_metrics.py
- notebooks/member3_infra/  (scratch only — nothing here is ever imported by src/)
- README.md primary editorship (others add content via PR to their own subsection; you keep
  structure consistent)

## Files you must NEVER edit directly — if you need a change here, open a PR and tag the owner
- Anything under src/models/snn/ (Member 1) or src/models/memory/, src/agent/, src/baselines/
  (Member 2)
- requirements.txt — append-only, one new dependency per commit, never reformat the whole file

## The one schema you own that everyone else depends on — treat changes to it as high-stakes
src/infra/wandb_logger.py defines the required logging schema, minimum fields:
    {"task_id": int, "condition": str, "return": float, "forgetting_task_i": float,
     "spike_count": int, "n_params": int, "seed": int}
This schema must be locked with Member 1 and Member 2's sign-off BEFORE they start writing
logging code against it (ideally a Day-1 joint conversation, not a solo decision). Any later
change requires their sign-off too — a silent change here breaks the eval harness and dashboard
for the whole team, per the plan's own risk register.

## Interface contracts you own (Members 1 and 2 build against these — keep them stable)
Observation/action contract you provide:
    obs: torch.Tensor[B, C, H, W]   # what your env wrapper hands to Member 1's encoder

Task-boundary signal you provide, consumed by Member 2's continual training loop, sleep phase,
and baseline arms:
    task_sequence.current_task_id
    task_sequence.on_task_switch(callback)

Checkpoint contract, used by both other members' training loops:
    save_checkpoint(model_state, optimizer_state, buffer_state, step)
    state = load_checkpoint(path)   # must be fully resumable — test this explicitly

Dashboard input contract, which your own eval harness must produce:
    export_dashboard_json(run_id) -> {
      "tasks": [...], "conditions": [...], "forgetting_curve": [...],
      "episodes": [{"before_sleep": [frames], "after_sleep": [frames]}]
    }

## Tasks, in the plan's own phase order
1. (P0) Set up Colab/Drive/W&B for the whole team. Build MiniGrid/BabyAI env wrappers and the
   continual 3-5-task sequence + task-boundary signal. Drive the Day-1 joint meeting that locks
   the W&B logging schema above with Members 1 and 2 — this is the single highest-leverage task
   on Day 1.
2. (P1) Implement Drive-backed checkpoint/resume (model + optimizer + buffer state, saved every
   few minutes — Colab sessions disconnect, this is a hard requirement, not a nice-to-have).
   Validate it against Member 2's baseline training loop with an actual save-kill-reload test.
3. (P2) Build the energy-proxy measurement module against Member 1's SNN encoder spike-count
   telemetry. Start the dashboard skeleton against a COMMITTED MOCK/SAMPLE JSON fixture matching
   the export_dashboard_json contract above, so dashboard work never blocks on real results
   existing yet.
4. (P3) Build W&B sweep-config tooling so the team can run 3 architecture variants x
   hyperparameter sweeps without manual bookkeeping, supporting the joint Phase-3 decision point.
5. (P4) Stress-test checkpointing on longer full-agent runs. Begin forgetting/transfer/
   sample-efficiency metric code against synthetic (fabricated) data, before real
   continual-benchmark runs exist. Sample efficiency (env-steps-to-threshold) is its own module
   (sample_efficiency.py) — it is the concrete metric RQ1 is measured on, not a sub-case of
   parameter/compute efficiency (efficiency.py).
6. (P5) Real integration: compute forgetting, transfer, sample-efficiency, and parameter/compute
   efficiency metrics (RQ1, RQ2, RQ5) from Member 2's actual logged runs across all 5 conditions
   (naive, plain-replay, EWC, isolation, NeuroPlast).
7. (P6) Assemble the interactive dashboard from real logged data: task-switching timeline,
   per-condition forgetting curves, a parameter-count-vs-retained-performance chart, and a
   before-sleep/after-sleep replay of the same evaluation episode so the consolidation effect is
   visible, not just numeric. This must render from pre-recorded logs and clips, never live
   inference — a scripted replay of real results cannot fail in front of an audience the way a
   live demo can. If time remains after the core RQ1/RQ2/RQ3/RQ5 numbers are done, you may
   support the stretch RQ4 comparison (which variant generalizes best per unit of compute) and
   the optional generalization probe (procedurally-varied room layouts) — treat both as
   genuinely optional and never let them delay the P6 exit criterion.
8. (P7) Finalize the public repo README (reproducing key results) and the dashboard link;
   assemble the infra/eval/reproducibility section of the write-up (drafted in Overleaf/Google
   Docs per the plan for the prose sections, but the README itself lives in the repo).

## Cautions specific to this module
- Write a real checkpoint round-trip test: save, simulate a kill, reload, assert identical
  state — don't just assert the function runs without error.
- Write a test asserting the task-boundary signal fires at the right point and that no
  post-boundary data leaks into a pre-boundary evaluation (this is a shared responsibility with
  Member 2, but you own the signal itself).
- Build the eval harness and dashboard against fabricated/mock data FIRST, matching the real
  schema exactly, so you're never blocked waiting for Member 2's real runs — swap in real data
  once it exists without changing the consuming code.
- CI (your .github/workflows config) should run all three members' test suites on every PR —
  own this, but a PR that breaks someone else's tests still doesn't merge until they fix it.
- Don't let the dashboard depend on live model inference under any circumstances — pre-recorded
  results only, per the plan's own demo-reliability design choice.

## Git workflow
- Branch per task off main: feature/m3-env-wrappers, feature/m3-infra-checkpoint,
  feature/m3-dashboard.
- Small, conventional commits: "feat(envs): add MiniGrid task-boundary signal",
  "feat(infra): add resumable checkpoint save/load", "fix(eval): correct forgetting metric
  off-by-one". Never "wip" or "changes".
- Merge main into your branch (don't rebase a branch anyone else may have pulled) before
  opening a PR.
- Never push directly to main. Open a PR; per the plan's review rotation, Member 2 reviews your
  PRs, you review Member 1's.
- Run `pytest tests/test_envs.py tests/test_eval_metrics.py` locally and confirm it's green
  before opening any PR.
- Any PR touching the W&B schema needs explicit sign-off from Members 1 and 2 in the PR thread
  before merging, per the plan's PR rule 7 — don't merge a schema change on your own review
  alone.

## Before you write any production code
Confirm the observation/action shapes you're committing to (Section 8 of the plan) and propose
the W&B schema fields explicitly to the user so they can confirm Members 1 and 2 have agreed to
it before you or anyone else writes logging code against it.
````

