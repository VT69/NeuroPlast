# NeuroPlast — Project Memory for Claude Code

## Project Brief
NeuroPlast is a solo capstone project: a controlled empirical study of how
biologically inspired mechanisms affect continual reinforcement learning,
using a hybrid SNN-based agent as the experimental substrate. It is not an
attempt to build a brain, and it does not assume any mechanism helps.

The mechanisms under test, each isolated through ablations:
- spike-based representations (SNN encoder vs CNN)
- local plasticity (STDP / Hebbian / three-factor rules)
- homeostatic stabilisation
- offline sleep-like consolidation (vs replay, EWC, naive fine-tuning)
- shared parameterisation (vs parameter isolation)
- Transformer working memory

The scientific value comes from finding out which mechanisms actually
contribute, not from making all of them win. Negative results are results:
keep them in the log, never delete them.

## Where things live
- `PROGRESS.md`: live state. Read it first, every session. It is the source
  of truth for results and overrides every other document.
- `docs/LITERATURE_CONTEXT.md`: literature synthesis, evidence hierarchy, and
  claims to avoid. Read it before designing a new experiment or writing any
  results text. Its final "Corrections and additions" section overrides its
  earlier sections.
- `docs/papers/`: source PDFs. Check numbers against these before quoting them.
- `docs/NeuroPlast_Project_Plan.docx`: the original proposal. Historical
  background only; the framing above supersedes it.

## Research questions (fixed targets; everything else is negotiable)
- **RQ1**: Does local plasticity (STDP/Hebbian) improve learning beyond
  backprop alone, beyond homeostasis alone, and beyond a random-update control?
- **RQ2**: Does sleep-style offline consolidation reduce forgetting beyond
  replay and EWC, at matched replay and compute budgets?
- **RQ3**: Does the SNN encoder reach a better accuracy-vs-operations frontier
  than a sparsified CNN? Energy claims are proxies unless measured on hardware.
- **RQ4**: Which variant is most robust per unit of compute?
- **RQ5**: Do shared weights retain more performance per unit of total memory
  (parameters plus replay buffer) than parameter isolation, and do they
  transfer? Isolation always wins on raw forgetting by construction.
- **Transformer**: Does explicit context help on tasks that actually require
  memory, beyond a simpler non-attention memory baseline?

## Rules for adding anything new
Before adding a mechanism, state in `PROGRESS.md`:
1. which literature motivates it,
2. which NeuroPlast result it addresses,
3. what confound it introduces,
4. what ablation would show it actually helped.

Add controls before mechanisms. Prefer the smallest experiment that answers
the question. If a mechanism failed, report the failure; don't tune it until
it looks good.

## Claims hygiene
- Report p-values and seed counts with every comparison. With bimodal
  outcomes, report solve rate.
- Keep protocol changes (learning rate, frame budgets) consistent across all
  arms of a comparison, and log them.
- Label every claim as literature, our result, hypothesis, or interpretation.

## Tech stack (preferences, not limits)
PyTorch, own LIF implementation (snnTorch as a test-time cross-check),
Gymnasium + MiniGrid/BabyAI, CleanRL-style PPO, CSV logging. Everything runs
on CPU (no GPU on the dev box or the user's machine); time goes to
environment stepping, so parallelise across cores, not devices. Swap anything
that gets in the way and log why.

## Working autonomously, including overnight
Sessions may run unattended for hours.
- Update `PROGRESS.md` continuously, not just at the end, with a "what to look
  at first" note that is always current.
- Don't stop and wait for input at decision points. Make the call, log the
  reasoning, keep going.
- Don't stall on one blocker; note it and move on.
- Pin threads (`OMP_NUM_THREADS=1`), look up PIDs before killing processes,
  and checkpoint periodically.
- You have full latitude on implementation and architecture. The research
  questions and the rules above are the fixed constraints.
