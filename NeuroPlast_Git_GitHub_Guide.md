# NeuroPlast — Git & GitHub Practical Guide

*A standalone, hands-on companion to `docs/TEAM_PLAN.md`. That document says **who** owns what; this one is entirely about the git/GitHub mechanics — exact commands, exact settings, and exactly what to do when something is about to conflict or already has. Every member should read their own playbook (Sections 15–17) at minimum; Sections 1–14 are shared reference.*

---

# 0. How to use this document

- Read Sections 1–4 once, together, on Day 1 — they're one-time repo setup.
- Read Sections 5–14 once, individually, before your first PR — they're the rules everyone follows regardless of role.
- Jump straight to your own playbook (Section 15, 16, or 17) whenever you want a quick "what do I actually type" refresher.
- Keep Section 19 (Quick Reference Card) open in a tab while you work.

This guide assumes the file ownership, branch names, and interfaces already defined in `docs/TEAM_PLAN.md` Sections 2, 7, 8, 11, and 16. If anything here seems to contradict that document, the team plan wins — flag it and fix this file.

---

# 1. One-Time Setup (all three members, Day 1)

**1.1 Git identity — run once per machine:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your_github_email@example.com"
```

**1.2 Safe defaults that prevent accidental history-rewriting:**
```bash
git config --global pull.rebase false      # `git pull` = fetch + merge, not fetch + rebase, by default
git config --global push.default simple    # only push the current branch, never all branches
git config --global init.defaultBranch main
```
The team's policy is "merge main into your branch, never rebase a shared branch" (Section 10 below) — `pull.rebase false` makes that the default behavior instead of something you have to remember every time.

**1.3 Clone the repo:**
```bash
git clone https://github.com/<org>/neuroplast.git
cd neuroplast
```

**1.4 Create your virtual environment from the shared `requirements.txt`:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Do this again — a full `pip install -r requirements.txt` — every single time you pull a change to `requirements.txt`, not just on Day 1. See Section 13.

**1.5 Copy the environment template:**
```bash
cp .env.example .env
# then fill in your own W&B API key and Drive-mount path in .env — never commit .env
```

---

# 2. Repository Configuration (whoever creates the repo — recommended: Member 3, Day 1)

**2.1 Branch protection on `main`** — GitHub → Settings → Branches → Add rule for `main`:
- ✅ Require a pull request before merging
- ✅ Require at least 1 approval
- ✅ Require review from Code Owners (see 2.3 below)
- ✅ Require status checks to pass before merging (select the CI job from 2.2)
- ✅ Do not allow bypassing the above settings (applies the rule to admins too)
- ✅ Do not allow force pushes
- ✅ Do not allow deletions

This turns "never push directly to `main`" and "PRs require review" from a rule people remember into a rule GitHub enforces.

**2.2 Minimal CI — `.github/workflows/ci.yml` (owned by Member 3):**
```yaml
name: NeuroPlast CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run full test suite
        run: pytest tests/ -v
```
This runs everyone's tests on every PR — a PR that breaks another member's tests fails CI and can't merge, per branch protection above.

**2.3 `CODEOWNERS` — `.github/CODEOWNERS` (owned by Member 3, mirrors the Ownership Matrix in `TEAM_PLAN.md` Section 7):**
```text
# NeuroPlast CODEOWNERS
# GitHub auto-requests review from the matching owner on any PR touching these paths.
# Replace the handles below with real GitHub usernames on Day 1.

# Member 1 — SNN + hybrid learning rule
/src/models/snn/                        @member1-handle
/tests/test_snn.py                      @member1-handle
/tests/test_learning_rule.py            @member1-handle

# Joint file — both must review
/src/models/snn/variants/variant_c.py   @member1-handle @member2-handle

# Member 2 — agent, memory, sleep, baselines
/src/models/memory/                     @member2-handle
/src/models/policy_heads.py             @member2-handle
/src/agent/                             @member2-handle
/src/baselines/                         @member2-handle
/tests/test_agent.py                    @member2-handle
/tests/test_sleep.py                    @member2-handle
/tests/test_baselines.py                @member2-handle

# Member 3 — envs, infra, eval, dashboard, CI, README
/src/envs/                              @member3-handle
/src/infra/                             @member3-handle
/src/eval/                              @member3-handle
/dashboard/                             @member3-handle
/.github/                               @member3-handle
/tests/test_envs.py                     @member3-handle
/tests/test_eval_metrics.py             @member3-handle
/README.md                              @member3-handle

# High-stakes shared files — require ALL THREE to review
/requirements.txt                       @member1-handle @member2-handle @member3-handle
/src/infra/wandb_logger.py              @member1-handle @member2-handle @member3-handle
```
With "Require review from Code Owners" turned on in branch protection, GitHub will now physically block a merge to `wandb_logger.py` or `requirements.txt` until all three members have approved — this is the single most reliable way to enforce the plan's PR rule 7 ("any PR that changes the W&B schema needs sign-off from every consumer").

---

# 3. `.gitignore` — What Never Gets Committed, and Why

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
.venv/
venv/
env/

# Jupyter — notebook outputs are the #1 cause of pointless "conflicts" in ML repos
.ipynb_checkpoints/

# Secrets / machine-specific config
.env
*.env.local

# ML artifacts — Drive and W&B are the source of truth, never Git (TEAM_PLAN.md Section 17)
data/checkpoints/
data/logged_runs/
*.pt
*.pth
*.ckpt
wandb/
outputs/
runs/

# OS cruft
.DS_Store
Thumbs.db

# Dashboard build output (if a bundler is ever introduced)
dashboard/node_modules/
dashboard/dist/

# Exception: the committed sample fixture the dashboard builds against before real data exists
!dashboard/data/sample_fixture.json
```
Every line above exists because that artifact is either regenerable, machine-specific, or actively harmful to commit (huge binary, secret, or noisy diff). If you're about to `git add` something matching one of these patterns, stop — it almost certainly shouldn't be tracked.

---

# 4. Notebook Hygiene — `nbstripout`

Notebooks are the hardest thing to text-merge in any ML repo, because the `.ipynb` file format embeds cell *outputs* (including base64-encoded images) as JSON. Two people running the same notebook cell produces a "conflict" even if the code is identical, because the output blob differs. Since notebooks in this project are scratch-only and per-member (`notebooks/member1_snn/`, etc.), you shouldn't ever conflict with a teammate — but you can still make your own history unreadable if outputs are committed. Install `nbstripout` once, and it becomes a non-issue:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```
This registers a git filter that strips cell outputs automatically at commit time, so your notebook history stays small and diffable, and you never accidentally commit a screenshot-sized output blob.

---

# 5. Branch Naming & Lifecycle

| Member | Branch prefix examples |
|---|---|
| Member 1 | `feature/m1-snn-encoder`, `feature/m1-stdp-rule`, `feature/m1-variant-a`, `feature/m1-variant-b` |
| Member 2 | `feature/m2-agent-loop`, `feature/m2-sleep-phase`, `feature/m2-baselines`, `feature/m2-buffer` |
| Member 3 | `feature/m3-env-wrappers`, `feature/m3-infra-checkpoint`, `feature/m3-eval-metrics`, `feature/m3-dashboard` |
| Joint | `feature/m1-m2-variant-c` |

Rules:
- One branch = one task, not one person. A branch that's been open for more than ~5 working days without a PR is a sign the task is too big — split it.
- Branch off the latest `main`, never off another feature branch (that chains conflicts).
- Delete the branch immediately after merge (GitHub can do this automatically on merge — turn that setting on in repo settings, "Automatically delete head branches").

---

# 6. Daily Workflow (everyone, every day)

**Start of session — always sync first, even mid-task:**
```bash
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
```
Do this at the *start* of every session, not just before opening a PR. A branch that's 4 days stale is where painful conflicts come from — a branch that's synced daily almost never has one, because the diffs staying small is what makes merges trivial.

**While working — commit small and often:**
```bash
git add path/to/specific/file.py
git commit -m "feat(scope): short description"
```
Commit whenever you reach a working, logical checkpoint — not just at the end of the day. Small commits are easier to review, easier to revert individually, and never lose more than a few minutes of work if something goes wrong.

**End of session — always push, even if the task isn't finished:**
```bash
git push origin your-feature-branch
```
Uncommitted or unpushed work sitting on your laptop for days is the single easiest way to create a large, painful merge later — and it's not backed up anywhere if your machine has a problem.

**Before opening a PR — sync one more time and test:**
```bash
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
# resolve any conflicts now (Section 9), not during review
pytest tests/test_<your_module>.py
git push origin your-feature-branch
# then open the PR on GitHub
```

---

# 7. Command Cheat Sheet

```bash
git status                          # what's staged, unstaged, untracked
git diff                            # unstaged changes
git diff --staged                   # staged changes
git log --oneline --graph --all     # visual history across all branches
git branch -vv                      # local branches + what they track
git stash                           # shelve uncommitted changes temporarily
git stash pop                       # bring them back
git fetch origin                    # update your knowledge of remote branches without merging
git checkout -b new-branch          # create + switch in one step
```

---

# 8. Where Conflicts Actually Happen in NeuroPlast, and How to Prevent Each One

| File / area | Why it's risky | Prevention |
|---|---|---|
| `requirements.txt` | Every member adds dependencies here; multi-line diffs are the classic conflict generator | Add **one dependency per commit**, always appended at the bottom, never reformatted or alphabetized as a side effect |
| `src/infra/wandb_logger.py` (the schema) | Everyone's code depends on the exact field names | Locked jointly on Day 1 (`TEAM_PLAN.md` Interface 7); any later change needs sign-off from all three via CODEOWNERS before it can merge |
| `src/models/snn/variants/variant_c.py` | The only file two people edit directly | Coordinate in chat *before* opening your editor, not after you're mid-edit; small, frequent commits; pull before every session (Section 6) |
| `README.md` | Everyone eventually wants to add a line | Member 3 keeps overall structure; others add content to their own named subsection only, via PR, never a restructure |
| `.github/workflows/*`, `.github/CODEOWNERS` | Misconfiguration silently breaks CI for everyone | Member 3 only, PR-only, never edited casually |
| `src/agent/buffer.py` | Consumed by both `sleep.py` and all four `baselines/*.py` files | If you change its signature, update *every* caller in the **same commit** — a half-migrated interface breaks the branch even though Git sees no textual conflict (see Section 13) |
| Notebooks | JSON diff on every cell run | Per-member folders + `nbstripout` (Sections 4–5); nothing in `notebooks/` is ever imported by `src/`, so there's structurally nothing to conflict with |
| Anyone's isolated module dir (`src/models/snn/`, `src/agent/`, `src/envs/`, etc.) | Should be near-zero risk by design | Stays that way only if nobody edits another member's directory directly — if you need a change there, open a PR against it, don't just fix it yourself |

---

# 9. Worked Example: Resolving a Real Merge Conflict, Step by Step

Say Member 1 is on `feature/m1-snn-encoder` and added `snntorch` to `requirements.txt`, while `main` has since gained `wandb` (added by Member 3) at nearly the same line.

**Step 1 — try to sync:**
```bash
git checkout feature/m1-snn-encoder
git merge main
```
```text
Auto-merging requirements.txt
CONFLICT (content): Merge conflict in requirements.txt
Automatic merge failed; fix conflicts and then commit the result.
```

**Step 2 — see what's conflicted:**
```bash
git status
```
```text
both modified:   requirements.txt
```

**Step 3 — open the file. Git has inserted conflict markers:**
```text
torch>=2.1
gymnasium
minigrid
<<<<<<< HEAD
snntorch
=======
wandb
>>>>>>> main
pytest
```
`<<<<<<< HEAD` down to `=======` is *your* branch's version; `=======` down to `>>>>>>> main` is what came from `main`. In this case the right resolution is obvious — keep both lines:

```text
torch>=2.1
gymnasium
minigrid
snntorch
wandb
pytest
```
Delete all three marker lines (`<<<<<<<`, `=======`, `>>>>>>>`) — leaving one in by accident is the most common conflict-resolution mistake and it will break `pip install`.

**Step 4 — mark it resolved and finish the merge:**
```bash
git add requirements.txt
git commit -m "merge: resolve requirements.txt conflict, keep snntorch and wandb"
pip install -r requirements.txt        # re-install — the dependency set just changed
pytest tests/test_snn.py               # re-run your tests before pushing
git push origin feature/m1-snn-encoder
```

The same pattern (see markers → decide what the *correct combined* result is, not just "pick one side" → delete markers → re-test → commit) applies to any conflicted file. The only NeuroPlast-specific file where "pick one side" is *wrong by design* is `variant_c.py` — since Members 1 and 2 are supposed to be actively coordinating on it, a conflict there is a signal to talk to each other before resolving, not just a text problem to merge past.

---

# 10. Merge vs. Rebase — the Team Policy, in Practice

**Team rule, restated with commands: use `git merge main`, never `git rebase main`, on any branch a teammate might have pulled.**

```bash
# CORRECT — bring your branch up to date safely
git checkout your-feature-branch
git merge main

# WRONG on a shared/pushed branch — rewrites commit hashes, breaks anyone who pulled it
git rebase main
```

`git rebase` is fine in exactly one situation: cleaning up your *own local, not-yet-pushed* commits before your first push, e.g., squashing three "wip" commits into one clean commit:
```bash
git rebase -i HEAD~3       # only if nobody else has this branch checked out yet
```
Once you've pushed a branch and a teammate might have pulled it (this matters most for `variant_c.py`), never rebase it again — merge only, from then on.

---

# 11. Force-Push Safety & When (Almost Never) It's OK

- Force-pushing `main` is **disabled at the repo level** (Section 2.1) — it should be structurally impossible, not just against the rules.
- Force-pushing a feature branch is occasionally legitimate right after a local interactive rebase (Section 10) — but only if you are the *sole* person who has ever pushed that branch. If in doubt, ask in the team channel first.
- If you must, use `--force-with-lease`, never bare `--force`:
```bash
git push --force-with-lease origin your-feature-branch
```
`--force-with-lease` refuses to overwrite the remote branch if someone else has pushed to it since you last fetched — bare `--force` will happily destroy their work. There is essentially never a reason to use bare `--force` on this project.

---

# 12. Recovering From Mistakes

**"I want to see my recent history, including commits I might have 'lost':**
```bash
git reflog
```
This shows every place `HEAD` has pointed recently, including commits that a `reset --hard` or a bad rebase seems to have deleted — they're still there until garbage collection, and `reflog` finds the hash.

**"I ran `git reset --hard` and want the old state back":**
```bash
git reflog                      # find the commit hash from before the reset
git reset --hard <that-hash>    # restores it — only ever do this on your own local branch
```

**"Something bad got merged into `main` and I need to undo it" (main is protected — you can't reset it):**
```bash
git revert <bad-commit-hash>
git push origin main            # via a PR, per branch protection — a revert is still a normal PR
```
`git revert` creates a *new* commit that undoes the change, preserving history — this is always the right tool for undoing something on a shared branch. `git reset` rewrites history and must never be used on `main` or any branch others have pulled.

---

# 13. Beyond Git: Semantic Conflicts and Dependency Drift

Git only catches *textual* conflicts — two people editing the same lines. It cannot catch a **semantic conflict**: your branch and `main` both merge cleanly, tests still technically run, but the combined behavior is now wrong because an interface changed underneath a caller Git never flagged. Two concrete NeuroPlast risks:

- **Interface drift.** If Member 1 changes `encode()`'s return shape, Git will merge that change into Member 2's branch with zero conflict markers — the files aren't even the same file. The break only shows up at runtime, in Member 2's tests, potentially days later. **Rule: if you change any Section-8 interface, `grep` the whole repo for its call sites before opening the PR, not just the file you edited**, and say so explicitly in the PR description so the reviewer checks it too.
- **Dependency drift.** Whenever `requirements.txt` changes (yours or a merged-in change from someone else), re-run `pip install -r requirements.txt` immediately — don't wait until something breaks mysteriously. A version bump that only lives in one person's `pip freeze` is a "works on my machine" bug waiting to happen, and it will not show up as a merge conflict at all.

Both of these are exactly why CI (Section 2.2) runs the *whole* test suite on every PR, not just the touched module's tests — it's the only automated backstop for semantic conflicts that Git itself can't see.

---

# 14. Pull Request Checklist (run through this before every PR)

1. `git merge main` into your branch — is it done, and clean?
2. Did you `pip install -r requirements.txt` again if `requirements.txt` changed?
3. Do your own module's tests pass locally?
4. If you touched a Section-8 interface, did you grep for all call sites across the repo?
5. Is the PR scoped to one task — not several unrelated changes bundled together?
6. Does the PR description name which interface(s), if any, it touches?
7. Did you double-check no `.pt`, `.pth`, checkpoint, or `.env` file snuck into `git status`?
8. Opened against `main`, not against another feature branch?

---

# 15. Member 1 Playbook — SNN Perception + Hybrid Learning Rule

**Your branches:** `feature/m1-snn-encoder`, `feature/m1-stdp-rule`, `feature/m1-variant-a`, `feature/m1-variant-b`, and jointly `feature/m1-m2-variant-c`.

**Your daily routine:**
```bash
git checkout main && git pull origin main
git checkout feature/m1-<task>
git merge main
# ... work, small commits ...
git add src/models/snn/<file>.py tests/test_snn.py
git commit -m "feat(snn): <what you did>"
git push origin feature/m1-<task>
```

**Conflict risk for you: structurally near-zero in normal operation.** `src/models/snn/` (outside `variant_c.py`) is yours alone — as long as nobody else edits it and you don't edit `src/agent/`, `src/envs/`, or `src/infra/`, there is nothing to conflict with. Your actual risks are:
- **`variant_c.py`.** This is the one file you share with Member 2. Message them before you start a session on it, not after; pull immediately before editing; commit and push in small chunks so neither of you is ever more than an hour of work away from the other's latest state.
- **Stale branches.** SNN/STDP work can involve long focused stretches without an obvious "commit point." Force yourself to `git merge main` at least every 2 days regardless of how far along the task is — a 2-week-old branch merged all at once is where real conflicts come from, even in an otherwise isolated module.
- **Interface drift you cause.** If you ever need to change `encode()`'s output shape or dtype, that's a Section-8 interface change — grep for every place `encode(` is called (it will be inside Member 2's `transformer_memory.py`) before opening the PR, and say so in the PR description so Member 2 reviews it with that in mind.

**Reviewer rotation:** Member 3 reviews your PRs; you review Member 2's.

---

# 16. Member 2 Playbook — Agent, Sleep, and Baselines

**Your branches:** `feature/m2-agent-loop`, `feature/m2-buffer`, `feature/m2-sleep-phase`, `feature/m2-baselines`, and jointly `feature/m1-m2-variant-c`.

**Your daily routine:**
```bash
git checkout main && git pull origin main
git checkout feature/m2-<task>
git merge main
# ... work, small commits ...
git add src/agent/<file>.py tests/test_agent.py
git commit -m "feat(agent): <what you did>"
git push origin feature/m2-<task>
```

**Conflict risk for you: low from other people, moderate from your own scope.** You own the largest file surface in the project (`src/models/memory/`, `src/agent/`, `src/baselines/`), so the main risk isn't a teammate editing your files — it's your own PRs becoming too large to review or test cleanly. Specific things to watch:
- **Don't bundle unrelated work into one PR.** It's tempting to open one big PR covering the agent loop *and* all four baselines *and* sleep, since they're all "yours." Don't — split them into separate PRs (`AGENT-01`, `BASE-01`..`BASE-05`, `SLEEP-01`/`SLEEP-02` from the task board are already sized right). A 2,000-line PR is where semantic bugs hide and where a reviewer rubber-stamps instead of actually reviewing.
- **`buffer.py` is a shared dependency inside your own scope.** Both `sleep.py` and all four `baselines/*.py` files call into it. If you change `buffer.add()` or `buffer.sample()`'s signature, update every caller in the *same* commit — a half-migrated buffer interface will pass Git's merge cleanly (no conflict markers) and then fail at runtime in a file you didn't touch in that commit. This is exactly the semantic-conflict risk from Section 13, and you're the person most likely to hit it since you own both sides of that interface.
- **`variant_c.py`** — same coordination protocol with Member 1 as described in Section 15.
- **Isolation baseline correctness isn't a git issue, but merging a broken one is.** Before opening `BASE-05`'s PR, actually run the frozen-weights assertion test (Section 18 of `TEAM_PLAN.md`) — a merge that passes CI but ships a baseline that silently isn't isolated is worse than a merge conflict, because nothing will ever flag it for you.

**Reviewer rotation:** Member 1 reviews your PRs; you review Member 3's.

---

# 17. Member 3 Playbook — Envs, Infra, Eval, Dashboard

**Your branches:** `feature/m3-env-wrappers`, `feature/m3-infra-checkpoint`, `feature/m3-eval-metrics`, `feature/m3-dashboard`.

**Your daily routine:**
```bash
git checkout main && git pull origin main
git checkout feature/m3-<task>
git merge main
# ... work, small commits ...
git add src/infra/<file>.py tests/test_envs.py
git commit -m "feat(infra): <what you did>"
git push origin feature/m3-<task>
```

**Conflict risk for you: you're the hub, so process PRs promptly.** Your own module directories (`src/envs/`, `src/infra/`, `src/eval/`, `dashboard/`) are as isolated as everyone else's — but you also own or co-own every high-traffic shared file: `README.md`, `.github/CODEOWNERS`, `.github/workflows/`, and (jointly) `requirements.txt` and `wandb_logger.py`. That makes you the most likely person to have *multiple open PRs waiting on you at once*. Specific things to watch:
- **Don't let PRs against `README.md` or `requirements.txt` sit for days.** A stale open PR against a shared file is the most likely multi-day conflict source in this project — if Member 1 and Member 2 both open README PRs the same week and yours takes a week to merge, the second one will conflict with the first. Review and merge these quickly, even if the content itself is simple.
- **When merging two same-day PRs that both touch `wandb_logger.py`,** re-pull `main` and re-run the full test suite between merges — don't merge both back-to-back on stale local state, even if GitHub says both are individually mergeable.
- **You set up and maintain `CODEOWNERS` and branch protection (Section 2)** — if a new shared file appears later that isn't already covered by a CODEOWNERS line, add it before it becomes a conflict source, not after.
- **Checkpoint round-trip correctness isn't a git issue either, but it gates everyone's merges.** Your `checkpointing.py` is a dependency of both Member 1's and Member 2's training loops — before merging any change to it, actually run the save-kill-reload test, since a subtly broken checkpoint format will silently corrupt every long training run downstream without ever showing up as a merge conflict.

**Reviewer rotation:** Member 2 reviews your PRs; you review Member 1's.

---

# 18. Weekly Team Git Sync (10 minutes, same time every week)

Not a status meeting — just three questions, so nobody is ever surprised by what's coming:
1. **What branches does each of us have open right now, and how old are they?** (Anything older than ~5 working days gets split or merged this week.)
2. **Is anyone about to change a Section-8 interface or the W&B schema?** Flag it *before* you start, not in the PR.
3. **Any file that isn't in `CODEOWNERS` yet but probably should be?**

---

# 19. Quick Reference Card

```bash
# Start of every session
git checkout main && git pull origin main && git checkout my-branch && git merge main

# Small, frequent commits
git add <specific-file>
git commit -m "feat(scope): what changed"
git push origin my-branch

# Before opening a PR
git merge main            # sync one more time
pytest tests/<my-tests>   # confirm green
git push origin my-branch # then open the PR on GitHub

# Resolving a conflict
git status                          # see what's conflicted
# open the file, resolve between <<<<<<< / ======= / >>>>>>>, delete the markers
git add <resolved-file>
git commit -m "merge: resolve conflict in <file>"
pytest tests/<my-tests>             # re-test after any conflict resolution
git push origin my-branch

# Never do these on main or a shared branch
git rebase main                     # only ever on your own unpushed commits
git push --force                    # use --force-with-lease if you truly must, and warn first
git reset --hard                    # only on your own local branch

# Recovery
git reflog                          # find a "lost" commit's hash
git revert <hash>                   # undo something already on main, safely
```
