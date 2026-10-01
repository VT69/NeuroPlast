# NeuroPlast — Progress Log

Living log, updated as work happens. Newest status at the top; decisions log
and detailed notes below.

## RESULTS FREEZE — 2026-09-30 (01:30 UTC)
All experimental results for the capstone are frozen as of session 4. No further runs feed the final answers;
anything run later goes in a clearly separate "post-freeze" section and does not change the tables below.
Final answers: "Answers so far (FINAL)" and "Evidence hierarchy (FINAL)" further down. Every analysis script was
rerun on the frozen data at 01:25 UTC (`rq1_stats.py`, `memory_fair.py`, `rq5_stats.py`, `analyze.py all`,
`s4_stats.py`; `record_demo.py` for the demo). Not rerun: `rq3_encoders.py` (trains models) and
`benchmark_compute.py` (wall-clock timing; needs an idle machine), whose outputs are unchanged from session 3.
Results went in as they came out, including the nulls: session 4 turned two earlier positive hints into nulls
(sleep > replay; DFA+homeostasis solve rate) and found no usable memory benchmark.

## Session 8 (2026-10-01): final revision before review (no new runs; frozen results unchanged)
**What to look at first:** `paper/main.pdf` (preprint) and `paper/main_anonymous.pdf` (TMLR), Section 5.5's post-freeze
paragraph and Table 4's last row; deck slides 10 and 16-18; the dashboard's memory section and "What's next".

**Post-freeze result added (labelled as such everywhere).** The follow-up in `explore/` (open-ended session on the
`explore` branch, 2026-09-30/10-01; its log is `explore/EXPLORE.md`) is now on `main`. Only the folder at the branch tip
was copied, not the branch history, so the `.pt` checkpoints in that history stay off `main`. C4 was prospectively
specified before its runs: on CNN fetch5 with seeds 201-210 in every arm, quarter-width isolation (5 x 44,928
parameters, 0.90 MB) reached 0.972 ± 0.002. That beat a shared trunk with LwF-int8 (0.944, 0.91 MB; +0.028
[0.015, 0.041], Holm p = 0.002) and with replay (0.920, 1.83 MB; +0.052 [0.015, 0.089], Holm p = 0.011), and was
−0.004 [−0.005, −0.002] against full-width isolation (3.20 MB). Limits stated wherever the result appears: the shared
trunk was not shrunk the same way, and the SNN case is untested (the only narrow-SNN runs were a 2-seed pilot at 150k
frames/task). The macros (`post*`) come from `paper/compute_numbers.py::postfreeze()`, which reads
`explore/runs/confirm/` and asserts that all four arms share seeds 201-210.

**Wording and framing (paper, deck, dashboard):**
* new title
* "pre-registered" became "prospectively specified in a version-controlled experiment log"
* headline: "no statistically reliable improvement beyond the matched control; given the seed counts, these results
  rule out large effects rather than show equivalence". One deliberate qualifier was added: "and only where the
  intervals are narrow". The STDP AUC CI (−0.26 to +0.33) and the SNN small-buffer CI (±0.17) do not rule out large
  effects.
* precise statements replace "STDP adds nothing" and "sleep does not beat replay"
* six tested mechanisms plus an inconclusive Transformer
* DFA is "backprop-free encoder credit assignment (heads use exact gradients)"
* "shared trunk with per-task heads", with the parameters that scale with task count: trunk 142,832; per task 17,160
  (head 17,032 + embedding 128); new macros `paramsCnnTrunk`, `paramsPerTaskShared`, `paramsHead`, `paramsTaskEmb`
* a baselines subsection: regularisation / replay / isolation, plus LwF from the follow-up
* "estimated arithmetic energy proxy"; the 80 vs 16 ms timing is defined as measured cost in our implementation (CPU
  process time, 1 thread, one forward + backward + Adam step on 256 DoorKey-6x6 observations, mean of 5 after a
  warm-up; `scripts/benchmark_compute.py`)
* every result is labelled [confirmatory] / [exploratory] / [descriptive], and Table 4 has a Label column
* a seed-set statement: all arms share seeds, except the unmatched SNN+Transformer sleep arm (1-6) vs replay (1-8) in
  Table 3 / Fig. 3
* the deck conclusion and the dashboard scorecard now use survived / conditional / did not survive / unresolved

**Future work.** The preprint has a section "Future Work and Proposed Next Phase" with the scaling plan (JAX +
XLand-MiniGrid on cloud GPUs), validation first, the questions, the smaller follow-ups, the methodology and the timeline.
The anonymous build has one concise paragraph without a timeline. The deck has two "Next phase" slides before the
conclusion; the limitations slide no longer repeats future work. The dashboard has a "What's next" section without a
timeline. New bib entries `li2017learning`, `shin2017continual` and `nikulin2024xland` are marked TODO verify (no local
PDFs).

**Already done in session 7 and re-verified:** `--suite`/`--method` in the reproducibility commands, TrueType figure
fonts (pdftotext gives "−" and "λ"), Figure 3 CIs clipped to [0, 1], and `relevant_literature/` absent from every
branch's history (checked on a fresh clone).

## Session 7 (2026-09-30): final wording fixes; relevant_literature/ purged (no new experiments; results frozen)
Wording fixes, identical across the paper, deck and dashboard (commit 43b2a28, after the purge):
1. **Small-buffer result** (SNN, 200 states/task vs isolation). "Ties/matches" became "similar mean accuracy (0.838 vs
   0.833; 3 seeds, wide CI) in less than half the memory". This covers the abstract, intro, 5.5 (which now says the CI
   [−0.167, +0.177] rules out neither a large deficit nor a large advantage), the Fig. 5 caption, the Table 4
   shared-weights row, the conclusion, deck slides 7/9/15 (+ notes) and the dashboard (scorecard, key-result text, chart
   annotation). New macros: `smallSnnTwoHundredMean`, `snnIsoMean`, `smallSnnTwoHundredN`, `smallSnnTwoHundredMemPct`.
2. **Closing lesson.** "Every apparent win disappeared ..." became "effects shrank or disappeared once seeds, training
   budgets, replay budgets and memory accounting were made fair; the homeostasis-DFA speed-up survived". Applied in the
   conclusion and on deck slides 12 and 15 (+ notes).
3. **Memory benchmark.** "No benchmark we tried both required memory and was learnable within budget" (abstract, intro,
   deck slides 7 and 14 + notes). S7 was learnable but did not require memory, so the old wording was inaccurate.
4. **Reproducibility commands** now render as `--suite`, `--method`, `--config`, `--seed`, `--set` (checked with
   pdftotext).
5. **Figure fonts.** The figures use CMU Serif (TrueType with a Unicode map, `fonts-cmu`) instead of matplotlib's cmr10.
   The old fonts extracted as "¡0.02" and "¸=100"; pdftotext now gives "−0.02" (U+2212) and "λ=100", in the figures and
   in both paper PDFs.
6. **Figure 3.** CI whiskers are clipped to [0, 1], and the caption says so.
7. **Section 3.4.** Pre-registration is now "recorded in the project's version-controlled log before the runs"; the
   paper says "phases", not "sessions", in 3.4 and the limitations.
8. **relevant_literature/.** Untracked, added to `.gitignore` (kept locally), and purged from the whole history with
   `git filter-repo --path relevant_literature/ --invert-paths` (same identity callback). No other copies were found in
   history (the old session-2 zip has none).
   * backup #2 (`git bundle --all`, SHA-256 fab5c3b1…ba044) was sent to you in 17 parts before the rewrite
   * force-pushed with the lease pinned to cda8964; `main` = `session-3`
   * verified on a fresh clone; the pack shrank from 483 MiB to 455 MiB
   * `refs.bib` and `paper/README.md` now say the PDFs were checked against your local copies
   * hashes in this file were updated again

## Session 6 (2026-09-30): presentation materials and git history (no new experiments; results stay frozen)
Produced, in order (hashes after the rewrite):
* **Dashboard redesign** (395b411). `demo/index.html` got a softer visual system, a "What we built" architecture
  diagram (inline SVG), a plain-language scorecard with verdict badges (2 help conditionally, 1 trade-off, 3 no
  effect, 1 untested), and the subtitle "Mostly no, with two specific exceptions". The memory chart has its own
  sequence control (default SNN, 3 tasks), a y-axis from 0.7, and the key result annotated: sleep at 200 states/task
  ties isolation in 46% of the memory. The naive agent's card notes that its first task-1 clip happens to succeed (1
  of 3) and that the matrix gives the real average. Checked in a browser in light, dark and at 390 px: no console
  errors, no horizontal overflow.
* **GIFs re-rendered** from the same episodes (same rooms, actions and outcomes; `demo/assets/index.md` is
  unchanged apart from a playback note): 400 ms per frame and a 1.5 s hold on each episode's last frame. Decision
  logged here: the naive agent's time-outs are 256 steps, which is 100 s at 400 ms, so steps from the 24th on play at
  50 ms with a visible "fast-forward 8x" label.
* **Paper polish** (10ecd9b):
  * abstract rewritten (242 words); headline answer with the two exceptions; explicit contributions list
  * CLEAR positioning in related work and Section 4.5: our replay keeps CLEAR's cloning losses but not its V-trace
    RL loss on replayed data, and sleep = the same losses applied offline + self-distillation, so the matched test
    isolates *when* replay happens
  * self-contained captions; figures redrawn in Computer Modern at printed size (vector PDF) with the dashboard
    palette (non-method series in grey/violet)
  * a full reproducibility statement with a new number, `computeCoreHours` = 110 CPU hours over 289 runs,
    computed from run logs
  * "causal Transformer" corrected: there is no causal mask, but the window holds only past frames
  * two builds: `paper/main.pdf` (preprint, your name) and `paper/main_anonymous.pdf` (TMLR double-blind, no name,
    no repo URL); every page was rendered and checked
* **Presentation** (cac4e93):
  * `presentation/NeuroPlast_final_review.pptx` (+ `.pdf`): 15 slides + 4 backup, speaker notes on every slide
  * built with python-pptx from the same numbers pipeline, with the paper's figures, one native chart and the
    dashboard palette
  * converted with LibreOffice (libreoffice-impress installed for this) and every slide checked for overflow
* **Git history rewrite** (last):
  * backup: `git bundle create --all` (4 refs, verified, SHA-256 696470db…d324b6); sent to you in 17 parts of
    29 MB (upload limit 30 MiB), with `JOIN_BACKUP.txt`
  * branches before: GitHub `main` b414299 (fully contained in `session-3`), GitHub `session-3` 34e3d27. The only
    other branch, `claude/keen-lamport-wrnfpj`, was already gone from GitHub; its local copy (579f5c7, authored
    "Claude", with Co-Authored-By/Claude-Session trailers) had a file tree identical to adabbd9 (now 0b15bd0) in `session-3`, so
    nothing unique was lost (it is in the bundle)
  * `git filter-repo` on a fresh mirror: all 26 commits now have author and committer `Vaibhav Tiwari
    <75622110+VT69@users.noreply.github.com>`. 9 early commits had the display name "VT" with the same email.
    Claude/Anthropic trailers were removed (none were left on GitHub's branches)
  * the file tree was byte-identical to before (tree 21d76f2 at the time)
  * pushed with `--force-with-lease` pinned to the old hashes: `session-3` and `main` both at cac4e93 (plus this
    PROGRESS commit); no other remote branches exist, no PRs, no tags
  * old hashes in this file were replaced (4 places); README and paper had none

### Left for you to review / fill in (session 6)
1. Slide 1 of the deck: roll number, guide, department, institution, date.
2. Paper: affiliation and email (preprint); the AI-assistance disclosure; the venue of Yu et al.; the
   `% TODO verify` bib entries (Chaudhry, Neftci, Eshraghian, Bi & Poo, Frémaux, Turrigiano, Nøkland, Horowitz,
   Schulman, Huang, Vaswani, Welch, Holm, Newcombe); Khetarpal volume/pages; Mallya venue.
3. The anonymous build says the code is "provided as supplementary material": attach an anonymised copy when you
   submit (e.g. anonymous.4open.science), or change the sentence.
4. The preprint cites https://github.com/VT69/NeuroPlast: make sure the repo is public before the review.
5. ~~`relevant_literature/` (publisher PDFs) is committed in the repo~~ Done in session 7: removed and purged from history.
6. GitHub may keep the old, now-unreachable commits (including the old Claude-authored one) accessible by hash
   until its garbage collection runs, and the contributors list can lag. If Claude still shows as a contributor
   after a day, GitHub Support can purge cached views.
7. The compressed-replay idea on slide 14 is marked "idea, untested"; reword it if you want it framed differently.

## Session 5 (2026-09-30): paper draft and demo dashboard (no new experiments; results stay frozen)
Produced (all on `session-3`: paper in c838dce, dashboard in 5eb5bda; hashes after the session-6 history rewrite):
* **Paper draft** `paper/main.pdf` (15 pages, TMLR style, builds with `cd paper && latexmk -pdf main.tex`, no LaTeX
  errors). Sections: intro, related work, setup (pre-registration, Welch, Fisher, Holm), mechanisms, one results
  subsection per mechanism, "what didn't survive controls", limitations, conclusion, reproducibility, AI-assistance
  disclosure (TODO for you), appendices (protocol constants, extra figures).
* **Numbers pipeline** `paper/compute_numbers.py` regenerates `paper/numbers.tex` (359 LaTeX macros) and
  `paper/NUMBERS.md` (each number with its description and source file in `runs/` or `results/`). The text contains
  no hand-typed result numbers; every result goes through a macro. All values were cross-checked against the
  pre-registered session 3/4 tables.
* **Figures** `paper/figures/make_figures.py`: forgetting curves by method, accuracy vs total memory, the RQ1 arms,
  the op-count frontier, the Block 1 replay-budget comparison, and the memory probes (PDF and PNG).
* **Bibliography** `paper/refs.bib`: entries checked against `docs/papers/` PDFs, plus well-known external entries
  marked `% TODO verify`. No invented references.
* **Dashboard** `demo/index.html`: a single self-contained page (2.0 MB, data and GIFs embedded, works offline).
  It contains the scorecard (built from the paper's own numbers pipeline, so it matches the paper), the naive vs
  sleep GIFs with the run's real accuracy matrix and the 3-seed mean, forgetting curves for every method on 4
  sequences, accuracy vs total memory, table views, and dark mode. Rebuild: `python demo/build_dashboard.py`.
  Checked at 1280 px (light and dark) and 390 px: no console errors, no horizontal overflow.
* **Live episode** `demo/live_episode.py --agent {naive,sleep} --task {0,1,2} [--window] [--gif out.gif]`: one
  CPU episode from `runs/demo_ckpt/`, printed step by step, followed by the agent's measured accuracy matrix.

### Left for you to review
1. Paper TODOs (red in the PDF):
   * your affiliation and email
   * the repository URL (Reproducibility section)
   * **the AI-assistance disclosure text**; only you can write this accurately
2. Bibliography (`paper/refs.bib`):
   * entries marked `% TODO verify` (external, not in `docs/papers/`): Chaudhry, Neftci, Eshraghian, Bi & Poo,
     Frémaux, Turrigiano, Nøkland, Horowitz, Schulman, Huang, Vaswani, Welch, Holm, Newcombe
   * Khetarpal: volume and pages
   * Mallya: venue
   * Yu et al. (Self-Consolidation): venue and year are unknown and show as a visible note in the PDF
3. Submission mode: `\usepackage[preprint]{tmlr}` shows your name. An anonymous TMLR submission needs the option
   removed (then also drop the repo URL and author block).
4. Claim labels (`\lit`, `\ours`, `\interp`): check that every interpretive sentence is tagged the way you would tag it.
5. Protocol constants in appendix A were read from code and configs (see `NUMBERS.md` sources); worth one look.
6. The GRU memory baseline is implemented and tested but was never run (the Transformer question is untested). The
   paper says so.
7. The dashboard GIFs are 3 episodes per task in fixed evaluation rooms; the matrices next to them are the scored
   values. Nothing was chosen by outcome.

## Session 4 (2026-09-29/30) — final experimental session before the results freeze (complete, 01:25 UTC)
Machine: fresh container, 4 cores, 15 GB RAM, 20 GB free disk; 4 workers, OMP_NUM_THREADS=1. Heartbeat Monitor
(4-min status line) runs whenever the queue runs: the container is reclaimed ~5 min after the session goes idle
(session 3 lesson). Budget ~9 h x 4 cores = ~36 core-h; plan below uses ~30.

### Code changes (tests: `tests/test_session4.py`, full suite 28/28 passing)
* `Sleep(replay_budget=N)`: total old-task replayed states over the whole sequence, spread as evenly as possible
  over the schedule's sleep phases (replaces steps_per_phase). Test: a sleep run consumes exactly N.
* `continual.py` saves `final_agent.pt` after the last `end_task`. `taskN/final.pt` is saved *before* end_task,
  so for sleep it misses the last end-of-task sleep phase and is not the agent scored in R.
* `mem_kwargs={kind: gru}`: 1-layer GRU over the same window of encoded frames as the Transformer (non-attention
  memory baseline with identical context). Test: output matches the cached-feature path and depends on history.

### Pre-registration (written before any session-4 run; one planned test per block)
**Block 1 — sleep vs replay, replay samples matched exactly (RQ2).** The 28% mismatch exists only on the full
hybrid: sleep 16 phases x 700 steps x 128 = 1,433,600 old-task states per run vs replay 0.91-1.48M (replay's count
varies by seed because PPO's KL early stopping skips minibatches). CNN/SNN trunks are already ~matched (sleep
0.54M vs replay 0.55-0.59M), so the block runs on the hybrid. Design: sleep seed s gets
`replay_budget` = the measured `replay_samples` of replay seed s (exact, per seed), everything else identical
(cl_hybrid_v2.yaml: lr 3e-4, 400k frames/task, buffer 5000/task, fetch3). Existing replay seeds 1-6 are reused;
replay seeds 7-8 are new, then matched sleep seeds 7-8. Arm name `sleep_matched`.
Planned test: final ACC, sleep_matched vs replay, two-sided Welch, alpha 0.05, n = 8 vs 8.
Stopping rule (decided now): if seeds 7-8 of both arms don't finish this session, analyse seeds 1-6 (6 vs 6).
Descriptive only: FORGET/BWT; matched vs unmatched sleep on the same seeds (how much the extra replay mattered).
What "matched" does not remove: sleep still differs in *when* replay happens (offline phases vs interleaved) and
adds current-task rehearsal on the same number of steps; that is the mechanism under test.

**Block 2 — DFA with vs without homeostasis (RQ1-adjacent).** DoorKey-6x6, rq1_v2.yaml (400k frames), DFA on the
SNN encoder (`dfa=true dfa_scope=encoder`) vs the same + homeostasis (`stdp={alpha: 0.0, homeo: 0.15}`), 10 seeds
each (existing dfae s1-5 and dfae_homeo s1-3 reused; same config and code path, new seeds 6-10 and 4-10).
Planned tests (two statistics, one contrast; Holm across the two): (i) solve rate (final eval return > 0.5),
two-sided Fisher exact; (ii) AUC (mean training return), two-sided Welch. Alpha 0.05 after Holm.

**Block 3 — memory benchmark (Transformer question).** *Probe (sizing, no test):* CNN single frame (fs1) and CNN
frame stack 12 (fs12) on MiniGrid-MemoryS11 and -S13, lr 3e-4, 2M frames, seed 1 (4 runs). Why S11/S13: on S7 a
memoryless CNN got 91.5% because the 7x7 view (which includes the agent's own row) shows the cue and both choice
objects from one pose (cue 4-5 cells from the junction); on S11/S13 the cue is 8/10 cells from the junction,
beyond the 6-cell view depth, and the 1-wide hallway can't encode the cue by position, so a memoryless agent that
reaches the junction is at 50%. Window 12: after the last possible cue sighting an optimal path needs ~6 (S11) /
~8 (S13) more actions, so K=12 leaves slack. *Map rule:* use the smallest map with fs1 success <= 0.60 and
fs12 >= fs1 + 0.25. If neither qualifies: if fs12 is still rising at 2M, rerun S11 fs12 at 4M; otherwise report
that no memory benchmark was established within budget (negative result) and skip the main runs.
*Frames:* 2M per run, or 3M if the fs12 probe's training return rose by > 0.05 over its last 25% of frames.
*Main arms (3 seeds each, lr 3e-4 for all, window/stack K=12):* CNN fs1, CNN fs12, SNN fs12, SNN+Transformer
(window 12), and, last and only if budget allows, SNN+GRU (window 12).
Planned test: eval success (200 episodes, sampled policy), SNN+Transformer vs SNN fs12, two-sided Welch, n=3 vs 3.
Descriptive: CNN fs1 (memoryless floor), CNN fs12 (does memory help at all), SNN+GRU.
*Investigation trigger:* if SNN fs12 mean success < CNN fs1 mean - 0.20, before interpreting: run SNN fs1 on the
same map (is it the spiking trunk or the stacked input?) and check per-run encoder activity; the planned test is
still reported, labelled confounded if the diagnostic points at the trunk.

**Block 4 — demo assets (no test).** Checkpoints exist for every fetch3 CNN run, but the sleep ones are pre-final-
sleep (see code changes). Rerun seed 1 of fetch3 naive and sleep (CNN, cl_cnn.yaml) into `runs/demo_ckpt` with the
final-agent checkpoint (check: R should reproduce the original seed-1 runs), then record GIFs of each agent on every
task (sampled policy, same env seeds for both agents) into `demo/assets/` with an index.

### Block 1 result (our result, n=8/arm, pre-registered): full hybrid, sleep vs replay with replayed samples matched exactly (`results/s4_stats.md`), 01:25 UTC
Every seed matched exactly (sleep_matched consumed the same number of old-task replayed states as the same-seed
replay run: 0.91M-1.48M per run).
| metric | sleep_matched | replay | diff | Welch p | perm p | 95% bootstrap CI |
|---|---|---|---|---|---|---|
| **ACC (planned test)** | 0.879 ± 0.043 | 0.862 ± 0.038 | +0.017 | **0.415** | 0.402 | [-0.020, +0.053] |
| FORGET (descriptive) | 0.020 ± 0.011 | 0.019 ± 0.017 | +0.001 | 0.881 | 0.876 | [-0.012, +0.014] |
| BWT (descriptive) | -0.017 ± 0.015 | -0.016 ± 0.015 | -0.001 | 0.922 | 0.923 | - |
per-seed ACC: sleep_matched 0.901, 0.846, 0.828, 0.934, 0.910, 0.819, 0.886, 0.912; replay 0.844, 0.841, 0.817, 0.918,
0.818, 0.890, 0.876, 0.896. **Reading.** At an equal replay budget, sleep and replay are indistinguishable on
accuracy (p=0.42) and identical on forgetting (0.020 vs 0.019). Descriptive: the unmatched sleep runs (1.43M
states, seeds 1-6) had ACC 0.887 and FORGET 0.006; the same seeds matched give 0.873 and 0.020 (paired diff -0.014,
p=0.069). So sleep's small edge in sessions 2-3 (lower forgetting, +0.03 ACC) came from the extra ~28% of
replayed samples, not from consolidating offline. RQ2's answer is now a clean null on every trunk.

### Status notes
* 19:35: 5th worker started for the exploratory 6M probe (5 processes on 4 cores; slowed Block 1 ~15-25%).
* 23:58: container restarted (monitor and workers gone; this time not an idle reclaim: the heartbeat was
  running). Lost: matched sleep seed 8, ~55 min in, on its last task. Workers restarted at 23:58; seed 8 reran
  from scratch. Every other run had already finished (rc=0 throughout the session).

### Block 2 result (our result, n=10/arm): DFA with vs without homeostasis, DoorKey-6x6 (`results/s4_stats.md`), 20:00 UTC
| arm | solved | AUC | final eval | failing seeds (eval) |
|---|---|---|---|---|
| DFA + homeostasis | **7/10** | **0.402 ± 0.307** | 0.645 ± 0.443 | s4 (0.00), s9 (0.04), s10 (0.00) |
| DFA | 2/10 | 0.076 ± 0.101 | 0.188 ± 0.254 | 8 seeds (solvers: s5 0.71, s9 0.53) |
| planned test | raw p | Holm (2 tests) |
|---|---|---|
| solve rate, Fisher exact | 0.070 | 0.070 (**not significant**) |
| AUC, Welch (+0.326) | 0.0087 | **0.017 (significant)** |
**Reading.** Homeostasis makes DFA learn faster on average (AUC 5x higher, survives Holm), but the solve-rate effect
does not reach significance: with 10 seeds, 3 of the 7 new homeostasis seeds failed outright, so session 2's
"3/3 vs 1/5, p=0.0003 on AUC" was an overestimate of reliability. Interpretation: homeostasis is a real
but partial fix for DFA's runaway firing; it does not make weight-transport-free learning reliable on DoorKey-6x6.
Compare backprop + homeostasis (session 3): 10/10 solved, AUC 0.616.

### Block 3 result: no memory benchmark established within budget (pre-registered outcome), 19:35 UTC
Probe (1 seed, lr 3e-4, 2M frames, eval = 200 episodes, sampled policy; chance at the junction = 0.50):
| map | CNN single frame | CNN frame stack 12 | fs12 train return, 75% -> 100% of frames | episode length |
|---|---|---|---|---|
| MemoryS11 | 0.430 | 0.465 | 0.491 -> 0.509 (+0.018) | 9-10 steps |
| MemoryS13 | 0.485 | 0.460 | 0.491 -> 0.477 (-0.014) | 12-13 steps |
**Decision by the pre-registered rule:** neither map qualifies (fs12 would need >= fs1 + 0.25) and fs12 is not
still rising (<= 0.05 over the last 25%, the same threshold as the frames rule, since "still rising" had no
number), so the **main arms were not run** and the planned test (SNN+Transformer vs SNN fs12) does not exist.
**Reading (our result, n=1 per cell).** The memoryless floor works as intended: at chance on both maps, unlike
S7 (91.5%). But the memory-equipped CNN is also at chance, and episode lengths (9-13 steps ~ walking straight to
the junction) show why: no agent learned the information-gathering behaviour (turn back, look at the cue, then
go). With a frame stack the cue is in the input only in episodes that start next to it (about 1 in 8 starts on
S11), so the reward signal for using memory is weak. The bottleneck is exploration/credit assignment, not context
length, and a 2M-frame PPO budget is not enough on these maps. **Transformer question: remains untested**; the
honest statement is that none of our MiniGrid memory maps gave a usable benchmark at this compute (S7 doesn't need
memory, S11/S13 aren't learned by memory agents in 2M frames).
**Exploratory extension (NOT pre-registered, decided after seeing the probe):** S11 CNN fs12 at 6M frames, 1 seed
(~2.6 core-h of the ~14 freed by skipping the main arms), only to size a future test ("is it learnable with 3x
the frames?"). It enters no test. Queue `jobs/s4_q2.txt`.
*Result (22:50 UTC):* eval success 0.485; training return 0.485 / 0.485 / 0.493 / 0.507 / 0.503 / 0.482 at
1-6M frames, episode length ~10 throughout. **Flat at chance for 6M frames**: with 3x the frames the frame-stacked
CNN still never learns to look back at the cue. A future memory test needs either a map/variant where the cue is
seen on the way to the decision (e.g. the agent starts in the cue room facing the hallway), an exploration aid, or a
much larger budget (the MiniGrid literature reports recurrent PPO needing tens of millions of frames on the larger
Memory maps; not checked against the papers here, so treat as a hypothesis).

### Block 4 result: demo assets (done 19:02 UTC) -> `demo/assets/index.md`
Reruns of fetch3 naive and sleep, CNN trunk, seed 1 (`runs/demo_ckpt`, 4.5 and 6 min) **reproduce the original
seed-1 accuracy matrices exactly** (naive final row 0.126 / 0.542 / 0.956; sleep 0.978 / 0.979 / 0.979), so the
GIFs show the agents that were scored. `scripts/record_demo.py` renders each agent (after all 3 tasks) on each task:
3 episodes, sampled policy as in the eval, same rooms for both agents (the scored evaluation's first rooms).
Naive: task 2 (trained last) 3/3 episodes; task 0 1/3 (two 256-step timeouts); task 1 2/3. Sleep: 9/9.
(A dry run on arbitrary seed 777 had a naive agent spinning in place on task 2: checked against `evaluate()`,
it was that room, not a bug: 5/6 from that seed. The committed GIFs use the eval's own rooms instead.)

### Plan and status
| block | runs | est. core-h | status |
|---|---|---|---|
| 3 probe | 4 | 1.3 | **done** 19:35: no benchmark (see Block 3 result) |
| 4 reruns + GIFs | 2 | 0.6 | **done** 19:02 |
| 2 DFA +/- homeostasis | 12 | 2.5 | **done** 20:00 (see Block 2 result) |
| 1 replay s7-8, sleep_matched s1-6 | 8 | 9.5 | **done** |
| 3 main (4 arms x 3 seeds, +GRU if budget) | 12-15 | 11-17 | **not run** (pre-registered rule); exploratory 6M fs12 probe instead (2.6) |
| 1 sleep_matched s7-8 | 2 | 2.7 | **done** 01:23 (seed 8 rerun after the 23:58 restart) |

## Session 3 (2026-09-28) — controls & statistical confidence (complete, ~19:40 UTC)
No new mechanisms. Colab unavailable, so extra seeds ran here.
Machine: 4 cores, 15 GB RAM, 20 GB free disk -> 4 workers, OMP_NUM_THREADS=1.
Budget ~9 h x 4 cores = ~36 core-h. Runtime estimates are session-1/2 means measured under
the same ~4-jobs-on-4-cores load.

Repo note (hashes updated after the session-6 history rewrite): GitHub `main` then (0bb8ab5;
session 2 was 5d6397c) had the session-2 work inside a `neuroplast_session2_code_results/`
subfolder (plus the zip itself); the repo root there is still session-1 code. This working copy (pushed as
branch `session-3`) has the session-2 code at the root (correct) + 0bb8ab5's CLAUDE.md and
docs/LITERATURE_CONTEXT.md; relative to main it drops `neuroplast_session2_code_results/`, its zip and the
old empty `src/`. The old branch `claude/keen-lamport-wrnfpj` is already gone from GitHub. §38.4 of the literature doc said backprop+homeostasis was never
run: it was (3 seeds, session 2); tonight took it to 10 (correction note in §38.8).

### Plan, ordered by scientific value per core-hour
| # | block | jobs | est. core-h | status |
|---|---|---|---|---|
| P3a | isolation-budget probes: single-task SNN on FetchObj5 tasks, 600k frames (sizes P3b) | 4 | 1.1 | **done** -> budget 450k/task |
| P1 | RQ1 DoorKey-6x6 to 10 seeds: bp, bp+homeo, bp+stabilised STDP a=0.03, bp+random a=0.03+homeo (matched control) | 27 | 4.8 | **done**, n=10/arm, analysed below |
| P2 | memory-fair RQ5: analysis (no compute) + SNN fetch3 sleep/replay at buffers 200 and 2000/task, 3 seeds | 12 | 3.6 | **done** ~14:36 UTC (12/12), result below |
| P3b | fetch5 SNN isolation + sleep at 450k frames/task, 3 seeds | 6 | ~8 | **done** ~17:59 UTC (6/6), result below |
| P5 | memory pilot: membrane-reset check (code), MiniGrid-MemoryS7 + CNN frame-stack baseline | 5 | ~2 | **done** ~17:05 UTC (5/5), result below |
| P4 | hybrid sleep + replay, seeds 4-6, same protocol (lr 3e-4, 400k/task) | 6 | ~11 | **done** 19:36 UTC (6/6), result below |
Why P2-P5 didn't finish: a platform permission-checker outage (~12:45 onward) blocked every shell action for
hours, and the container was paused for most of that time, freezing the four workers mid-job (each log ended
on a START with no END, which first looked like a kill). They were **not** killed: when the container resumed
they carried on by themselves, and at commit time they are running the four P2 seed-1 jobs (SNN fetch3
sleep/replay at buffers 200 and 2000). Per the user's instruction they were neither restarted nor stopped.
Those four in-flight run directories (`runs/continual_snn_budget/`) are left out of the `session-3` commit
(half-written). Every job that finished exited rc=0, no crashes. The remaining queue is below.
**Update 2026-09-28 13:04 UTC:** the container restarted and the workers were gone. P2 buf200 seed 1 (sleep and
replay) had finished. At the user's request, 4 one-slot workers were restarted on
`jobs/s3_q1.txt -> s3_q2 -> s3_q3 -> s3_q4` (logs appended to `runs/logs/s3_worker_{1..4}.log`, 10 P2 jobs left
in q1). Expected wall time is ~6-7 h if the container stays up. The finished runs exist only in this container
until they're committed.
**Monitoring (user's standing instructions):** check every ~15 min, restart workers if gone, commit + push each
finished block (P2, P3b, P5, P4) to `session-3` as the user, record the SNN per-memory result right after P2.
Helper: `python scripts/_s3_monitor.py [--restart | --commit "msg"]` (git-excluded). 13:25 UTC check: the
permission classifier was down again (shell and scheduling blocked), so the 15-min check-ins stopped re-arming;
worker logs showed the 4 P2 jobs running since 13:04, P2 at 2/12. 13:28: the container had restarted again
(workers gone; the 4 in-flight P2 jobs were near the end of task 3 and lost that progress). Workers restarted,
check-ins re-armed. Note: a restarted continual run starts from task 0, not from its last finished task.
13:46: workers gone again; the container had rebooted at 13:46:15 (the moment the check-in fired) and the last
run write was 13:34, ~5 min after the session went idle. **Diagnosis: the container is reclaimed a few minutes
after the Claude session goes idle, so 15-min wake-ups can never let a 15-20 min job finish.** Fix: a Monitor
with a 4-min heartbeat keeps the session active while workers run (30-min send_later as fallback).
It worked from 13:46 to ~16:05 (P2 finished, P3b 5/6, P5 3/5). ~16:05-16:29 the container was reclaimed again:
a scheduling call sat waiting for manual approval (~25 min, auto mode off), the session counted as idle, and the
in-flight P3b sleep seed 3 (~1 h into ~87 min) and P5 runs were lost. Workers restarted 16:29.

### P1 RQ1 — FINAL, pre-registered (n=10/arm, DoorKey-6x6, 400k frames; `results/rq1_stats.md`)
solved = final eval return > 0.5. Fisher = two-sided exact test on solve counts; Welch = two-sided on AUC
(mean train return) and final eval return; Holm across the four contrasts, separately per statistic.
| arm | solved | AUC | final eval | frames to 0.9 (solvers) |
|---|---|---|---|---|
| bp | 8/10 | 0.416 ± 0.232 | 0.766 ± 0.398 | 251k |
| homeo (bp + homeostasis) | **10/10** | **0.616 ± 0.113** | **0.917 ± 0.137** | 179k |
| tfs0.03 (bp + stabilised STDP, incl. homeostasis) | 8/10 | 0.515 ± 0.300 | 0.765 ± 0.395 | 178k |
| rands0.03 (bp + same-RMS random update + homeostasis) | 8/10 | 0.482 ± 0.323 | 0.742 ± 0.400 | 160k |

| contrast | Fisher p (Holm) | ΔAUC | Welch AUC p (Holm) | Welch eval p (Holm) |
|---|---|---|---|---|
| (a) tfs0.03 vs homeo: does STDP add beyond homeostasis? | 0.474 (1.000) | -0.101 | 0.338 (0.724) | 0.273 (0.864) |
| (b) tfs0.03 vs rands0.03: beyond matched random perturbation? | 1.000 (1.000) | +0.033 | 0.818 (0.818) | 0.899 (0.899) |
| (c) homeo vs bp: does homeostasis help backprop? | 0.474 (1.000) | +0.200 | 0.029 (0.117) | 0.280 (0.864) |
| (d) rands0.03 vs homeo: does perturbation add beyond homeostasis? | 0.474 (1.000) | -0.134 | 0.241 (0.724) | 0.216 (0.864) |
**All four pre-registered contrasts are null after Holm correction, on every statistic.** The only raw
p < 0.05 is (c) on AUC (homeostasis learns faster than plain backprop, +0.20 AUC, p=0.029), which does not
survive Holm (0.117). Point estimates: homeostasis is best on every column (10/10 solved, lowest variance);
STDP is indistinguishable from a same-size random update (b: p=0.82) and, if anything, slightly below
homeostasis alone. Plain-backprop failures are seeds 3 and 9; STDP's are 3 and 7; random's include 7.
At n=10, bimodal outcomes give limited power: a solve-rate difference of 10/10 vs 8/10 can't be
significant by Fisher (minimum p=0.47), so only continuous metrics could reach significance here.
The session-2 "homeostasis helps backprop" hint (p=0.09 at n=3) moved to raw p=0.029 at n=10: suggestive,
not confirmed. Context arms (not pre-registered, n=3-5) are in `results/rq1_stats.md`; the DFA ones repeat
session 2: DFA alone 1/5 solved, DFA+homeostasis 3/3.

### P4 result (our result, n=6/arm): full hybrid (SNN + Transformer), fetch3, sleep vs replay (`runs/continual_hybrid`)
Same protocol as session 2 (cl_hybrid_v2.yaml: lr 3e-4, 400k frames/task, buffer 5000/task); seeds 4-6 added.
| metric | sleep | replay | diff | Welch p | perm p | 95% bootstrap CI |
|---|---|---|---|---|---|---|
| ACC | 0.887 ± 0.045 | 0.855 ± 0.041 | +0.032 | 0.22 | 0.22 | [-0.012, +0.075] |
| FORGET | 0.006 ± 0.021 | 0.021 ± 0.017 | -0.016 | 0.19 | 0.18 | [-0.035, +0.004] |
| BWT | -0.003 ± 0.021 | -0.019 ± 0.015 | +0.016 | 0.18 | 0.18 | [-0.004, +0.035] |
per-seed ACC: sleep 0.897, 0.854, 0.868, **0.946, 0.926, 0.829**; replay 0.844, 0.841, 0.817, **0.918, 0.818, 0.890**
(seeds 4-6 bold). **Reading.** The n=3 trend (p=0.07, "every sleep seed > every replay seed") did not hold up:
seed 6 reverses it, and at n=6 sleep vs replay is **not significant** on ACC (p=0.22) or forgetting (p=0.19).
Point estimates still favour sleep slightly (+0.03 ACC, lower forgetting). Confound noted, not new: sleep consumes
~28% more replayed samples than replay (1.43M vs 1.12M per run), so even this small edge is not budget-matched.
Also with n=6: hybrid sleep vs isolation ACC 0.887 vs 0.832 (p=0.11), FWT +0.42 (p=0.095).

### P3b result (our result, n=3/arm): fetch5 SNN at a fair isolation budget, 450k frames/task (`runs/continual5_snn_450k`)
Pre-registered: sleep vs isolation ACC (Welch + permutation), sleep FWT vs 0 (one-sample t), tasks learned
(R[k][k] >= 0.8) per arm. Same config for both arms (cl_snn.yaml, 450k frames/task; sleep buffer 5000/task).
| metric | sleep (shared) | isolation | test |
|---|---|---|---|
| ACC | 0.915 ± 0.014 | 0.871 ± 0.026 | +0.044, Welch p=0.078, perm p=0.20 (min 0.10), bootstrap CI [+0.015, +0.069] |
| ACC per 100k params | 0.400 | 0.109 | p<0.001 |
| ACC per MB total memory | 0.163 (5.62 MB) | **0.272** (3.20 MB) | isolation ahead, as in P2 |
| FWT | +0.31 ± 0.17 | 0 (reference) | one-sample p=0.089 |
| tasks learned (>= 0.8) | 14/15 | 12/15 | - |
| FORGET | 0.007-0.017 | 0.00-0.03 (eval noise: isolation can't forget) | - |
Session 2 at 150k/task had sleep 0.778 vs isolation 0.475 (p=0.02, n=2). **Reading.** (1) Most of that
gap was the short budget: at 450k/task isolation rises to 0.871 and the gap shrinks from +0.30 to +0.04,
which is no longer significant by the pre-registered tests (Welch 0.078, perm 0.20; the bootstrap CI excludes 0
but is unreliable at n=3). (2) What remains is consistent with positive transfer: later tasks are learned better
by the shared net (task 4: 0.95-0.97 vs 0.69-0.82 for fresh nets; FWT +0.31, p=0.089). Task 0 is slow for both
arms (0.73-0.86), as the P3a probes predicted. (3) Per total memory at the default 5000-state buffer, isolation is
still ahead (0.27 vs 0.16 ACC/MB); a small-buffer fetch5 SNN arm was not run.

### P5 result (pilot, 1 seed/arm, sizing only): MiniGrid-MemoryS7, 1M frames (`runs/s3_mem_pilot`)
Random policy: 24% success. Eval: 200 episodes, stochastic (sampled) policy, as in all other NeuroPlast evals.
| arm | memory mechanism | lr | eval success | train return @250k / 500k / 1M |
|---|---|---|---|---|
| CNN, 1 frame | none | 1e-3 | 0.915 | 0.47 / 0.88 / 0.91 |
| CNN, frame stack 4 | last 4 frames as channels | 1e-3 | **0.995** | 0.58 / 0.90 / 0.96 |
| CNN, frame stack 8 | last 8 frames | 1e-3 | 0.835 | 0.72 / 0.88 / 0.84 |
| SNN, frame stack 8 | last 8 frames | 1e-3 | 0.465 | 0.48 / 0.41 / 0.54 |
| SNN + Transformer, window 8 | attention over 8 encoded frames | 3e-4 | 0.970 | 0.47 / 0.47 / 0.96 |
**Reading (our result, n=1, no statistics).** (1) **MemoryS7 does not isolate memory**: a memoryless CNN reaches
91.5% (a policy that can't recall the cue should be near 50% at the choice point). Interpretation: in the 7-wide
map the agent can carry the cue in its own position/heading ("memory in the environment"), or the cue is still
in view near the decision point. So a Transformer benefit can't be shown cleanly on S7. (2) Memory still helps a
little: frame stack 4 takes the CNN from 0.915 to 0.995. (3) On the SNN, channel-stacking 8 frames fails (0.47),
while the Transformer over 8 SNN-encoded frames solves it (0.97), but only after 500k frames and with a different
lr (3e-4 vs 1e-3: the protocol confound carried over from session 2's hybrid). Hypothesis: attention integrates
multi-frame spiking input better than wide channel stacks; untested at n>1 or matched lr.
**Next step for the Transformer question:** a map where the memoryless baseline is at chance (MemoryS11/S13; check
with a 1-frame CNN first), 3 seeds per arm, matched lr (3e-4 for all arms, or a small lr sweep per arm), arms:
1-frame, frame stack 4, SNN+Transformer, CNN+Transformer.

### Pre-registered comparisons and statistics (written before running)
* **RQ1** (DoorKey-6x6, 400k frames, n=10/arm; arms bp, homeo, tfs0.03, rands0.03):
  outcomes are bimodal, so the primary statistic is the solve rate (final eval return > 0.5), Fisher's
  exact test (two-sided); secondary is AUC (mean train return) and final eval return, Welch t-test.
  Primary contrasts: (a) tfs0.03 vs homeo (does STDP add beyond homeostasis?); (b) tfs0.03 vs
  rands0.03 (beyond matched random perturbation?); (c) homeo vs bp (does homeostasis help backprop,
  or only DFA?); (d) rands0.03 vs homeo (does generic perturbation add beyond homeostasis?).
  Holm correction across (a)-(d) reported alongside raw p.
* **RQ5 memory-fair**: total memory = fp32 parameters + stored replay states at measured size (188 B/state
  plain trunks, 632 B/state hybrid; minimum format, the implementation's 2x cache noted separately);
  sleep also holds a transient 5000-state reservoir + a network snapshot during training (reported as peak).
  Tests: sleep/replay at small buffers vs isolation on ACC, Welch + exact permutation + bootstrap CI.
* **Isolation budget** (P3b): at a budget where fresh SNNs learn each task, sleep vs isolation ACC
  (Welch + permutation), sleep FWT vs 0 (one-sample t); per-task "learned" = R[k][k] >= 0.8, counts per arm.
* **Hybrid** (P4): sleep vs replay ACC and FORGET, Welch, with n = 5-6.
* Pilots (P3a, P5): sizing only, no conclusions.

### P3a probes (sizing): fresh single-task SNN on fetch5 tasks, 600k frames (`runs/s3_probes`)
| probe | 150k | 300k | 450k | 600k | first >= 0.8 |
|---|---|---|---|---|---|
| task 0, seed 1 | 0.36 | 0.49 | 0.73 | 0.85 | 595k |
| task 2, seed 1 | 0.68 | 0.86 | 0.91 | 0.91 | 129k |
| task 3, seed 1 | 0.62 | 0.87 | 0.80 | 0.88 | 219k |
| task 4, seed 2 | 0.63 | 0.82 | 0.89 | 0.91 | 188k |
150k/task (the session-2 protocol) was clearly too short for isolation. **Decision: 450k frames/task for all
fetch5 SNN arms in P3b** (3/4 probes >= 0.8 by then; 600k would be more reliable but ~11 vs ~8 core-h and would
push out P4). Caveat carried into the result: a fresh net can still be slow on some tasks at 450k (task 0 above).

### P5 check (no compute): SNN membrane state resets every environment step
`SNNEncoder.forward` initialises every layer's membrane to zero on each call (`state = [(None, None)] * 4`), and
each environment step is one call; verified empirically (same frame gives identical output regardless of the
preceding frames). The SNN integrates only over its T=4 internal steps within a frame. **So "SNN, single frame" in
the section-27 ablation is a memoryless baseline, not implicit temporal memory.** Persistent membrane state would
be a new mechanism (not tonight). Added a frame-stack memory baseline (`frame_stack=k`: last k one-hot frames as
input channels, pre-episode padding zeroed; `tests/test_agent.py`) and MiniGrid-MemoryS7 (random policy: 24%
success, max 245 steps). Pilot queue: `jobs/s3_q3.txt`.

### P2 result (no new compute): memory-fair RQ5 changes an earlier claim
Measured replay-buffer cost (TaskBuffer, `scripts/memory_fair.py`): **188 B per stored state** on the
CNN/SNN trunks (uint8 7x7x3 observation 147 B + mask 1 + task id 8 + teacher logits 28 + value 4) and
**632 B** for the hybrid (4-frame window). The code keeps a second concatenated copy (2x RAM; an
implementation artefact, reported separately). Full table: `results/rq5_memory.md`, figures
`results/figs/rq5_memory_fetch3.png` / `_fetch5.png`.
| setting | isolation total MB (ACC) | sleep @5000/task total MB (ACC) | ACC per MB iso vs sleep |
|---|---|---|---|
| CNN fetch3 | 1.92 (0.979) | 3.60 (0.978) | 0.51 vs 0.27 |
| SNN fetch3 | 1.92 (0.833) | 3.60 (0.889) | 0.43 vs 0.25 |
| Hybrid fetch3 | 3.52 (0.832) | 10.79 (0.887, n=6) | 0.24 vs 0.08 |
| CNN fetch5 | 3.20 (0.976) | 5.62 (0.928) | 0.31 vs 0.17 |
| CNN fetch3, sleep @200/task | 1.92 (0.979) | **0.89 (0.964)** | 0.51 vs **1.08** |
**Flag:** the session-1/2 claim "shared weights are 2.5-3.5x more efficient than isolation" is true *per
parameter* only. Counting the replay buffer, at the default 5000 states/task isolation uses *less* total
memory and is ~2x more ACC-per-MB efficient everywhere. Shared weights win per total memory only with small
buffers (CNN: 200/task -> 0.964 ACC at 0.89 MB). The SNN-trunk small-buffer arms (P2 runs) decide
whether that holds where transfer exists: see next section.

### P2 result (our result, n=3/arm): SNN trunk, fetch3, small buffers vs isolation (`results/rq5_stats.md`)
Pre-registered test: arm vs isolation on ACC (Welch + exact permutation + bootstrap CI), plus ACC per MB of total
memory. Isolation: 1.92 MB, ACC 0.833 ± 0.073. Same protocol as all session-1/2 SNN fetch3 arms (150k frames/task).
| arm | total MB | ACC | ACC diff vs iso (Welch p / perm p / 95% CI) | ACC per MB (iso 0.433) | FORGET |
|---|---|---|---|---|---|
| sleep @200/task | 0.89 | 0.838 ± 0.016 | +0.005 (0.92 / 1.00 / [-0.07, +0.07]) | **0.940** (p=0.0003) | 0.04-0.08 |
| replay @200/task | 0.89 | 0.792 ± 0.183 | -0.041 (0.75 / 0.90 / [-0.25, +0.11]) | 0.889 (p=0.057) | 0.03-0.11 |
| sleep @2000/task (memory-matched to isolation) | 1.91 | 0.847 ± 0.083 | +0.014 (0.84 / 1.00 / [-0.08, +0.11]) | 0.444 (p=0.76) | 0.00-0.08 |
| replay @2000/task (memory-matched) | 1.91 | 0.828 ± 0.167 | -0.005 (0.96 / 1.00 / [-0.18, +0.14]) | 0.434 (p=0.99) | 0.00-0.05 |
**Reading.** (1) On the SNN trunk, shared-weight sleep with 200 states/task matches isolation's ACC (0.838 vs 0.833,
p=0.92) in 46% of the memory, so ~2.2x the ACC per MB; the per-MB gain is mostly arithmetic (memory is fixed per
arm), the test that matters is the ACC tie. (2) At matched total memory (2000/task, 1.91 vs 1.92 MB) sleep, replay
and isolation are indistinguishable (all p>=0.84): the positive transfer seen per parameter (FWT +0.20) does
**not** turn into a raw-ACC advantage at equal memory. (3) Sleep is much more consistent than replay at 200/task
(SD 0.016 vs 0.183; replay seed 3 collapses to 0.58), but sleep vs replay is not significant at n=3.
(4) Seed 3 is the weak seed for both 2000/task arms (0.75, 0.64); isolation's weak seed is s2 (0.76): SNN runs are
high-variance, and n=3 can only detect large effects (a tie here means "no large difference", not equivalence).
Compared with the CNN: small buffers cost the CNN a little ACC (sleep@200 -0.015, CI excludes 0) while the SNN
loses none, consistent with the SNN being the trunk where sharing helps.

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
        **[Session 3: superseded. At n=6 sleep vs replay is p=0.22; seed 6 reverses the ordering. See P4 result.]**
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
        **[Superseded at n=10: backprop+homeo 10/10, AUC Holm p=0.117 (session 3); DFA+homeo 7/10 vs DFA 2/10,
        Fisher p=0.070, AUC Holm p=0.017 (session 4).]**
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

## Status at end of session 1 (2026-09-27 ~02:30)
All six items on the priority list are done, plus first-pass answers on all five RQs.
- [x] 1. Scaffold + deps + MiniGrid smoke test (21 unit tests, `pytest tests/`)
- [x] 2. CNN + PPO baseline: DoorKey-5x5 / LavaGapS5 / DoorKey-6x6, 100% eval success
- [x] 3. SNN encoder (tested vs snnTorch) + RQ3 accuracy-vs-sparsity frontier
- [x] 4. STDP / eligibility-trace / three-factor module (tested)
- [x] 5. Hybrid agent (SNN + Transformer + heads) solves DoorKey-5x5
- [x] 6. fetch3 continual suite: naive, EWC, replay, sleep, isolation; CNN and SNN trunks
- [x] Beyond the list: variant B (local-only encoder), variant C (DFA / e-prop, no backprop),
      stabilised STDP, sleep+STDP, small-buffer RQ2, compute benchmark (RQ4), Colab sweep notebook

## Answers so far (FINAL, after session 4; tables in `results/`, p = Welch t-test unless noted)
**RQ1: does a local Hebbian/STDP term alongside the global rule help?** *No evidence that it does. At
n=10 with pre-registered tests, nothing is significant after Holm; STDP is indistinguishable from a
same-size random update.*
- Vanilla STDP hurts (runaway excitation; p≈0.02 on DoorKey-5x5, session 1). Large stabilised STDP
  (alpha 0.3) also breaks learning.
- Session 3, DoorKey-6x6, n=10/arm: bp 8/10 solved, bp+homeostasis 10/10, bp+stabilised STDP 8/10,
  bp+matched random 8/10. STDP vs random: p=0.82 (AUC). STDP vs homeostasis alone: ΔAUC -0.10, p=0.34.
  Homeostasis vs plain backprop: ΔAUC +0.20, raw p=0.029, **Holm 0.117: not significant**.
- With DFA as the global term (no weight transport), homeostasis helps a lot but does not make it reliable.
  Session 4, pre-registered, n=10/arm: DFA+homeostasis 7/10 solved vs DFA 2/10 (Fisher p=0.070, Holm 0.070,
  **not significant**); AUC 0.40 vs 0.08 (Welch p=0.0087, **Holm 0.017, significant**). Session 2's 3/3 vs 1/5
  overstated the reliability: 3 of the 7 new homeostasis seeds failed. This is still the largest
  local-plasticity effect in the project, and it's about homeostasis, not STDP.
**RQ2: does sleep beat replay / EWC?** *Beats EWC and naive everywhere; ties replay on every trunk. With the
replay budget matched exactly (session 4, pre-registered, hybrid, n=8/arm) sleep = replay on accuracy (0.879 vs
0.862, p=0.42) and forgetting (0.020 vs 0.019): its earlier small edge came from ~28% more replayed samples.*
- CNN trunk: sleep 0.978 vs replay 0.970 (p=0.28). SNN trunk: 0.889 vs 0.814 (p=0.51).
- Full hybrid (SNN + Transformer), n=6 (session 3): sleep 0.887 vs replay 0.855 (p=0.22); forgetting 0.006 vs
  0.021 (p=0.19) vs naive 0.54. The n=3 trend (p=0.07) did not survive more seeds.
- Budget-matched, full hybrid, n=8 (session 4): sleep 0.879 vs replay 0.862, p=0.42; forgetting 0.020 vs 0.019.
  Matching the budget removes the forgetting difference entirely (unmatched sleep, same seeds: 0.006).
- 5 tasks (CNN): sleep 0.928 = replay 0.926 (p=0.97), both ~0 forgetting; naive 0.416.
- Small buffers: no significant difference. EWC: stability-plasticity trade-off.
- STDP inside sleep: vanilla catastrophic, stabilised harmless and useless.
**RQ3: SNN vs CNN on accuracy vs sparsity.** *Qualified yes* (unchanged; `results/rq3_doorkey6.md`):
equal at matched ops above ~33k ops/frame; below that only the SNN works (~2x fewer ops);
the big energy gap assumes neuromorphic per-op costs.
**RQ4: most robust per unit of compute?** *The plain CNN, ~5.7x ahead of any spiking variant*
(`results/rq4.md`; continual ACC per unit of training compute: CNN 0.98, SNN + sleep 0.17, SNN + homeostasis
0.16, DFA + homeostasis 0.09, stabilised STDP 0.08, full hybrid 0.06). Local-only encoders (B) never solve.
(Corrected in session 3: the earlier text quoted stale values, 0.21/0.11/0.09/0.08 and "~4x"; the ranking is
unchanged except that STDP and DFA+homeostasis are now effectively tied.)
**RQ5: shared weights vs isolation.** *Shared weights win per parameter, but per total memory
(parameters + replay buffer) only at small buffers. At the default 5000 states/task, isolation is ~2x
more ACC per MB everywhere we measured.* (`results/rq5_memory.md`, `results/rq5_stats.md`)
- Per parameter (unchanged): ACC per 100k params, shared vs isolation: CNN 0.50 vs 0.20; SNN 0.46 vs 0.17;
  hybrid 0.27 vs 0.10; 5-task CNN 0.41 vs 0.12 (all p<0.01).
- Per total memory, the session-1/2 "2.5-3.5x more efficient" claim reverses. The replay buffer costs
  188 B/state (632 B for the hybrid), so 5000/task is 2.8 MB on fetch3 against 0.78 MB of shared weights.
  ACC per MB, isolation vs sleep@5000: CNN 0.51 vs 0.27; SNN 0.43 vs 0.25; hybrid 0.24 vs 0.08;
  5-task CNN 0.31 vs 0.17; 5-task SNN 0.15 vs 0.14 (near parity only because isolation fails at 150k/task).
- Small buffers restore the shared-weight advantage. CNN: sleep at 200/task gets 0.964 ACC in 0.89 MB
  (1.08/MB vs isolation 0.51/MB; a small ACC cost, -0.015). SNN (session 3, n=3): sleep at 200/task matches
  isolation's ACC (0.838 vs 0.833, p=0.92) in 0.89 vs 1.92 MB (0.94 vs 0.43 ACC/MB).
- At *matched* total memory (SNN, 2000 states/task = 1.91 MB vs isolation 1.92 MB), sleep 0.847, replay 0.828,
  isolation 0.833: no difference (p>=0.84, n=3). Shared weights' advantage is memory efficiency at small
  buffers, not higher accuracy at equal memory.
- Transfer depends on the trunk: positive where learning from scratch is slow (SNN +0.20, p=0.05; hybrid
  +0.42, p=0.10; 5-task SNN +0.31, p=0.089), ~0 on the 3-task CNN, negative on the 5-task CNN (plasticity
  cost).
- The 5-task SNN "shared sleep beats isolation 0.778 vs 0.475 (p=0.02)" result was mostly a budget artefact.
  With a fair 450k frames/task (session 3, n=3), isolation reaches 0.871 and sleep 0.915: +0.044, Welch
  p=0.078, perm p=0.20, not significant. Sleep learns 14/15 tasks to >= 0.8 vs 12/15.
**Memory / Transformer question.** *Untested: no usable memory benchmark at this compute.* The SNN resets its
membrane every environment step, so "SNN, single frame" is memoryless. On the Markov fetch tasks the
Transformer costs ~3x sample efficiency and has nothing to remember. MiniGrid-MemoryS7 doesn't need memory (a
memoryless CNN gets 91.5%: the cue and both choices are visible from one pose). On MemoryS11/S13 the memoryless
CNN is at chance, as needed, but so is a CNN with a 12-frame stack, at 2M frames and (S11, exploratory) at 6M:
no agent learned to go back and look at the cue, so the pre-registered rule skipped the Transformer comparison
(session 4). The only Transformer data point is the n=1 S7 pilot (SNN+Transformer 97% vs SNN frame stack 47%,
different lr), which can't carry a claim.

## Evidence hierarchy (FINAL, as of session 4)
| claim | evidence | strength |
|---|---|---|
| Plain CNN is the most compute-efficient (RQ4) | ~5.7x margin, consistent over tasks | **strong** |
| Sleep/replay prevent forgetting vs naive/EWC (RQ2) | large effects, all trunks, 3-5 tasks | **strong** |
| Shared weights > isolation per parameter (RQ5) | all p<0.01 | **strong** (but see next row) |
| Isolation > shared weights per total memory at default buffers (RQ5) | arithmetic on measured sizes; no stats needed | **strong** |
| Shared-weight sleep at small buffers matches isolation ACC in <half the memory (RQ5) | CNN n=3 (small ACC cost); SNN n=3, ACC tie p=0.92 | moderate (n=3; a tie, not equivalence) |
| Shared weights beat isolation on ACC at matched total memory (RQ5) | SNN n=3, p>=0.84 | **unsupported (null)** |
| Vanilla / large STDP hurts (RQ1) | p≈0.02, mechanism identified (runaway excitation) | moderate |
| Homeostasis speeds up DFA learning (weight-transport-free) | pre-registered, n=10: AUC 0.40 vs 0.08, Holm p=0.017 | **moderate-strong** |
| Homeostasis makes DFA reliably solve DoorKey-6x6 | pre-registered, n=10: 7/10 vs 2/10, Fisher p=0.070 | **not significant** (was: moderate at n=3-5) |
| SNN beats CNN only at very low op budgets (RQ3) | 2 seeds/point, clean frontier | moderate |
| Homeostasis helps backprop (RQ1) | n=10, raw p=0.029 on AUC, Holm 0.117; solve rate 10/10 vs 8/10 (p=0.47) | **preliminary, not significant** |
| Sleep > replay on the full hybrid (RQ2) | budget-matched, pre-registered, n=8: +0.017 ACC, p=0.42; FORGET 0.020 vs 0.019 | **unsupported (null)** |
| Sleep's earlier forgetting edge came from extra replay, not offline consolidation (RQ2) | same seeds unmatched vs matched: FORGET 0.006 -> 0.020, ACC -0.014 (p=0.069, descriptive) | moderate (descriptive, n=6) |
| Positive transfer on slow-learning trunks (RQ5) | FWT p=0.05-0.10 on SNN/hybrid/5-task SNN, n=3 | preliminary |
| Shared-weight sleep beats isolation on raw ACC, 5-task SNN (RQ5) | fair budget: +0.044, p=0.078 (perm 0.20), n=3; the p=0.02 result was a short-budget artefact | **not significant** (was: preliminary) |
| STDP adds anything beyond homeostasis or random updates (RQ1) | n=10, p=0.34 / 0.82 | **unsupported (null)** |
| Transformer memory earns its cost | no benchmark: S7 needs no memory; S11/S13 not learned by memory agents in 2-6M frames | **untested** |

## Future work (after the freeze; nothing here is running or queued)
| candidate | runs | est. core-h | would answer |
|---|---|---|---|
| A memory benchmark agents can learn: MemoryS11 with the agent starting in the cue room facing the hallway (cue seen on the way), or an exploration aid; confirm the memoryless CNN is at chance and a frame-stack CNN beats it before any Transformer arm | 4 probes, then ~12-15 | ~2, then ~12-15 | the Transformer question (still untested) |
| fetch5 SNN sleep at a small buffer (200/task), 450k frames/task, 3 seeds | 3 | ~4.5 | RQ5 per-memory on the 5-task SNN (isolation wins at 5000/task) |
| Homeostasis target-rate sweep (bp+homeo and DFA+homeo, 3 targets x 5 seeds) | 30 | ~6 | whether the homeostasis effect is robust to its one hyperparameter, and why 3/10 DFA+homeo seeds fail |
| More seeds for RQ3's frontier points (2 seeds/point now) | ~20 | ~4 | RQ3's "SNN wins below ~33k ops/frame" at n>=5 |
Persistent membrane state as an SNN memory mechanism is a new mechanism: it needs the CLAUDE.md 4-point
justification first.

## What to look at first when you're back
0. **Session 7** (wording fixes, history purge), then **Session 6**: the review list, the deck (`presentation/`), the two paper builds, and the
   git-history note. Then **Session 5**: the paper draft (`paper/main.pdf`, numbers in `paper/NUMBERS.md`), the dashboard
   (`demo/index.html`), and the review list (TODOs only you can fill in).
1. **RESULTS FREEZE** note at the top, then "Answers so far" and "Evidence hierarchy" (both final).
2. Session 4 at the top: the pre-registration, then the four block results. Block 1 (sleep = replay once the replay
   budget is matched, p=0.42), Block 2 (homeostasis speeds DFA learning, AUC Holm p=0.017, but the solve rate 7/10
   vs 2/10 is not significant), Block 3 (no memory benchmark: the Transformer question stays untested), Block 4
   (`demo/assets/index.md`: 6 GIFs, naive vs sleep after the full sequence).
3. Tables: `results/s4_stats.md` (session-4 planned tests), `results/rq1_stats.md`, `results/rq5_stats.md`,
   `results/rq5_memory.md` (+ `results/figs/rq5_memory_fetch3.png`), `results/continual.md`, `results/rq4.md`.
4. `docs/LITERATURE_CONTEXT.md` §38.8: correction note from session 3. PROGRESS.md overrides it where they differ
   (e.g. session 4's Block 1/2 results postdate it).
5. Git: history was rewritten in session 6 (every commit authored and committed by you; no Claude trailers), and
   `main` now equals `session-3`. See the Session 6 section for the backup, the verification and what to do with
   your local copy. Helpers `scripts/_s3_monitor.py` / `_s4_monitor.py` are local only (git-excluded).

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
