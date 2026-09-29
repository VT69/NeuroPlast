# Session 4 pre-registered tests

Planned tests are labelled; everything else is descriptive. Welch = two-sided Welch t-test; perm = exact two-sided permutation test; CI = bootstrap 95% CI.

## Block 1: full hybrid, sleep vs replay with replayed samples matched exactly (RQ2)

(not enough runs yet)


## Block 2: DoorKey-6x6, DFA on the SNN encoder with vs without homeostasis

| arm | n | solved | AUC | final eval | frames to 0.9 (solvers) |
|---|---|---|---|---|---|
| DFA + homeostasis | 10 | 7/10 | 0.402 ± 0.307 | 0.645 ± 0.443 | 214k |
| DFA | 10 | 2/10 | 0.076 ± 0.101 | 0.188 ± 0.254 | - |

| planned test | p | Holm (2 tests) |
|---|---|---|
| solve rate, Fisher exact | 0.0698 | 0.0698 |
| AUC, Welch (diff +0.326) | 8.69e-03 | 1.74e-02 |

per-seed eval: DFA+homeo s1=0.95, s2=0.96, s3=0.95, s4=0.00, s5=0.96, s6=0.70, s7=0.93, s8=0.96, s9=0.04, s10=0.00; DFA s1=0.02, s2=0.02, s3=0.01, s4=0.07, s5=0.71, s6=0.12, s7=0.00, s8=0.35, s9=0.53, s10=0.04

## Block 3: memory benchmark

Probe (sizing only, 1 seed, lr 3e-4, 2M frames):

| run | eval success | eval return | mean episode length |
|---|---|---|---|
| probe_S11_cnn_fs12_s1 | 0.465 | 0.458 | 10.2 |
| probe_S11_cnn_fs1_s1 | 0.430 | 0.424 | 8.6 |
| probe_S13_cnn_fs12_s1 | 0.460 | 0.454 | 12.3 |
| probe_S13_cnn_fs1_s1 | 0.485 | 0.479 | 11.4 |
