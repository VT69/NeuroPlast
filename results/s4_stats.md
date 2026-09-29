# Session 4 pre-registered tests

Planned tests are labelled; everything else is descriptive. Welch = two-sided Welch t-test; perm = exact two-sided permutation test; CI = bootstrap 95% CI.

## Block 1: full hybrid, sleep vs replay with replayed samples matched exactly (RQ2)

(not enough runs yet)


## Block 2: DoorKey-6x6, DFA on the SNN encoder with vs without homeostasis

| arm | n | solved | AUC | final eval | frames to 0.9 (solvers) |
|---|---|---|---|---|---|
| DFA + homeostasis | 3 | 3/3 | 0.634 ± 0.027 | 0.953 ± 0.003 | 213k |
| DFA | 5 | 1/5 | 0.082 ± 0.123 | 0.166 ± 0.305 | - |

| planned test | p | Holm (2 tests) |
|---|---|---|
| solve rate, Fisher exact | 0.1429 | 0.1429 |
| AUC, Welch (diff +0.551) | 3.20e-04 | 6.39e-04 |

per-seed eval: DFA+homeo s1=0.95, s2=0.96, s3=0.95; DFA s1=0.02, s2=0.02, s3=0.01, s4=0.07, s5=0.71

## Block 3: memory benchmark

