# Session 4 pre-registered tests

Planned tests are labelled; everything else is descriptive. Welch = two-sided Welch t-test; perm = exact two-sided permutation test; CI = bootstrap 95% CI.

## Block 1: full hybrid, sleep vs replay with replayed samples matched exactly (RQ2)

Seeds analysed: [1, 2, 3, 4, 5, 6, 7, 8] (stopping rule: 8 vs 8 if seeds 7-8 finished, else 1-6).

| seed | replay: replayed states | sleep_matched: replayed states | match | replay ACC | sleep_matched ACC | unmatched sleep ACC (1.43M states) |
|---|---|---|---|---|---|---|
| 1 | 919,680 | 919,680 | exact | 0.844 | 0.901 | 0.897 |
| 2 | 946,432 | 946,432 | exact | 0.841 | 0.846 | 0.854 |
| 3 | 1,475,200 | 1,475,200 | exact | 0.817 | 0.828 | 0.868 |
| 4 | 908,672 | 908,672 | exact | 0.918 | 0.934 | 0.946 |
| 5 | 1,344,128 | 1,344,128 | exact | 0.818 | 0.910 | 0.926 |
| 6 | 1,140,992 | 1,140,992 | exact | 0.890 | 0.819 | 0.829 |
| 7 | 1,217,408 | 1,217,408 | exact | 0.876 | 0.886 | - |
| 8 | 1,108,096 | 1,108,096 | exact | 0.896 | 0.912 | - |

| metric | sleep_matched | replay | diff | Welch p | perm p | 95% bootstrap CI |
|---|---|---|---|---|---|---|
| ACC **(planned test)** | 0.879 ± 0.043 | 0.862 ± 0.038 | +0.017 | 0.415 | 0.402 | [-0.020, +0.053] |
| FORGET (descriptive) | 0.020 ± 0.011 | 0.019 ± 0.017 | +0.001 | 0.881 | 0.876 | [-0.012, +0.014] |
| BWT (descriptive) | -0.017 ± 0.015 | -0.016 ± 0.015 | -0.001 | 0.922 | 0.923 | [-0.014, +0.013] |

Descriptive: matched vs unmatched sleep, same seeds [1, 2, 3, 4, 5, 6]: 0.873 vs 0.887 (paired diff -0.014, paired t p=0.069).

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
| probe_S11_cnn_fs12_6M_s1 | 0.485 | 0.478 | 9.6 |
| probe_S11_cnn_fs12_s1 | 0.465 | 0.458 | 10.2 |
| probe_S11_cnn_fs1_s1 | 0.430 | 0.424 | 8.6 |
| probe_S13_cnn_fs12_s1 | 0.460 | 0.454 | 12.3 |
| probe_S13_cnn_fs1_s1 | 0.485 | 0.479 | 11.4 |
