# RQ5 statistics: shared-weight sleep vs parameter isolation

Unpaired tests (isolation builds fresh networks, so seeds aren't paired). Welch = two-sided Welch t-test; perm = exact two-sided permutation test (min attainable p with 3 vs 3 is 0.10); CI = 95% bootstrap CI of (sleep − isolation). FWT row: one-sample t-test of sleep's FWT against 0 (isolation's from-scratch curves are the reference, FWT_isolation ≡ 0).


## CNN trunk, fetch3 (`runs/continual`): sleep n=3, isolation n=3

| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |
|---|---|---|---|---|---|---|
| ACC | 0.978 ± 0.001 | 0.979 ± 0.001 | -0.001 | 0.22 | 0.20 | [-0.002, +0.000] |
| ACC / 100k params | 0.503 ± 0.000 | 0.204 ± 0.000 | +0.299 | 2.56e-08 | 0.10 | [+0.299, +0.299] |
| FWT (sleep vs 0) | -0.017 ± 0.141 | 0 (reference) | -0.017 | 0.854 (1-sample) | - | - |

per-seed ACC: sleep s1=0.978, s2=0.977, s3=0.978; isolation s1=0.979, s2=0.978, s3=0.979

## SNN trunk, fetch3 (`runs/continual_snn`): sleep n=3, isolation n=3

| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |
|---|---|---|---|---|---|---|
| ACC | 0.889 ± 0.003 | 0.833 ± 0.073 | +0.056 | 0.318 | 0.40 | [-0.014, +0.130] |
| ACC / 100k params | 0.457 ± 0.002 | 0.173 ± 0.015 | +0.283 | 0.000849 | 0.10 | [+0.269, +0.298] |
| FWT (sleep vs 0) | 0.199 ± 0.079 | 0 (reference) | +0.199 | 0.0489 (1-sample) | - | - |

per-seed ACC: sleep s1=0.890, s2=0.891, s3=0.885; isolation s1=0.838, s2=0.757, s3=0.904

## Full hybrid (SNN + Transformer), fetch3 (`runs/continual_hybrid`): sleep n=3, isolation n=3

| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |
|---|---|---|---|---|---|---|
| ACC | 0.873 ± 0.022 | 0.832 ± 0.037 | +0.041 | 0.194 | 0.20 | [+0.003, +0.083] |
| ACC / 100k params | 0.266 ± 0.007 | 0.095 ± 0.004 | +0.172 | 1.53e-05 | 0.10 | [+0.165, +0.179] |
| FWT (sleep vs 0) | 0.423 ± 0.244 | 0 (reference) | +0.423 | 0.095 (1-sample) | - | - |

per-seed ACC: sleep s1=0.897, s2=0.854, s3=0.868; isolation s1=0.790, s2=0.848, s3=0.859

## CNN trunk, fetch5 (`runs/continual5_cnn`): sleep n=3, isolation n=3

| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |
|---|---|---|---|---|---|---|
| ACC | 0.928 ± 0.065 | 0.976 ± 0.000 | -0.047 | 0.334 | 0.10 | [-0.122, -0.001] |
| ACC / 100k params | 0.406 ± 0.029 | 0.122 ± 0.000 | +0.284 | 0.00334 | 0.10 | [+0.251, +0.304] |
| FWT (sleep vs 0) | -0.525 ± 0.722 | 0 (reference) | -0.525 | 0.335 (1-sample) | - | - |

per-seed ACC: sleep s1=0.957, s2=0.974, s3=0.854; isolation s1=0.976, s2=0.975, s3=0.976

## SNN trunk, fetch5 (150k/task) (`runs/continual5_snn`): sleep n=2, isolation n=2

| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |
|---|---|---|---|---|---|---|
| ACC | 0.778 ± 0.007 | 0.475 ± 0.023 | +0.303 | 0.0227 | 0.33 | [+0.281, +0.324] |
| ACC / 100k params | 0.340 ± 0.003 | 0.059 ± 0.003 | +0.280 | 0.000117 | 0.33 | [+0.276, +0.285] |
| FWT (sleep vs 0) | 0.335 ± 0.072 | 0 (reference) | +0.335 | 0.0962 (1-sample) | - | - |

per-seed ACC: sleep s1=0.773, s2=0.783; isolation s1=0.458, s2=0.492
