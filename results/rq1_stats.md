# RQ1 statistics (pre-registered, session 3): DoorKey-6x6, 400k frames

solved = final eval return > 0.5 (outcomes are bimodal). Fisher = two-sided Fisher's exact test on solve counts; Welch = two-sided Welch t-test; Holm = Holm-adjusted p across the four primary contrasts (computed separately for each statistic). frames_to_0.9 is over solving seeds only.

## Arms

| arm | n | solved | solve rate | AUC | final eval | frames to 0.9 (solvers) |
|---|---|---|---|---|---|---|
| bp | 10 | 8/10 | 0.80 | 0.416 ± 0.232 | 0.766 ± 0.398 | 251k |
| homeo | 10 | 10/10 | 1.00 | 0.616 ± 0.113 | 0.917 ± 0.137 | 179k |
| tfs0.03 | 10 | 8/10 | 0.80 | 0.515 ± 0.300 | 0.765 ± 0.395 | 178k |
| rands0.03 | 10 | 8/10 | 0.80 | 0.482 ± 0.323 | 0.742 ± 0.400 | 160k |
| *context arms (not part of the pre-registered test)* | | | | | | |
| frozen | 3 | 2/3 | 0.67 | 0.441 ± 0.302 | 0.769 ± 0.288 | 225k |
| rand0.3 | 5 | 4/5 | 0.80 | 0.404 ± 0.268 | 0.754 ± 0.421 | 284k |
| tfs0.3 | 3 | 2/3 | 0.67 | 0.397 ± 0.295 | 0.670 ± 0.428 | 254k |
| hebs0.03 | 3 | 1/3 | 0.33 | 0.188 ± 0.304 | 0.333 ± 0.489 | 217k |
| dfae | 10 | 2/10 | 0.20 | 0.076 ± 0.101 | 0.188 ± 0.254 | - |
| dfae_homeo | 10 | 7/10 | 0.70 | 0.402 ± 0.307 | 0.645 ± 0.443 | 214k |
| dfae_tfs0.03 | 5 | 5/5 | 1.00 | 0.544 ± 0.073 | 0.944 ± 0.009 | 253k |
| dfae_rands0.03 | 3 | 3/3 | 1.00 | 0.576 ± 0.118 | 0.936 ± 0.016 | 241k |

## Pre-registered contrasts

| question | contrast | solved | Fisher p (Holm) | ΔAUC | Welch AUC p (Holm) | Welch eval p (Holm) |
|---|---|---|---|---|---|---|
| (a) does STDP add beyond homeostasis? | tfs0.03 vs homeo | 8/10 vs 10/10 | 0.474 (1.000) | -0.101 | 0.338 (0.724) | 0.273 (0.864) |
| (b) does STDP add beyond matched random perturbation? | tfs0.03 vs rands0.03 | 8/10 vs 8/10 | 1.000 (1.000) | +0.033 | 0.818 (0.818) | 0.899 (0.899) |
| (c) does homeostasis help backprop (not only DFA)? | homeo vs bp | 10/10 vs 8/10 | 0.474 (1.000) | +0.200 | 0.029 (0.117) | 0.280 (0.864) |
| (d) does generic perturbation add beyond homeostasis? | rands0.03 vs homeo | 8/10 vs 10/10 | 0.474 (1.000) | -0.134 | 0.241 (0.724) | 0.216 (0.864) |
