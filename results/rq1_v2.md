# RQ1 v2: stabilised STDP + backprop (SNN encoder, PPO, DoorKey-6x6)

AUC = mean training return over the run (sample efficiency); frames_to_0_9 = first update where the 100-episode moving train return >= 0.9 (mean over seeds that reach it). solved = seeds whose final eval return > 0.5 (outcomes are bimodal on DoorKey-6x6). p = two-sided Welch t-test of AUC vs `bp` (no multiple-comparison correction).

| arm          |   seeds | solved   | eval_return   | AUC           |   p(AUC vs bp) | frames_to_0_9   |
|:-------------|--------:|:---------|:--------------|:--------------|---------------:|:----------------|
| bp           |       5 | 4/5      | 0.769 ± 0.419 | 0.408 ± 0.238 |          1     | 254k (1 never)  |
| frozen       |       3 | 2/3      | 0.769 ± 0.288 | 0.441 ± 0.302 |          0.882 | 225k (1 never)  |
| hebs0.03     |       3 | 1/3      | 0.333 ± 0.489 | 0.188 ± 0.304 |          0.352 | 217k (2 never)  |
| hebs0.3      |       3 | 0/3      | 0.047 ± 0.018 | 0.210 ± 0.160 |          0.212 | - (3 never)     |
| rand0.3      |       5 | 4/5      | 0.754 ± 0.421 | 0.404 ± 0.268 |          0.979 | 284k (1 never)  |
| tfs0.03      |       5 | 4/5      | 0.766 ± 0.413 | 0.487 ± 0.314 |          0.667 | 195k (1 never)  |
| tfs0.3       |       3 | 2/3      | 0.670 ± 0.428 | 0.397 ± 0.295 |          0.959 | 254k (1 never)  |
| tfs0.3_local |       3 | 0/3      | 0.240 ± 0.240 | 0.108 ± 0.119 |          0.056 | - (3 never)     |
