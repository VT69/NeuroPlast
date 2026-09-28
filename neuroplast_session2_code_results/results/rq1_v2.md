# RQ1 v2: stabilised STDP + backprop (SNN encoder, PPO, DoorKey-6x6)

AUC = mean training return over the run (sample efficiency); frames_to_0_9 = first update where the 100-episode moving train return >= 0.9 (mean over seeds that reach it). solved = seeds whose final eval return > 0.5 (outcomes are bimodal on DoorKey-6x6). p = two-sided Welch t-test of AUC vs `bp` (no multiple-comparison correction).

| arm            |   seeds | solved   | eval_return   | AUC           |   p(AUC vs bp) | frames_to_0_9   |
|:---------------|--------:|:---------|:--------------|:--------------|---------------:|:----------------|
| bp             |       5 | 4/5      | 0.769 ± 0.419 | 0.408 ± 0.238 |          1     | 254k (1 never)  |
| dfae           |       5 | 1/5      | 0.166 ± 0.305 | 0.082 ± 0.123 |          0.035 | - (5 never)     |
| dfae_homeo     |       3 | 3/3      | 0.953 ± 0.003 | 0.634 ± 0.027 |          0.102 | 213k (0 never)  |
| dfae_rands0.03 |       3 | 3/3      | 0.936 ± 0.016 | 0.576 ± 0.118 |          0.233 | 241k (0 never)  |
| dfae_tfs0.03   |       5 | 5/5      | 0.944 ± 0.009 | 0.544 ± 0.073 |          0.28  | 253k (0 never)  |
| dfae_tfs0.3    |       5 | 0/5      | 0.206 ± 0.167 | 0.166 ± 0.126 |          0.091 | - (5 never)     |
| frozen         |       3 | 2/3      | 0.769 ± 0.288 | 0.441 ± 0.302 |          0.882 | 225k (1 never)  |
| hebs0.03       |       3 | 1/3      | 0.333 ± 0.489 | 0.188 ± 0.304 |          0.352 | 217k (2 never)  |
| hebs0.3        |       3 | 0/3      | 0.047 ± 0.018 | 0.210 ± 0.160 |          0.212 | - (3 never)     |
| homeo          |       3 | 3/3      | 0.958 ± 0.009 | 0.668 ± 0.122 |          0.088 | 165k (0 never)  |
| rand0.3        |       5 | 4/5      | 0.754 ± 0.421 | 0.404 ± 0.268 |          0.979 | 284k (1 never)  |
| tfs0.03        |       5 | 4/5      | 0.766 ± 0.413 | 0.487 ± 0.314 |          0.667 | 195k (1 never)  |
| tfs0.3         |       3 | 2/3      | 0.670 ± 0.428 | 0.397 ± 0.295 |          0.959 | 254k (1 never)  |
| tfs0.3_local   |       3 | 0/3      | 0.240 ± 0.240 | 0.108 ± 0.119 |          0.056 | - (3 never)     |
