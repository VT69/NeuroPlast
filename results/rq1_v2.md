# RQ1 v2: stabilised STDP + backprop (SNN encoder, PPO, DoorKey-6x6)

AUC = mean training return over the run (sample efficiency); frames_to_0_9 = first update where the 100-episode moving train return >= 0.9 (mean over seeds that reach it). solved = seeds whose final eval return > 0.5 (outcomes are bimodal on DoorKey-6x6). p = two-sided Welch t-test of AUC vs `bp` (no multiple-comparison correction).

| arm            |   seeds | solved   | eval_return   | AUC           |   p(AUC vs bp) | frames_to_0_9   |
|:---------------|--------:|:---------|:--------------|:--------------|---------------:|:----------------|
| bp             |      10 | 8/10     | 0.766 ± 0.398 | 0.416 ± 0.232 |          1     | 251k (2 never)  |
| dfae           |       5 | 1/5      | 0.166 ± 0.305 | 0.082 ± 0.123 |          0.003 | - (5 never)     |
| dfae_homeo     |       3 | 3/3      | 0.953 ± 0.003 | 0.634 ± 0.027 |          0.017 | 213k (0 never)  |
| dfae_rands0.03 |       3 | 3/3      | 0.936 ± 0.016 | 0.576 ± 0.118 |          0.154 | 241k (0 never)  |
| dfae_tfs0.03   |       5 | 5/5      | 0.944 ± 0.009 | 0.544 ± 0.073 |          0.139 | 253k (0 never)  |
| dfae_tfs0.3    |       5 | 0/5      | 0.206 ± 0.167 | 0.166 ± 0.126 |          0.018 | - (5 never)     |
| frozen         |       3 | 2/3      | 0.769 ± 0.288 | 0.441 ± 0.302 |          0.907 | 225k (1 never)  |
| hebs0.03       |       3 | 1/3      | 0.333 ± 0.489 | 0.188 ± 0.304 |          0.323 | 217k (2 never)  |
| hebs0.3        |       3 | 0/3      | 0.047 ± 0.018 | 0.210 ± 0.160 |          0.143 | - (3 never)     |
| homeo          |      10 | 10/10    | 0.917 ± 0.137 | 0.616 ± 0.113 |          0.029 | 179k (0 never)  |
| rand0.3        |       5 | 4/5      | 0.754 ± 0.421 | 0.404 ± 0.268 |          0.931 | 284k (1 never)  |
| rands0.03      |      10 | 8/10     | 0.742 ± 0.400 | 0.482 ± 0.323 |          0.607 | 160k (3 never)  |
| tfs0.03        |      10 | 8/10     | 0.765 ± 0.395 | 0.515 ± 0.300 |          0.422 | 178k (2 never)  |
| tfs0.3         |       3 | 2/3      | 0.670 ± 0.428 | 0.397 ± 0.295 |          0.924 | 254k (1 never)  |
| tfs0.3_local   |       3 | 0/3      | 0.240 ± 0.240 | 0.108 ± 0.119 |          0.018 | - (3 never)     |
