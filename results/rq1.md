# RQ1: hybrid STDP + backprop (SNN encoder, PPO, DoorKey-5x5)

AUC = mean training return over the run (sample efficiency); frames_to_0_9 = first update where the 100-episode moving train return >= 0.9 (mean over seeds that reach it). solved = seeds whose final eval return > 0.5 (outcomes are bimodal on DoorKey-6x6). p = two-sided Welch t-test of AUC vs `bp` (no multiple-comparison correction).

| arm       |   seeds | solved   | eval_return   | AUC           |   p(AUC vs bp) | frames_to_0_9   |
|:----------|--------:|:---------|:--------------|:--------------|---------------:|:----------------|
| bp        |       3 | 3/3      | 0.959 ± 0.000 | 0.664 ± 0.161 |          1     | 122k (0 never)  |
| bp_linear |       3 | 2/3      | 0.668 ± 0.504 | 0.464 ± 0.381 |          0.47  | 118k (1 never)  |
| cnn       |       3 | 3/3      | 0.964 ± 0.000 | 0.744 ± 0.140 |          0.551 | 80k (0 never)   |
| dfa       |       3 | 2/3      | 0.773 ± 0.274 | 0.361 ± 0.197 |          0.111 | 245k (1 never)  |
| frozen    |       3 | 3/3      | 0.957 ± 0.003 | 0.751 ± 0.120 |          0.499 | 87k (0 never)   |
| heb0.1    |       3 | 0/3      | 0.188 ± 0.109 | 0.135 ± 0.067 |          0.018 | - (3 never)     |
| heb1      |       3 | 0/3      | 0.231 ± 0.057 | 0.148 ± 0.044 |          0.024 | - (3 never)     |
| rand1     |       3 | 3/3      | 0.957 ± 0.003 | 0.742 ± 0.045 |          0.494 | 99k (0 never)   |
| tf0.1     |       3 | 3/3      | 0.763 ± 0.194 | 0.635 ± 0.150 |          0.831 | 120k (1 never)  |
| tf1       |       3 | 0/3      | 0.219 ± 0.066 | 0.162 ± 0.118 |          0.015 | - (3 never)     |
| tf1_local |       3 | 0/3      | 0.303 ± 0.083 | 0.169 ± 0.066 |          0.021 | - (3 never)     |
