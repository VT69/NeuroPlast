# RQ4: robustness per unit of compute

Joins runs/rq1 (DoorKey-5x5), runs/rq1_v2 (DoorKey-6x6), the continual sweeps and results/compute.json (CPU process time per PPO step, batch 256, 1 thread). Inference energy per decision is in results/rq3_doorkey6.md (SNN: 15-200 nJ depending on sparsity, CNN: 1653 nJ dense / 130-940 nJ event-driven). 'continual ACC per train-second' = final continual ACC / CPU-seconds per training step (higher = more robust per unit of training compute).

| variant                                   | solve rate (5x5+6x6 seeds)   | AUC 5x5       | AUC 6x6       | continual ACC (fetch3)           |   train ms/step (CPU) |   params | continual ACC per train-second   |
|:------------------------------------------|:-----------------------------|:--------------|:--------------|:---------------------------------|----------------------:|---------:|:---------------------------------|
| CNN (reference)                           | 3/3                          | 0.744 ± 0.140 | -             | 0.978 ± 0.001 [sleep]            |                    16 |   159864 | 60.2                             |
| A: SNN, surrogate-gradient BPTT           | 7/8                          | 0.664 ± 0.161 | 0.408 ± 0.238 | 0.889 ± 0.003 [sleep]            |                    69 |   160104 | 12.9                             |
| A + Transformer memory (full hybrid)      | 1/1                          | 0.693 ± 0.000 | -             | -                                |                    70 |   293352 | -                                |
| A + stabilised 3-factor STDP (a=0.03/0.3) | 4/5                          | -             | 0.487 ± 0.314 | 0.881 ± 0.028 [sleep_stdps0.003] |                   132 |   160104 | 6.7                              |
| B: local-only SNN encoder                 | 0/6                          | 0.169 ± 0.066 | 0.108 ± 0.119 | -                                |                   161 |   160104 | -                                |
| C: DFA SNN, no backprop                   | 2/3                          | 0.361 ± 0.197 | -             | -                                |                    78 |   144104 | -                                |
