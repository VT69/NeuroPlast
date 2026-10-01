# Phase C secondary analyses (exploratory; not pre-registered)

## C3 (CNN fetch3, ~14k replayed samples)

- Runs where a learned old task collapsed below 0.5: sleep 4/8, replay 0/8 (Fisher exact p = 0.0769).
- Task 0 at the end: sleep 0.08, 0.20, 0.49, 0.89, 0.93, 0.03, 0.78, 0.90; replay 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98.
- Last task (no forgetting possible yet; measures plasticity): sleep 0.973 ± 0.008, replay 0.889 ± 0.055; difference +0.084 [+0.038, +0.130], Welch p = 0.00323.
- Replayed samples: sleep 14,080, replay 13,986 (range 13,866-14,016).

## C2 (SNN fetch3), paired by seed

- ACC: LwF-int8 − replay@1380 = -0.016 [-0.029, -0.004], paired t p = 0.0196 (n = 8 seeds).
- FORGET: LwF-int8 − replay@1380 = +0.017 [+0.010, +0.023], paired t p = 0.000379 (n = 8 seeds).
- Seed-level correlation of ACC across arms: r = 0.98. Seed 204 fails task 0 in both arms (0.47 right after its own training), so the seed, not the method, decides it.

## C1 (CNN fetch5)

- Spread of final ACC: LwF-int8 SD 0.018 (min 0.911), replay@973 SD 0.052 (min 0.787); Brown-Forsythe p = 0.126.
- Paired by seed: LwF-int8 − replay@973 = +0.024 [-0.005, +0.053], paired t p = 0.0901.
- LwF-int8 is above replay@973 on 8/10 seeds.
