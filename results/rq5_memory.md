# RQ5, memory-fair: ACC vs total memory (parameters + replay buffer)

Total memory = fp32 parameters + stored replay states at the measured minimum size (188 B/state for CNN/SNN trunks, 632 B/state for the hybrid's 4-frame windows). 'as implemented' doubles the buffer (the code caches a concatenated copy). Sleep's transient training peak adds a <=5000-state current-task reservoir + one network snapshot. Buffer counted = buffer_per_task x tasks (what's needed to keep learning). ACC/MB = ACC per MB of total memory.


## fetch3, CNN trunk

| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch3_naive @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.555 ± 0.013 | 0.714 |
| fetch3_ewc_lam10 @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.816 ± 0.120 | 1.050 |
| fetch3_ewc_lam100 @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.674 ± 0.030 | 0.867 |
| fetch3_ewc_lam1000 @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.545 ± 0.006 | 0.701 |
| fetch3_sleep_buf50 @150k | 3 | 150k | 50 | 0.78 | 0.03 | **0.81** | 0.83 | 2.52 | 0.846 ± 0.086 | 1.050 |
| fetch3_replay_buf50 @150k | 3 | 150k | 50 | 0.78 | 0.03 | **0.81** | 0.83 | - | 0.885 ± 0.081 | 1.099 |
| fetch3_replay_buf200 @150k | 3 | 150k | 200 | 0.78 | 0.11 | **0.89** | 1.00 | - | 0.946 ± 0.035 | 1.063 |
| fetch3_sleep_buf200 @150k | 3 | 150k | 200 | 0.78 | 0.11 | **0.89** | 1.00 | 2.61 | 0.964 ± 0.009 | 1.083 |
| fetch3_isolation @150k | 3 | 150k | 0 | 1.92 | 0.00 | **1.92** | 1.92 | - | 0.979 ± 0.001 | 0.510 |
| fetch3_replay @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | - | 0.970 ± 0.009 | 0.270 |
| fetch3_sleep @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.31 | 0.978 ± 0.001 | 0.272 |

## fetch3, SNN trunk

| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch3_naive @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.550 ± 0.130 | 0.707 |
| fetch3_ewc_lam100 @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.716 ± 0.199 | 0.920 |
| fetch3_naive_wakestdp @150k | 3 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | - | 0.270 ± 0.027 | 0.347 |
| fetch3_sleep_dream @150k | 1 | 150k | 0 | 0.78 | 0.00 | **0.78** | 0.78 | 2.50 | 0.204 | 0.262 |
| fetch3_replay_buf200 @150k | 3 | 150k | 200 | 0.78 | 0.11 | **0.89** | 1.00 | - | 0.792 ± 0.183 | 0.889 |
| fetch3_sleep_buf200 @150k | 3 | 150k | 200 | 0.78 | 0.11 | **0.89** | 1.00 | 2.61 | 0.838 ± 0.016 | 0.940 |
| fetch3_sleep_buf2000 @150k | 3 | 150k | 2000 | 0.78 | 1.13 | **1.91** | 3.03 | 3.62 | 0.847 ± 0.083 | 0.444 |
| fetch3_replay_buf2000 @150k | 3 | 150k | 2000 | 0.78 | 1.13 | **1.91** | 3.03 | - | 0.828 ± 0.167 | 0.434 |
| fetch3_isolation @150k | 3 | 150k | 0 | 1.92 | 0.00 | **1.92** | 1.92 | - | 0.833 ± 0.073 | 0.433 |
| fetch3_sleep_homeo @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.845 ± 0.034 | 0.235 |
| fetch3_sleep_stdps0.003 @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.881 ± 0.028 | 0.245 |
| fetch3_sleep_stdps0.01 @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.866 ± 0.045 | 0.241 |
| fetch3_sleep_dfaeh @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.450 ± 0.026 | 0.125 |
| fetch3_replay @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | - | 0.814 ± 0.162 | 0.226 |
| fetch3_sleep @150k | 3 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.889 ± 0.003 | 0.247 |
| fetch3_sleep_stdp @150k | 1 | 150k | 5000 | 0.78 | 2.82 | **3.60** | 6.42 | 5.32 | 0.163 | 0.045 |

## fetch3, Hybrid trunk

| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch3_naive @400k | 3 | 400k | 0 | 1.31 | 0.00 | **1.31** | 1.31 | - | 0.452 ± 0.016 | 0.345 |
| fetch3_isolation @400k | 3 | 400k | 0 | 3.52 | 0.00 | **3.52** | 3.52 | - | 0.832 ± 0.037 | 0.236 |
| fetch3_sleep_whomeo @400k | 3 | 400k | 5000 | 1.31 | 9.48 | **10.79** | 20.27 | 15.26 | 0.840 ± 0.051 | 0.078 |
| fetch3_sleep_wtfs @400k | 1 | 400k | 5000 | 1.31 | 9.48 | **10.79** | 20.27 | 15.26 | 0.896 | 0.083 |
| fetch3_sleep @400k | 6 | 400k | 5000 | 1.31 | 9.48 | **10.79** | 20.27 | 15.26 | 0.887 ± 0.045 | 0.082 |
| fetch3_replay @400k | 6 | 400k | 5000 | 1.31 | 9.48 | **10.79** | 20.27 | - | 0.855 ± 0.041 | 0.079 |

## fetch5, CNN trunk

| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch5_naive @150k | 3 | 150k | 0 | 0.91 | 0.00 | **0.91** | 0.91 | - | 0.416 ± 0.098 | 0.454 |
| fetch5_isolation @150k | 3 | 150k | 0 | 3.20 | 0.00 | **3.20** | 3.20 | - | 0.976 ± 0.000 | 0.305 |
| fetch5_sleep @150k | 3 | 150k | 5000 | 0.91 | 4.70 | **5.61** | 10.31 | 7.47 | 0.928 ± 0.065 | 0.165 |
| fetch5_replay @150k | 3 | 150k | 5000 | 0.91 | 4.70 | **5.61** | 10.31 | - | 0.926 ± 0.049 | 0.165 |

## fetch5, SNN trunk

| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |
|---|---|---|---|---|---|---|---|---|---|---|
| fetch5_naive @150k | 2 | 150k | 0 | 0.92 | 0.00 | **0.92** | 0.92 | - | 0.405 ± 0.058 | 0.443 |
| fetch5_isolation @150k | 2 | 150k | 0 | 3.20 | 0.00 | **3.20** | 3.20 | - | 0.475 ± 0.023 | 0.148 |
| fetch5_isolation @450k | 3 | 450k | 0 | 3.20 | 0.00 | **3.20** | 3.20 | - | 0.871 ± 0.026 | 0.272 |
| fetch5_sleep @150k | 2 | 150k | 5000 | 0.92 | 4.70 | **5.62** | 10.32 | 7.47 | 0.778 ± 0.007 | 0.138 |
| fetch5_sleep @450k | 3 | 450k | 5000 | 0.92 | 4.70 | **5.62** | 10.32 | 7.47 | 0.915 ± 0.014 | 0.163 |
