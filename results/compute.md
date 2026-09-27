# RQ4 compute per variant

CPU process time, 1 thread; encoder energy uses the RQ3 accounting (CNN: dense MACs x 4.6 pJ; SNN: SynOps x 0.9 pJ) at initialisation (trained SNNs are sparser, see results/rq3*.md). Memory agents encode one new frame per decision (cached features for the rest of the window); their train step re-encodes only the newest frame too.

| variant                                     |   params |   enc_ops_k_per_decision |   enc_energy_nj_per_decision |   rollout_ms_per_16 |   train_step_ms_per_256 |
|:--------------------------------------------|---------:|-------------------------:|-----------------------------:|--------------------:|------------------------:|
| CNN                                         |   159864 |                      156 |                     1.65e+03 |                1.15 |                    16.2 |
| A: SNN e2e (surrogate BPTT)                 |   160104 |                      160 |                   144        |                4.34 |                    68.8 |
| A+mem: SNN + Transformer (window 4)         |   293352 |                      150 |                   135        |                5.32 |                    69.9 |
| A+STDP: hybrid rule (stabilised, alpha 0.3) |   160104 |                      265 |                   238        |                3.72 |                   132   |
| B: local-only SNN encoder + backprop heads  |   160104 |                      158 |                   143        |                3.49 |                   161   |
| C: DFA SNN + linear readouts (no backprop)  |   144104 |                      340 |                   306        |                4.82 |                    78.4 |
