# Demo assets: fetch3 naive vs sleep (CNN trunk, seed 1)

Both agents were trained on the same 3-task sequence (FetchObj-0 -> 1 -> 2: fetch the red ball, then the green key, then the blue box; distractor pick-ups end the episode with 0). Every GIF shows the agent **after the whole sequence** (checkpoint `final_agent.pt` in `runs/demo_ckpt/`, i.e. the agent scored in the last row of the accuracy matrix), acting on one task. Actions are sampled from the policy, as in the scored evaluation. Each GIF has 3 episodes in the same rooms for both agents: the rooms of the first episode of the scored evaluation's envs 0-2 (seeds 12345, 13345, 14345).

What to compare: tasks 0 and 1 were learned earlier in the sequence, so forgetting would show there; task 2 was trained last. The last column is the measured score, the GIFs are 3 illustrative episodes.

Playback: 400 ms per frame, the last frame of each episode held 1.5 s; steps from 24 on play at 50 ms (labelled fast-forward), so a 256-step time-out doesn't take 100 s.

| GIF | agent | task | episodes (3) | agent's scored mean return on this task (100 eval episodes) |
|---|---|---|---|---|
| [fetch3_naive_task0.gif](fetch3_naive_task0.gif) | naive | 0: fetch the red ball | ✓ 0.99 (4 steps), ✗ 0.00 (256 steps), ✗ 0.00 (256 steps) | 0.126 |
| [fetch3_naive_task1.gif](fetch3_naive_task1.gif) | naive | 1: fetch the green key | ✓ 0.98 (5 steps), ✓ 0.57 (123 steps), ✗ 0.00 (256 steps) | 0.542 |
| [fetch3_naive_task2.gif](fetch3_naive_task2.gif) | naive | 2: fetch the blue box | ✓ 0.96 (10 steps), ✓ 0.98 (6 steps), ✓ 0.96 (11 steps) | 0.956 |
| [fetch3_sleep_task0.gif](fetch3_sleep_task0.gif) | sleep | 0: fetch the red ball | ✓ 0.99 (4 steps), ✓ 0.97 (8 steps), ✓ 0.93 (19 steps) | 0.978 |
| [fetch3_sleep_task1.gif](fetch3_sleep_task1.gif) | sleep | 1: fetch the green key | ✓ 0.98 (5 steps), ✓ 0.99 (3 steps), ✓ 0.95 (13 steps) | 0.979 |
| [fetch3_sleep_task2.gif](fetch3_sleep_task2.gif) | sleep | 2: fetch the blue box | ✓ 0.97 (9 steps), ✓ 0.97 (8 steps), ✓ 0.97 (9 steps) | 0.979 |

Regenerate: `python scripts/record_demo.py` (needs the two runs in `runs/demo_ckpt/`).
