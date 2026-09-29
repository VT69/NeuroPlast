"""Demo assets (session 4, block 4): GIFs of the fetch3 naive and sleep agents (CNN trunk, seed 1) on every
task AFTER the full 3-task sequence, from runs/demo_ckpt/<run>/final_agent.pt (the agent scored in R[-1]).

Same policy as the scored evaluation (actions sampled, not greedy). Each GIF shows EPISODES episodes in fixed rooms (the scored evaluation's
first rooms), identical for both agents, so the two GIFs of a task show the same rooms. Writes demo/assets/*.gif
and demo/assets/index.md.

    python scripts/record_demo.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.envs.continual import SUITES  # noqa: E402,F401  (registers FetchObj envs)
from neuroplast.envs.minigrid_env import make_env  # noqa: E402
from train import Config, build_agent  # noqa: E402

RUNS = {"naive": "runs/demo_ckpt/fetch3_naive_s1", "sleep": "runs/demo_ckpt/fetch3_sleep_s1"}
TARGET = {0: "red ball", 1: "green key", 2: "blue box"}
EPISODES, TORCH_SEED = 3, 0
# rooms = the first episode of eval envs 0..2 in train.evaluate (VecEnv seed 12345 -> env i reset with 12345 + 1000 i)
ENV_SEEDS = [12345 + 1000 * e for e in range(EPISODES)]
OUT = "demo/assets"


def load_agent(run_dir):
    r = json.load(open(os.path.join(run_dir, "results.json")))
    names = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in r["cfg"].items() if k in names})
    agent = build_agent(cfg)
    sd = torch.load(os.path.join(run_dir, "final_agent.pt"), map_location="cpu")
    agent.load_state_dict(sd["agent"] if "agent" in sd else sd)  # taskN/final.pt wraps {"agent", "cfg"}
    return agent.eval(), r


def banner(frame, line1, line2="", color=(230, 230, 230)):
    img = Image.fromarray(frame).resize((frame.shape[1] * 2, frame.shape[0] * 2), Image.NEAREST)
    out = Image.new("RGB", (img.width, img.height + 40), (20, 20, 20))
    out.paste(img, (0, 40))
    d = ImageDraw.Draw(out)
    d.text((8, 5), line1, fill=(230, 230, 230))
    d.text((8, 21), line2, fill=color)
    return out


@torch.no_grad()
def record(agent, method, task):
    env_id = SUITES["fetch3"][task]
    torch.manual_seed(TORCH_SEED)
    frames, outcomes = [], []
    for ep in range(EPISODES):
        env = make_env(env_id, render_mode="rgb_array")
        obs, _ = env.reset(seed=ENV_SEEDS[ep])
        ret, steps, done = 0.0, 0, False
        head = f"{method} agent (after all 3 tasks) | task {task}: fetch the {TARGET[task]}"
        while not done:
            frames.append(banner(env.render(), head, f"episode {ep + 1}/{EPISODES}, step {steps}"))
            o = torch.as_tensor(obs[None, None])
            m = torch.ones(1, 1, dtype=torch.bool)
            a, *_ = agent.get_action_and_value(o, m, task=task, greedy=False)
            obs, r, term, trunc, _ = env.step(int(a))
            ret, steps, done = ret + r, steps + 1, term or trunc
        ok = ret > 0
        end = f"episode {ep + 1}/{EPISODES}: {'SUCCESS' if ok else 'FAIL'} (return {ret:.2f}, {steps} steps)"
        frames += [banner(env.render(), head, end, (120, 230, 120) if ok else (240, 110, 110))] * 6  # hold
        outcomes.append((ok, ret, steps))
        env.close()
    path = os.path.join(OUT, f"fetch3_{method}_task{task}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return path, outcomes


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, R = [], {}
    for method, run_dir in RUNS.items():
        agent, res = load_agent(run_dir)
        R[method] = res["R"][-1]
        for task in range(3):
            path, outc = record(agent, method, task)
            rows.append((method, task, os.path.basename(path), outc))
            print(path, outc, flush=True)
    lines = ["# Demo assets: fetch3 naive vs sleep (CNN trunk, seed 1)\n",
             "Both agents were trained on the same 3-task sequence (FetchObj-0 -> 1 -> 2: fetch the red ball, then "
             "the green key, then the blue box; distractor pick-ups end the episode with 0). Every GIF shows the "
             "agent **after the whole sequence** (checkpoint `final_agent.pt` in `runs/demo_ckpt/`, i.e. the agent "
             "scored in the last row of the accuracy matrix), acting on one task. Actions are sampled from the policy, "
             "as in the scored evaluation. Each GIF has 3 episodes in the same rooms for both agents: the rooms of the "
             f"first episode of the scored evaluation's envs 0-2 (seeds {', '.join(map(str, ENV_SEEDS))}).\n",
             "What to compare: tasks 0 and 1 were learned earlier in the sequence, so forgetting would show there; "
             "task 2 was trained last. The last column is the measured score, the GIFs are 3 illustrative episodes.\n",
             "| GIF | agent | task | episodes (3) | agent's scored mean return on this task (100 eval episodes) |",
             "|---|---|---|---|---|"]
    for method, task, f, outc in rows:
        eps = ", ".join(f"{'✓' if ok else '✗'} {ret:.2f} ({n} steps)" for ok, ret, n in outc)
        lines.append(f"| [{f}]({f}) | {method} | {task}: fetch the {TARGET[task]} | {eps} | {R[method][task]:.3f} |")
    lines.append("\nRegenerate: `python scripts/record_demo.py` (needs the two runs in `runs/demo_ckpt/`).")
    with open(os.path.join(OUT, "index.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
