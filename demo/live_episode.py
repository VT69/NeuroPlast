"""Run one episode of a trained demo agent locally on CPU (optional; the dashboard does not need it).

    python demo/live_episode.py --agent sleep --task 0            # text trace
    python demo/live_episode.py --agent naive --task 0 --window   # watch it in a pygame window
    python demo/live_episode.py --agent naive --task 1 --gif out.gif

Agents: the fetch3 CNN agents of seed 1 after the full 3-task sequence (runs/demo_ckpt/fetch3_<agent>_s1/final_agent.pt),
the same checkpoints as the dashboard GIFs. Tasks: 0 = fetch the red ball, 1 = green key, 2 = blue box. Actions are
sampled from the policy, as in the scored evaluation. One episode is an anecdote; the measured score is the accuracy
matrix printed at the end (100 evaluation episodes per cell).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402

from record_demo import SUITES, TARGET, banner, load_agent, make_env  # noqa: E402

ACTIONS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["naive", "sleep"], default="sleep")
    ap.add_argument("--task", type=int, choices=[0, 1, 2], default=0)
    ap.add_argument("--seed", type=int, default=12345, help="room seed (12345 = first room of the scored evaluation)")
    ap.add_argument("--policy-seed", type=int, default=0, help="torch seed for action sampling")
    ap.add_argument("--greedy", action="store_true", help="take the argmax action instead of sampling")
    ap.add_argument("--window", action="store_true", help="render in a pygame window (needs a display)")
    ap.add_argument("--gif", help="also write the episode to this GIF")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.policy_seed)
    agent, res = load_agent(os.path.join(ROOT, f"runs/demo_ckpt/fetch3_{args.agent}_s1"))
    mode = "human" if args.window else ("rgb_array" if args.gif else None)
    env = make_env(SUITES["fetch3"][args.task], render_mode=mode)
    obs, _ = env.reset(seed=args.seed)
    print(f"{args.agent} agent (after all 3 tasks), task {args.task}: fetch the {TARGET[args.task]}, room seed {args.seed}")

    frames, ret, steps, done = [], 0.0, 0, False
    head = f"{args.agent} agent | task {args.task}: fetch the {TARGET[args.task]}"
    while not done:
        if args.gif:
            frames.append(banner(env.render(), head, f"step {steps}"))
        a, *_ = agent.get_action_and_value(torch.as_tensor(obs[None, None]), torch.ones(1, 1, dtype=torch.bool),
                                           task=args.task, greedy=args.greedy)
        obs, r, term, trunc, _ = env.step(int(a))
        ret, steps, done = ret + r, steps + 1, term or trunc
        print(f"  step {steps:3d}  {ACTIONS[int(a)]:<8s}" + (f"  reward {r:.2f}" if r else ""))
    ok = ret > 0
    print(f"{'SUCCESS' if ok else 'FAIL'}: return {ret:.2f} in {steps} steps"
          + ("" if ok else " (wrong object picked up, or time ran out)"))
    if args.gif:
        frames += [banner(env.render(), head, f"{'SUCCESS' if ok else 'FAIL'} (return {ret:.2f}, {steps} steps)",
                          (120, 230, 120) if ok else (240, 110, 110))] * 6
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=180, loop=0)
        print(f"wrote {args.gif}")
    env.close()

    print("\nMeasured score of this agent (mean return over 100 evaluation episodes; row = after training task i):")
    for i, row in enumerate(res["R"]):
        print(f"  after task {i}: " + "  ".join(f"task {j}: {v:.3f}" for j, v in enumerate(row)))
    print(f"  ACC (mean of the last row) = {res['metrics']['ACC']:.3f}")


if __name__ == "__main__":
    main()
