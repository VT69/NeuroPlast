"""Session 4 additions: exact sleep replay budget, GRU window memory, final-agent checkpoint."""
import json
import os
from dataclasses import replace

import torch

from neuroplast.envs.minigrid_env import VecEnv
from neuroplast.models.agent import Agent, FeatureCache
from neuroplast.sleep.sleep import Sleep
from train import load_config


def test_sleep_plan_matches_hybrid_schedule():
    cfg = replace(load_config("configs/cl_hybrid_v2.yaml", []), n_tasks=3)
    # 400k frames / (16 envs x 128 steps) = 195 updates -> 7 periodic + 1 end-of-task phase, tasks 1-2
    s = Sleep(replay_budget=1122560)
    plan = s.plan_steps(cfg)
    assert len(plan) == 16 and sum(plan) * 128 == 1122560 and max(plan) - min(plan) <= 1
    # the unmatched default: 16 x 700 x 128 = 1,433,600 (session 2/3 hybrid sleep runs)
    assert 16 * 700 * 128 == 1433600


def test_sleep_budget_is_consumed_exactly(tmp_path):
    from continual import run
    cfg = load_config("configs/cl_cnn.yaml", ["total_frames=8192", "num_envs=4", "num_steps=128"])
    # 8192 / 512 = 16 updates per task; period 5 -> 3 periodic + 1 end-of-task phase per task, 2 tasks
    res = run("fetch3", "sleep", cfg, 1, str(tmp_path), eval_episodes=2,
              method_kwargs=dict(period=5, replay_budget=128 * 37, buffer_per_task=64))
    assert res["replay_samples"] == 128 * 37 and res["sleep_phases"] == 8
    assert os.path.exists(tmp_path / "fetch3_sleep_s1" / "final_agent.pt")


def test_gru_memory_uses_history_and_matches_cache():
    torch.manual_seed(0)
    agent = Agent(encoder="cnn", memory=True, window=4, mem_kwargs={"kind": "gru"}).eval()
    assert agent.memory.__class__.__name__ == "GRUMemory"
    envs = VecEnv("MiniGrid-DoorKey-5x5-v0", 4, seed=0, history=4)
    cache = FeatureCache(4, 4, agent.feat_dim)
    o, m = envs.reset()
    for _ in range(30):
        ot, mt = torch.as_tensor(o), torch.as_tensor(m)
        with torch.no_grad():
            l_full, _ = agent(ot, mt)
            l_c, _ = agent(ot, mt, past_feats=cache.f)
        cache.write_newest(agent.last_frame_feat)
        assert torch.allclose(l_full, l_c, atol=1e-5)
        o, m, _, d, _ = envs.step(torch.randint(0, 7, (4,)).numpy())
        cache.advance(d)
    # output depends on older (valid) frames, not only the newest
    ot, mt = torch.as_tensor(o).clone(), torch.as_tensor(m).clone()
    mt[:] = True
    with torch.no_grad():
        a, _ = agent(ot, mt)
        ot2 = ot.clone()
        ot2[:, 0] = torch.randint(0, 6, ot2[:, 0].shape, dtype=ot2.dtype)
        b, _ = agent(ot2, mt)
    assert not torch.allclose(a, b)
