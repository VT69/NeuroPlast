import torch

from neuroplast.envs.minigrid_env import VecEnv
from neuroplast.models.agent import Agent, FeatureCache


def test_feature_cache_matches_full_window_encode():
    torch.manual_seed(0)
    for enc in ("cnn", "snn"):
        agent = Agent(encoder=enc, memory=True, window=4).eval()
        envs = VecEnv("MiniGrid-DoorKey-5x5-v0", 4, seed=0, history=4)
        cache = FeatureCache(4, 4, agent.feat_dim)
        o, m = envs.reset()
        for _ in range(40):  # long enough to span episode resets? use random actions
            ot, mt = torch.as_tensor(o), torch.as_tensor(m)
            with torch.no_grad():
                l_full, v_full = agent(ot, mt)
                l_c, v_c = agent(ot, mt, past_feats=cache.f)
            cache.write_newest(agent.last_frame_feat)
            assert torch.allclose(l_full, l_c, atol=1e-5) and torch.allclose(v_full, v_c, atol=1e-5)
            a = torch.randint(0, 7, (4,))
            o, m, _, d, _ = envs.step(a.numpy())
            cache.advance(d)


def test_multihead_task_routing():
    agent = Agent(n_tasks=3, multihead=True).eval()
    obs = torch.zeros(6, 1, 7, 7, 3, dtype=torch.uint8)
    mask = torch.ones(6, 1, dtype=torch.bool)
    task = torch.tensor([0, 1, 2, 0, 1, 2])
    logits, _ = agent(obs, mask, task)
    for t in range(3):
        l_t, _ = agent(obs[:1], mask[:1], t)
        assert torch.allclose(logits[t], l_t[0], atol=1e-6)
