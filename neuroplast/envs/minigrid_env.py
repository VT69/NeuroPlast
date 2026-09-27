"""MiniGrid environment helpers.

Observations are MiniGrid's 7x7x3 symbolic egocentric view (object idx, color
idx, state). We keep them as uint8 in the rollout buffers and one-hot them on
the fly in torch (`obs_to_onehot`) -> a (20, 7, 7) binary tensor. Binary one-hot
input is convenient for the SNN: it is already a spike pattern, so "direct
coding" (present the same binary frame at every SNN timestep) needs no extra
encoding step, and the CNN sees exactly the same tensor.

`VecEnv` is a tiny synchronous vector env with same-step autoreset and a
per-env frame history (for the Transformer working memory).
"""
from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers envs)
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper

N_OBJ, N_COLOR, N_STATE = 11, 6, 3
OBS_CHANNELS = N_OBJ + N_COLOR + N_STATE  # 20
N_ACTIONS = 7


def obs_to_onehot(obs: torch.Tensor) -> torch.Tensor:
    """(..., 7, 7, 3) uint8/long -> (..., 20, 7, 7) float one-hot."""
    obs = obs.long()
    o = F.one_hot(obs[..., 0].clamp(0, N_OBJ - 1), N_OBJ)
    c = F.one_hot(obs[..., 1].clamp(0, N_COLOR - 1), N_COLOR)
    s = F.one_hot(obs[..., 2].clamp(0, N_STATE - 1), N_STATE)
    x = torch.cat([o, c, s], dim=-1).float()  # (..., 7, 7, 20)
    return x.movedim(-1, -3)


def make_env(env_id: str, seed: int | None = None, **kwargs):
    if env_id in CUSTOM_ENVS:
        env = CUSTOM_ENVS[env_id](**kwargs)
    else:
        env = gym.make(env_id, **kwargs)
    env = ImgObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


class VecEnv:
    """Synchronous vector env with same-step autoreset and frame history.

    `history` > 1 keeps the last `history` observations per env (oldest first);
    `mask` marks which slots hold real frames (False = padding before the
    episode start). Episode stats are returned from `step` for logging.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int = 0, history: int = 1, env_kwargs=None):
        self.envs = [make_env(env_id, seed + i, **(env_kwargs or {})) for i in range(num_envs)]
        self.num_envs = num_envs
        self.history = history
        self.env_id = env_id
        self._seed = seed
        self.obs_hist = np.zeros((num_envs, history, 7, 7, 3), dtype=np.uint8)
        self.mask = np.zeros((num_envs, history), dtype=bool)
        self.ep_ret = np.zeros(num_envs, dtype=np.float32)
        self.ep_len = np.zeros(num_envs, dtype=np.int64)

    def _push(self, i, obs, reset):
        if reset:
            self.obs_hist[i] = 0
            self.mask[i] = False
        else:
            self.obs_hist[i, :-1] = self.obs_hist[i, 1:]
            self.mask[i, :-1] = self.mask[i, 1:]
        self.obs_hist[i, -1] = obs
        self.mask[i, -1] = True

    def reset(self):
        for i, e in enumerate(self.envs):
            obs, _ = e.reset(seed=self._seed + 1000 * i)
            self._push(i, obs, True)
        self.ep_ret[:] = 0
        self.ep_len[:] = 0
        return self.obs_hist.copy(), self.mask.copy()

    def step(self, actions):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        finished = []  # (return, length) of episodes that ended this step
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            obs, r, term, trunc, _ = e.step(int(a))
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            rewards[i] = r
            if term or trunc:
                dones[i] = 1.0
                finished.append((float(self.ep_ret[i]), int(self.ep_len[i])))
                self.ep_ret[i] = 0
                self.ep_len[i] = 0
                obs, _ = e.reset()
                self._push(i, obs, True)
            else:
                self._push(i, obs, False)
        return self.obs_hist.copy(), self.mask.copy(), rewards, dones, finished


# ---------------------------------------------------------------------------
# Custom tasks (registered lazily in CUSTOM_ENVS; see continual.py)
# ---------------------------------------------------------------------------
CUSTOM_ENVS: dict = {}


def register_custom(name):
    def deco(fn):
        CUSTOM_ENVS[name] = fn
        return fn
    return deco
