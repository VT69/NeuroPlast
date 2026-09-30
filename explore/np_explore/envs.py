"""CueFirst memory benchmark (idea 4).

MiniGrid's MemoryEnv starts the agent at a random x in the start room / hallway, facing the hallway, so the cue
object (at x = 1, one row above the hallway) is usually *behind* the agent and never seen unless it turns back.
Session 4 found that no agent learned to do that. CueFirst starts the agent at x = 1 facing the hallway: the cue
is directly to its left (visible at step 0) and out of view from the junction (8 cells east on S11). Everything
else (objects, reward, success/failure cells, max_steps) is MiniGrid's MemoryEnv unchanged.
"""
from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register, registry
from minigrid.envs.memory import MemoryEnv


class CueFirstMemoryEnv(MemoryEnv):
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.agent_pos = np.array((1, height // 2))
        self.agent_dir = 0


for size in (11, 13):
    _id = f"Explore-CueFirstS{size}-v0"
    if _id not in registry:
        register(id=_id, entry_point="np_explore.envs:CueFirstMemoryEnv", kwargs=dict(size=size, random_length=False))
