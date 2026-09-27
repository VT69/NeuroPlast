"""Continual-learning task suites.

`fetch3`: "FetchObj" room (6x6 interior) containing one red ball, one green
key and one blue box at random positions. Task k rewards picking up object k
(reward 1 - 0.9 * steps/max_steps, as in MiniGrid); picking up any other
object ends the episode with 0. Observations are statistically identical
across tasks, so the tasks can only be told apart by the task ID the agent is
given (task embedding + per-task heads = standard task-incremental setting).
This guarantees interference under naive fine-tuning (the same state demands
different actions), while navigation + pickup is a shared skill that allows
forward transfer (RQ5).

`natural3`: stock MiniGrid tasks with visually distinct observations
(DoorKey-5x5 -> LavaGapS5 -> Dynamic-Obstacles-5x5). A weaker-interference,
more "natural" sequence to check the conclusions aren't an artefact of the
adversarial fetch suite.
"""
from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv

from neuroplast.envs.minigrid_env import register_custom

FETCH_OBJECTS = [("ball", "red"), ("key", "green"), ("box", "blue")]


class FetchObjEnv(MiniGridEnv):
    def __init__(self, target: int = 0, size: int = 8, max_steps: int | None = None, **kwargs):
        self.target = target
        mission_space = MissionSpace(mission_func=lambda: "fetch the target object")
        super().__init__(mission_space=mission_space, grid_size=size, see_through_walls=True,
                         max_steps=max_steps or 4 * size * size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.objs = []
        for kind, color in FETCH_OBJECTS:
            obj = {"ball": Ball, "key": Key, "box": Box}[kind](color)
            self.place_obj(obj)
            self.objs.append(obj)
        self.place_agent()
        self.mission = "fetch the target object"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.carrying is not None:
            idx = next(i for i, o in enumerate(self.objs) if o is self.carrying)
            terminated = True
            reward = self._reward() if idx == self.target else 0.0
            info["picked"] = idx
        return obs, reward, terminated, truncated, info


for _k in range(3):
    register_custom(f"FetchObj-{_k}")(lambda _k=_k, **kw: FetchObjEnv(target=_k, **kw))

SUITES = {
    "fetch3": ["FetchObj-0", "FetchObj-1", "FetchObj-2"],
    "natural3": ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-LavaGapS5-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0"],
}
