"""Continual-learning baseline arms, used through `make_method(name, **kw)`.

A method may define:
  hooks(agent, task_idx, cfg) -> list of train.py hooks for training task_idx
  end_task(agent, task_idx, cfg)   called after task_idx finishes
"""
from __future__ import annotations

from neuroplast.baselines.ewc import EWC
from neuroplast.baselines.replay import Replay


class Naive:
    """Plain sequential fine-tuning."""


class Isolation:
    """One fresh network per task; handled directly in continual.py."""


def make_method(name, **kw):
    if name == "naive":
        return Naive()
    if name == "isolation":
        return Isolation()
    if name == "ewc":
        return EWC(**kw)
    if name == "replay":
        return Replay(**kw)
    if name == "sleep":
        from neuroplast.sleep.sleep import Sleep
        return Sleep(**kw)
    raise ValueError(name)
