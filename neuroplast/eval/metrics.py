"""Continual-learning metrics.

R[i][j] = performance (mean return) on task j after finishing training on
task i (row -1 / "init" = before any training is stored separately).

  ACC        mean_j R[T-1][j]                     final average performance
  BWT        mean_{j<T-1} R[T-1][j] - R[j][j]     backward transfer (<0 = forgetting)
  FORGET     mean_{j<T-1} max_{i<T-1} R[i][j] - R[T-1][j]   (Chaudhry et al. 2018)
  FWT_j      (AUC_j - AUC_ref_j) / (1 - AUC_ref_j)  forward transfer (Continual World):
             AUC of task j's training curve vs. a from-scratch reference curve
             on the same task, both normalised to [0, 1] return.
"""
from __future__ import annotations

import numpy as np


def cl_metrics(R, curves=None, ref_curves=None):
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    out = dict(ACC=float(R[-1].mean()))
    if T > 1:
        out["BWT"] = float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))
        out["FORGET"] = float(np.mean([R[:T - 1, j].max() - R[-1, j] for j in range(T - 1)]))
    if curves is not None and ref_curves is not None:
        fwt = []
        for c, rc in zip(curves, ref_curves):
            a, ar = auc(c), auc(rc)
            fwt.append((a - ar) / (1 - ar) if ar < 1 else 0.0)
        out["FWT"] = float(np.mean(fwt[1:])) if len(fwt) > 1 else 0.0
        out["FWT_per_task"] = [float(x) for x in fwt]
    return out


def auc(curve):
    """Mean of a return curve (list of returns at evenly spaced checkpoints)."""
    c = np.nan_to_num(np.asarray(curve, dtype=float), nan=0.0)
    return float(c.mean()) if len(c) else 0.0
