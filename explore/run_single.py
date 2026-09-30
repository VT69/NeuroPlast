"""Single-task runner for the exploration phase (registers the CueFirst envs, then runs train.py's main).

    python explore/run_single.py --config configs/mem_s4.yaml --set env_id=Explore-CueFirstS11-v0 out_dir=explore/runs/mem ...
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))
os.chdir(ROOT)

import np_explore.envs  # noqa: E402,F401  (registers Explore-CueFirstS11/S13)
import train  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "train.py"
    train.main()
