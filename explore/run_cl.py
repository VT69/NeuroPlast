"""Continual runner for the exploration phase: continual.py's run() plus the new arms (packnet, lwf).

    python explore/run_cl.py --suite fetch3 --method packnet --config configs/cl_cnn.yaml --seed 101 --out explore/runs/cl

Writes explore/runs/... only. PackNet's owner-map bits and LwF's snapshot are recorded in results.json.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explore"))
os.chdir(ROOT)

import continual  # noqa: E402
from np_explore import methods  # noqa: E402

_made = {}


def _make(name, **kw):
    m = methods.make_method(name, **kw)
    _made["m"] = m
    return m


continual.make_method = _make  # the only change: route method construction through the explore registry


def main():
    sys.argv[0] = "continual.py"
    if "--out" not in sys.argv:
        sys.argv += ["--out", "explore/runs/cl"]
    orig_run = continual.run

    def run(*a, **k):
        res = orig_run(*a, **k)
        m = _made.get("m")
        extra = dict(extra_bits=getattr(m, "extra_bits", 0), packnet_kept=getattr(m, "kept", None))
        out = os.path.join(a[4] if len(a) > 4 else k.get("out_root"), f"{a[0]}_{a[1]}{k.get('tag', a[7] if len(a) > 7 else '')}_s{a[3]}")
        path = os.path.join(out, "results.json")
        r = json.load(open(path))
        r.update(extra)
        json.dump(r, open(path, "w"), indent=1)
        res.update(extra)
        return res
    continual.run = run
    continual.main()


if __name__ == "__main__":
    main()
