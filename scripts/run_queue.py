"""Tiny resumable job queue: run shell commands N at a time.

Jobs file format, one job per line:   <done_marker_path> :: <shell command>
A job is skipped if its done marker exists or another live worker holds
its <marker>.lock (so several workers can drain the same file) (e.g. the run's eval.json /
results.json), so re-running the same queue after an interruption resumes it.
stdout/stderr of each job go to runs/logs/<marker-derived-name>.log.

    python scripts/run_queue.py jobs/cl_cnn.txt --procs 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def done(marker):
    if not os.path.exists(marker):
        return False
    if marker.endswith("results.json"):  # continual runs write results.json after every task
        try:
            return json.load(open(marker)).get("metrics") is not None
        except Exception:
            return False
    return True


def claim(marker):
    """Atomically claim a job via <marker>.lock holding our pid; stale locks (dead pid) are taken over."""
    os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
    lock = marker + ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                pid = int(open(lock).read().strip() or 0)
                os.kill(pid, 0)
                return None  # held by a live worker
            except (ProcessLookupError, ValueError):
                os.remove(lock)  # stale
            except PermissionError:
                return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--procs", type=int, default=4)
    a = p.parse_args()
    jobs = []
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        marker, cmd = (x.strip() for x in line.split("::", 1))
        if not done(marker):
            jobs.append((marker, cmd))
    print(f"{len(jobs)} jobs pending", flush=True)
    os.makedirs("runs/logs", exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # every job is single-threaded by design
    running = []
    while jobs or running:
        while jobs and len(running) < a.procs:
            marker, cmd = jobs.pop(0)
            if done(marker):
                continue
            lock = claim(marker)
            if lock is None:
                continue
            name = marker.replace("runs/", "").replace("/", "__").rsplit(".", 1)[0]
            log = open(f"runs/logs/{name}.log", "w")
            running.append((subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT), marker, time.time(), lock))
            print(f"START {marker}", flush=True)
        time.sleep(5)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                if os.path.exists(r[3]):
                    os.remove(r[3])
                print(f"END rc={r[0].returncode} {time.time() - r[2]:.0f}s {r[1]}", flush=True)


if __name__ == "__main__":
    main()
