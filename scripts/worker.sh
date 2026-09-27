#!/usr/bin/env bash
# Drain job files in order with N parallel slots. Several workers can run at once
# (jobs are claimed with lock files), e.g. start another worker when cores free up.
#   scripts/worker.sh 2 jobs/rq1.txt jobs/cl_snn.txt
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
procs=$1; shift
for f in "$@"; do
  python scripts/run_queue.py "$f" --procs "$procs"
done
