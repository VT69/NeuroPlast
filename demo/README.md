# Demo

- `index.html`: the results dashboard, a single self-contained page (data and GIFs embedded, no network access needed).
  Open it in any browser. It contains the mechanism scorecard (same numbers and conclusions as the paper), the
  naive vs sleep agents with their real accuracy matrices, forgetting curves for every method and sequence, and
  accuracy vs total memory.
- `build_dashboard.py`: rebuilds `index.html` from `runs/` and `results/` through the paper's
  `paper/compute_numbers.py`: `python demo/build_dashboard.py`. The page layout lives in `dashboard_template.html`.
- `live_episode.py`: optional. Runs one episode of the naive or sleep agent on CPU and prints the trace and the
  agent's measured accuracy matrix: `python demo/live_episode.py --agent sleep --task 0 [--window] [--gif out.gif]`.
- `assets/`: the GIFs and their episode index (`python scripts/record_demo.py`). Playback is 400 ms per frame with
  the last frame of each episode held 1.5 s; steps after the 24th play fast-forwarded (labelled in the GIF), so
  the naive agent's 256-step time-outs stay watchable. Same rooms, actions and outcomes as the scored evaluation.
