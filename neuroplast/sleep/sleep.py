"""Sleep-phase consolidation (RQ2).

Wake: normal PPO on the current task (no replay in the wake loss).
Sleep: every `period` PPO updates, and once at the end of each task, the
agent goes offline (no environment interaction) for `steps_per_phase`
gradient steps on:

  1. old-task replay distillation: KL(pi_stored || pi) + value MSE on states
     stored at the end of each earlier task (same buffer as the `replay`
     baseline, so the two differ only in *when* replay happens and in 2/3);
  2. current-task rehearsal: self-distillation to a frozen snapshot of the
     network taken at sleep onset, on a reservoir of recent wake states, so
     sleep consolidates rather than overwrites what was just learned;
  3. (SNN encoders) local unsupervised STDP with synaptic scaling on the
     spikes produced by the replayed states (hebbian mode of
     neuroplast/learning/hybrid.py), optionally driven by `dream` inputs
     (random binary frames matching the mean input statistics, no stored
     data at all -- the Tadros et al. 2022 "sleep-like replay" idea).

Replay budget: the `replay` baseline uses one batch of `batch` states per PPO
minibatch; `replay_samples` counts what each arm actually consumed so the
comparison can be budget-matched (see PROGRESS.md).
"""
from __future__ import annotations

import torch

from neuroplast.baselines.buffer import TaskBuffer, collect_states
from neuroplast.baselines.replay import distill_loss
from neuroplast.learning.hybrid import HybridSTDP
from neuroplast.models.agent import snapshot as make_snapshot


class _Reservoir:
    """Reservoir sample of recent wake states (obs window + mask)."""

    def __init__(self, size=5000):
        self.size, self.n = size, 0
        self.obs = self.mask = None

    def add(self, obs, mask):
        if self.obs is None:
            self.obs = torch.zeros((self.size, *obs.shape[1:]), dtype=obs.dtype)
            self.mask = torch.zeros((self.size, *mask.shape[1:]), dtype=mask.dtype)
        for i in range(len(obs)):
            j = self.n if self.n < self.size else int(torch.randint(0, self.n + 1, ()))
            if j < self.size:
                self.obs[j], self.mask[j] = obs[i], mask[i]
            self.n += 1

    def sample(self, b):
        idx = torch.randint(0, min(self.n, self.size), (b,))
        return self.obs[idx], self.mask[idx]


class _SleepHook:
    def __init__(self, sleep, task_idx):
        self.sleep, self.task = sleep, task_idx

    def after_step(self, agent, batch, update):
        # keep a small random subset of each minibatch as wake memories
        k = max(1, len(batch["idx"]) // 16)
        sel = torch.randperm(len(batch["idx"]))[:k]
        self.sleep.reservoir.add(batch["obs"][sel], batch["mask"][sel])

    def after_update(self, agent, update, stats):
        if update % self.sleep.period == 0 and len(self.sleep.buf):
            stats.update(self.sleep.sleep_phase(agent, self.task))


class Sleep:
    def __init__(self, period=25, steps_per_phase=700, batch=128, lr=3e-4, buffer_per_task=5000,
                 rehearsal_coef=1.0, stdp_alpha=0.0, stdp_layers="all", dream=False, distill=True,
                 end_of_task_sleep=True, stdp_center=False, stdp_homeo=None, replay_budget=None):
        """replay_budget: if set, the TOTAL number of old-task replayed states over the whole sequence
        (a multiple of `batch`). The steps are spread as evenly as possible over the sleep phases the
        schedule will run, replacing steps_per_phase, so a sleep run can be matched exactly to a replay
        run's measured `replay_samples` (session 4, block 1)."""
        self.period, self.steps, self.batch, self.lr = period, steps_per_phase, batch, lr
        self.replay_budget, self._plan = replay_budget, None
        self.buffer_per_task, self.rehearsal_coef = buffer_per_task, rehearsal_coef
        self.stdp_alpha, self.stdp_layers, self.dream, self.distill = stdp_alpha, stdp_layers, dream, distill
        self.end_of_task_sleep = end_of_task_sleep
        self.stdp_center, self.stdp_homeo = stdp_center, stdp_homeo
        self.buf = TaskBuffer()
        self.reservoir = _Reservoir()
        self.replay_samples = 0
        self.n_phases = 0
        self.input_mean = None  # mean one-hot input (for dreams)

    def hooks(self, agent, task_idx, cfg):
        self.reservoir = _Reservoir()
        if self.replay_budget is not None and self._plan is None:
            self._plan = self.plan_steps(cfg)
        return [_SleepHook(self, task_idx)]

    def plan_steps(self, cfg):
        """Per-phase step counts summing to replay_budget / batch. Phases run only once the buffer is
        non-empty (tasks 1..T-1): every `period` updates plus one at the end of each such task."""
        assert self.replay_budget % self.batch == 0, "replay_budget must be a multiple of batch"
        n_steps = self.replay_budget // self.batch
        updates = max(cfg.total_frames // (cfg.num_envs * cfg.num_steps), 1)
        per_task = updates // self.period + (1 if self.end_of_task_sleep else 0)
        n_phases = per_task * (cfg.n_tasks - 1)
        q, r = divmod(n_steps, n_phases)
        return [q + (1 if i < r else 0) for i in range(n_phases)]

    def _phase_steps(self):
        if self._plan is None:
            return self.steps
        return self._plan[self.n_phases] if self.n_phases < len(self._plan) else 0

    def end_task(self, agent, task_idx, cfg):
        if self.end_of_task_sleep and len(self.buf):
            self.sleep_phase(agent, task_idx)
        o, m, l, v = collect_states(agent, cfg.env_id, task_idx, self.buffer_per_task)
        self.buf.add(o, m, task_idx, l, v)
        from neuroplast.envs.minigrid_env import obs_to_onehot
        x = obs_to_onehot(self.buf.O[:, -1]).mean(0)
        self.input_mean = x

    def sleep_phase(self, agent, task_idx):
        stdp = None
        was_recording = getattr(agent.encoder, "record_spikes", False)
        if self.stdp_alpha > 0 and hasattr(agent.encoder, "lifs"):
            stdp = HybridSTDP(alpha=self.stdp_alpha, mode="hebbian", layers=self.stdp_layers, lr=self.lr,
                              center=self.stdp_center, homeo=self.stdp_homeo)
            stdp.setup(agent)
        snapshot = make_snapshot(agent) if self.rehearsal_coef > 0 and self.reservoir.n else None
        opt = torch.optim.Adam(agent.parameters(), lr=self.lr)
        agent.train()
        tot = 0.0
        steps = self._phase_steps()
        for _ in range(steps):
            loss = 0.0
            if self.dream and stdp is not None:
                # dream: random binary frames with the stored mean input statistics, STDP only
                x = (torch.rand(self.batch, *self.input_mean.shape) < self.input_mean).float()
                with torch.no_grad():
                    agent.encoder(x)
                stdp.after_step(agent, dict(adv=None, mask=None), 0)
                if not self.distill:
                    continue
            if snapshot is not None:
                o2, m2 = self.reservoir.sample(self.batch)
                with torch.no_grad():
                    l2, v2 = snapshot(o2, m2, task_idx)
                loss = self.rehearsal_coef * distill_loss(agent, o2, m2, task_idx, l2, v2)
            if self.distill:
                # last forward = old-task replay, so STDP below sees old-task spikes
                o, m, t, l, v = self.buf.sample(self.batch)
                loss = loss + distill_loss(agent, o, m, t, l, v)
                self.replay_samples += self.batch
            if isinstance(loss, float):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()
            tot += loss.item()
            if stdp is not None and not self.dream:
                stdp.after_step(agent, dict(adv=None, mask=None), 0)
        if stdp is not None:
            agent.encoder.record_spikes = was_recording  # a wake-time STDP hook may need it
            agent.encoder.record = []
        self.n_phases += 1
        return dict(sleep_loss=tot / max(steps, 1))
