"""RQ3: SNN vs dense CNN encoder, accuracy vs spike sparsity / energy.

Setup (standalone, supervised, so RL noise doesn't swamp the comparison):
  1. Roll out a trained CNN-PPO teacher (stochastic policy -> diverse states)
     on a MiniGrid task and label every visited observation with the
     teacher's greedy action. Train/test split by episode seed.
  2. Train encoder + linear readout (identical for both) by cross-entropy,
     plus lam * (mean activity) to trade accuracy for sparsity:
       CNN: activity = mean ReLU activation (L1),  SNN: mean spike rate.
     Sweep lam, and SNN timesteps T.
  3. Report test accuracy, closed-loop return of the cloned policy, and ops /
     energy per inference (see neuroplast/models/encoders.py for accounting).

Usage: python scripts/rq3_encoders.py --teacher runs/cnn_doorkey5_s1/final.pt
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.envs.minigrid_env import N_ACTIONS, VecEnv, obs_to_onehot  # noqa: E402
from neuroplast.models.encoders import make_encoder  # noqa: E402
from train import Config, build_agent  # noqa: E402


@torch.no_grad()
def collect(teacher, env_id, n_steps, seed, num_envs=16):
    envs = VecEnv(env_id, num_envs, seed=seed)
    o, m = envs.reset()
    X, Y, V = [], [], []
    for _ in range(n_steps // num_envs):
        ot = torch.as_tensor(o)
        logits, v = teacher(ot, torch.as_tensor(m))
        X.append(ot[:, -1].clone())
        Y.append(logits.argmax(-1))
        V.append(v)
        a = torch.distributions.Categorical(logits=logits).sample()
        o, m, _, _, _ = envs.step(a.numpy())
    return torch.cat(X), torch.cat(Y), torch.cat(V)


class Student(nn.Module):
    def __init__(self, kind, **kw):
        super().__init__()
        self.enc = make_encoder(kind, **kw)
        self.head = nn.Linear(self.enc.out_dim, N_ACTIONS)
        self.vhead = nn.Linear(self.enc.out_dim, 1)
        self.last_v = None

    def forward(self, obs_u8):
        z = self.enc(obs_to_onehot(obs_u8))
        self.last_v = self.vhead(z).squeeze(-1)
        return self.head(z)


@torch.no_grad()
def closed_loop(student, env_id, episodes=100, seed=777):
    envs = VecEnv(env_id, 16, seed=seed)
    o, _ = envs.reset()
    fin = []
    while len(fin) < episodes:
        a = student(torch.as_tensor(o[:, -1])).argmax(-1)
        o, _, _, _, f = envs.step(a.numpy())
        fin.extend(f)
    r = np.array([e[0] for e in fin[:episodes]])
    return float(r.mean()), float((r > 0).mean())


def run_one(args):
    kind, T, lam, seed, data_path, env_id, epochs = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    d = torch.load(data_path)
    Xtr, Ytr, Xte, Yte = d["Xtr"], d["Ytr"], d["Xte"], d["Yte"]
    Vtr, Vte = d.get("Vtr"), d.get("Vte")
    if Vtr is not None:  # z-score the value target so its loss scale is comparable to CE
        mu, sd = Vtr.mean(), Vtr.std()
        Vtr, Vte = (Vtr - mu) / sd, (Vte - mu) / sd
    kw = dict(T=T) if kind == "snn" else {}
    st = Student(kind, **kw)
    opt = torch.optim.Adam(st.parameters(), 1e-3)
    n = len(Xtr)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(st(Xtr[idx]), Ytr[idx]) + lam * st.enc.act_reg
            if Vtr is not None:  # value regression: graded target that doesn't saturate
                loss = loss + F.mse_loss(st.last_v, Vtr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - t0
    st.eval()
    with torch.no_grad():
        st.enc.track = True
        accs, stats, vpred = [], [], []
        for i in range(0, len(Xte), 1000):
            accs.append((st(Xte[i:i + 1000]).argmax(-1) == Yte[i:i + 1000]).float().mean().item())
            vpred.append(st.last_v)
            stats.append(dict(st.enc.stats))
        st.enc.track = False
    agg = {k: float(np.mean([s[k] for s in stats])) for k in ("dense_macs", "sparse_ops", "energy_pj", "energy_sparse_pj")}
    agg["rates"] = np.mean([s["rates"] for s in stats], 0).tolist()
    ret, succ = closed_loop(st, env_id)
    v_r2 = v_r2_train = None
    if Vte is not None:
        vp = torch.cat(vpred)
        v_r2 = float(1 - ((vp - Vte) ** 2).mean() / Vte.var())
        with torch.no_grad():  # train-set R^2 (underfit vs. generalisation gap)
            st(Xtr[:8000])
            v_r2_train = float(1 - ((st.last_v - Vtr[:8000]) ** 2).mean() / Vtr[:8000].var())
    res = dict(kind=kind, T=T if kind == "snn" else 1, lam=lam, seed=seed, acc=float(np.mean(accs)),
               return_mean=ret, success=succ, v_r2=v_r2, v_r2_train=v_r2_train, train_time=train_time, **agg)
    print(json.dumps(res), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="runs/cnn_doorkey5_s1/final.pt")
    p.add_argument("--env", default=None)
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-test", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--lams", default="0,0.03,0.1,0.3,1,3")
    p.add_argument("--Ts", default="2,4,8")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--out", default="runs/rq3_encoders")
    a = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.teacher)
    cfg = Config(**ck["cfg"])
    env_id = a.env or cfg.env_id
    data_path = os.path.join(a.out, "data.pt")
    if not os.path.exists(data_path):
        teacher = build_agent(cfg)
        teacher.load_state_dict(ck["agent"])
        teacher.eval()
        Xtr, Ytr, Vtr = collect(teacher, env_id, a.n_train, seed=1)
        Xte, Yte, Vte = collect(teacher, env_id, a.n_test, seed=99_999)
        torch.save(dict(Xtr=Xtr, Ytr=Ytr, Vtr=Vtr, Xte=Xte, Yte=Yte, Vte=Vte), data_path)
        print("collected", Xtr.shape, Xte.shape, "label dist", torch.bincount(Ytr, minlength=7).tolist())
    lams = [float(x) for x in a.lams.split(",")]
    Ts = [int(x) for x in a.Ts.split(",")]
    jobs = [("cnn", 1, l, s, data_path, env_id, a.epochs) for l, s in itertools.product(lams, range(a.seeds))]
    jobs += [("snn", T, l, s, data_path, env_id, a.epochs) for T, l, s in itertools.product(Ts, lams, range(a.seeds))]
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["kind"], r["T"], r["lam"], r["seed"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} jobs to run")
    with Pool(a.procs) as pool, open(res_path, "a") as f:
        for r in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
