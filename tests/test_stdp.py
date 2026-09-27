import torch
import torch.nn.functional as F

from neuroplast.learning.stdp import EligibilityTrace, stdp_conv, stdp_dense


def seq(T, B, n, spikes):
    s = [torch.zeros(B, n) for _ in range(T)]
    for t, i in spikes:
        s[t][:, i] = 1
    return s


def test_causal_pairing_potentiates():
    pre = seq(5, 1, 1, [(1, 0)])
    post = seq(5, 1, 1, [(3, 0)])
    assert stdp_dense(pre, post).item() > 0


def test_anticausal_pairing_depresses():
    pre = seq(5, 1, 1, [(3, 0)])
    post = seq(5, 1, 1, [(1, 0)])
    assert stdp_dense(pre, post).item() < 0


def test_same_step_is_causal():
    pre = seq(3, 1, 1, [(1, 0)])
    post = seq(3, 1, 1, [(1, 0)])
    assert stdp_dense(pre, post).item() == 1.0


def test_trace_decay_with_lag():
    lam = 0.5
    vals = []
    for lag in range(4):
        pre = seq(6, 1, 1, [(0, 0)])
        post = seq(6, 1, 1, [(lag, 0)])
        vals.append(stdp_dense(pre, post, lam_pre=lam).item())
    assert vals == [1.0, 0.5, 0.25, 0.125]


def test_no_cross_talk_between_synapses():
    pre = seq(4, 1, 2, [(0, 0)])      # only input 0 fires
    post = seq(4, 1, 2, [(1, 1)])     # only output 1 fires
    dW = stdp_dense(pre, post)
    assert dW[1, 0] > 0 and dW[0, 0] == 0 and dW[0, 1] == 0 and dW[1, 1] == 0


def test_modulator_sign_and_batch_mean():
    pre = seq(4, 2, 1, [(0, 0)])
    post = seq(4, 2, 1, [(1, 0)])
    base = stdp_dense(pre, post).item()
    assert stdp_dense(pre, post, torch.tensor([-1.0, -1.0])).item() == -base
    assert abs(stdp_dense(pre, post, torch.tensor([1.0, -1.0])).item()) < 1e-7


def test_conv_1x1_equals_dense():
    torch.manual_seed(0)
    T, B, C, O, H = 4, 3, 5, 6, 4
    pre = [(torch.rand(B, C, H, H) < 0.3).float() for _ in range(T)]
    post = [(torch.rand(B, O, H, H) < 0.3).float() for _ in range(T)]
    m = torch.randn(B)
    dconv = stdp_conv(pre, post, 1, m)[:, :, 0, 0]
    # dense over (batch x location) samples, modulator repeated per location
    flat = lambda z: z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    ddense = stdp_dense([flat(p) for p in pre], [flat(p) for p in post], m.repeat_interleave(H * H), mean=False) / B
    assert torch.allclose(dconv, ddense, atol=1e-5)


def test_conv_matches_bruteforce():
    torch.manual_seed(1)
    T, B, C, O = 3, 2, 2, 3
    pre = [(torch.rand(B, C, 4, 4) < 0.4).float() for _ in range(T)]
    post = [(torch.rand(B, O, 3, 3) < 0.4).float() for _ in range(T)]
    dW = stdp_conv(pre, post, 2, mean=False, lam_pre=0.5, lam_post=0.5)
    ref = torch.zeros(O, C, 2, 2)
    for o in range(O):
        for c in range(C):
            for a in range(2):
                for bb in range(2):
                    p = [z[:, c, a:a + 3, bb:bb + 3] for z in pre]
                    q = [z[:, o] for z in post]
                    fl = lambda s: [u.reshape(-1, 1) for u in s]
                    ref[o, c, a, bb] = stdp_dense(fl(p), fl(q), mean=False, lam_pre=0.5, lam_post=0.5).item()
    assert torch.allclose(dW, ref, atol=1e-5)


def test_online_eligibility_matches_batched_sum():
    torch.manual_seed(2)
    T, B, n_in, n_out = 6, 4, 5, 3
    pre = [(torch.rand(B, n_in) < 0.3).float() for _ in range(T)]
    post = [(torch.rand(B, n_out) < 0.3).float() for _ in range(T)]
    et = EligibilityTrace(n_in, n_out, lam_e=1.0)
    for p, q in zip(pre, post):
        et.step(p, q)
    assert torch.allclose(et.e, stdp_dense(pre, post, mean=False), atol=1e-5)
    w = torch.zeros(n_out, n_in)
    et.apply(w, modulator=-2.0, eta=0.1)
    assert torch.allclose(w, -0.2 * et.e)


def test_eligibility_decays_without_activity():
    et = EligibilityTrace(1, 1, lam_e=0.5)
    et.step(torch.ones(1, 1), torch.ones(1, 1))
    e0 = et.e.clone()
    et.step(torch.zeros(1, 1), torch.zeros(1, 1))
    assert torch.allclose(et.e, 0.5 * e0)
