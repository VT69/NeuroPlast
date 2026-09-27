import torch
import torch.nn as nn

from neuroplast.models.encoders import CNNEncoder, SNNEncoder, _conv_event_ops
from neuroplast.models.snn.lif import LIF, spike_fn


def test_lif_matches_snntorch_leaky():
    snn = __import__("snntorch")
    torch.manual_seed(0)
    cur = torch.rand(20, 5, 8) * 0.6
    ours = LIF(beta=0.8, threshold=1.0)
    ref = snn.Leaky(beta=0.8, threshold=1.0, reset_mechanism="subtract")
    mem_ref = ref.init_leaky()
    v = s = None
    for t in range(cur.shape[0]):
        s, v = ours(cur[t], v, s)
        s_ref, mem_ref = ref(cur[t], mem_ref)
        # same spike trains under reset-by-subtraction
        assert torch.equal(s, s_ref), t


def test_lif_constant_input_rate():
    lif = LIF(beta=1.0 - 1e-6, threshold=1.0)
    v = s = None
    n = 0
    for _ in range(100):
        s, v = lif(torch.full((1,), 0.25), v, s)
        n += int(s.item())
    assert 24 <= n <= 26  # integrate-and-fire with I=0.25, thr=1 -> rate 1/4


def test_surrogate_gradient_nonzero():
    v = torch.linspace(-1, 1, 11, requires_grad=True)
    spike_fn(v).sum().backward()
    assert (v.grad > 0).all()
    assert v.grad[5] == v.grad.max()  # peak at threshold


def test_encoders_shapes_and_initial_activity():
    torch.manual_seed(0)
    x = (torch.rand(32, 20, 7, 7) < 0.15).float()
    for enc in (CNNEncoder(), SNNEncoder(T=4)):
        enc.track = True
        y = enc(x)
        assert y.shape == (32, 128)
        assert set(enc.stats) >= {"dense_macs", "sparse_ops", "rates", "energy_pj"}
    snn = SNNEncoder(T=4)
    snn(x)
    r = snn.act_reg.item()
    assert 0.02 < r < 0.6, r  # not silent, not saturated at init


def test_event_ops_exact():
    conv = nn.Conv2d(2, 3, 2)
    x = torch.zeros(1, 2, 3, 3)
    x[0, 0, 1, 1] = 1  # centre pixel is in all 4 receptive fields
    x[0, 1, 0, 0] = 1  # corner pixel only in 1
    assert _conv_event_ops(x, conv).item() == (4 + 1) * 3


def test_snn_learns_toy_classification():
    torch.manual_seed(0)
    x = (torch.rand(256, 20, 7, 7) < 0.15).float()
    y = (x[:, 3].sum((1, 2)) > x[:, 5].sum((1, 2))).long()
    enc = SNNEncoder(T=4)
    head = nn.Linear(128, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 3e-3)
    for _ in range(150):
        loss = nn.functional.cross_entropy(head(enc(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (head(enc(x)).argmax(1) == y).float().mean().item()
    assert acc > 0.9, acc


def test_record_spikes_for_stdp():
    enc = SNNEncoder(T=3)
    enc.record_spikes = True
    enc((torch.rand(4, 20, 7, 7) < 0.2).float())
    assert len(enc.record) == 4
    pre, post = enc.record[3]
    assert len(pre) == 3 and pre[0].shape == (4, 1024) and post[0].shape == (4, 128)
