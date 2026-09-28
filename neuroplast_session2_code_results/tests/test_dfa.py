import torch

from neuroplast.models.agent import Agent


def _loss(agent, obs, mask):
    logits, value = agent(obs, mask)
    return logits.pow(2).sum() + (value - 1).pow(2).sum()


def test_dfa_equals_backprop_when_feedback_is_transposed_readout():
    """With B_last = W_readout^T, DFA's gradient for the last SNN layer == backprop's."""
    torch.manual_seed(0)
    obs = torch.randint(0, 6, (8, 1, 7, 7, 3)).to(torch.uint8)
    mask = torch.ones(8, 1, dtype=torch.bool)
    dfa = Agent(encoder="snn", hidden=0, dfa=True)
    bp = Agent(encoder="snn", hidden=0)
    bp.load_state_dict({k: v for k, v in dfa.state_dict().items() if ".dfa." not in "." + k})
    W = torch.cat([dfa.heads[0].actor.weight, dfa.heads[0].critic.weight], 0)  # (8, 128)
    dfa.dfa.B3.copy_(W.T.detach())
    _loss(dfa, obs, mask).backward()
    _loss(bp, obs, mask).backward()
    g_dfa, g_bp = dfa.encoder.fc.weight.grad, bp.encoder.fc.weight.grad
    assert g_bp.abs().sum() > 0
    assert torch.allclose(g_dfa, g_bp, atol=1e-5)
    # readout gradients are identical (they are local in both)
    assert torch.allclose(dfa.heads[0].actor.weight.grad, bp.heads[0].actor.weight.grad, atol=1e-6)
    # earlier layers get *different* (random-feedback) but nonzero gradients
    g0d, g0b = dfa.encoder.convs[0].weight.grad, bp.encoder.convs[0].weight.grad
    assert g0d.abs().sum() > 0 and not torch.allclose(g0d, g0b)


def test_two_forwards_keep_separate_errors():
    torch.manual_seed(1)
    agent = Agent(encoder="snn", hidden=0, dfa=True)
    o1 = torch.randint(0, 6, (4, 1, 7, 7, 3)).to(torch.uint8)
    o2 = torch.randint(0, 6, (6, 1, 7, 7, 3)).to(torch.uint8)
    m = lambda n: torch.ones(n, 1, dtype=torch.bool)
    (_loss(agent, o1, m(4)) + _loss(agent, o2, m(6))).backward()  # batch sizes differ: must not mix
    assert agent.encoder.convs[0].weight.grad.abs().sum() > 0


def _sum_loss(agent, obs, mask, task=None):
    logits, value = agent(obs, mask, task)
    return logits.pow(2).sum() + (value - 1).pow(2).sum()


def test_dfa_mlp_heads_match_backprop_with_transposed_feedback():
    """MLP heads: with B_ha / B_hc = the output layers' transposed weights (zero-padded to
    the [logits, value] error), head hidden-layer gradients == backprop's."""
    torch.manual_seed(3)
    obs = torch.randint(0, 6, (8, 1, 7, 7, 3)).to(torch.uint8)
    mask = torch.ones(8, 1, dtype=torch.bool)
    dfa = Agent(encoder="snn", hidden=64, dfa=True)
    bp = Agent(encoder="snn", hidden=64)
    bp.load_state_dict({k: v for k, v in dfa.state_dict().items() if ".dfa." not in "." + k})
    h = dfa.heads[0]
    Ba = torch.zeros(64, 8)
    Ba[:, :7] = h.actor[2].weight.T.detach()
    Bc = torch.zeros(64, 8)
    Bc[:, 7:] = h.critic[2].weight.T.detach()
    dfa.dfa.Bha.copy_(Ba)
    dfa.dfa.Bhc.copy_(Bc)
    _sum_loss(dfa, obs, mask).backward()
    _sum_loss(bp, obs, mask).backward()
    for name in ("actor", "critic"):
        for i in (0, 2):
            g_d = getattr(dfa.heads[0], name)[i].weight.grad
            g_b = getattr(bp.heads[0], name)[i].weight.grad
            assert torch.allclose(g_d, g_b, atol=1e-5), (name, i)
    # the encoder, however, learns from random feedback (differs from backprop)
    assert not torch.allclose(dfa.encoder.fc.weight.grad, bp.encoder.fc.weight.grad)


def test_dfa_multihead_mixed_batch_routes_error_rows():
    """A mixed-task batch gives each head the same gradient as its task's rows alone."""
    torch.manual_seed(4)
    obs = torch.randint(0, 6, (9, 1, 7, 7, 3)).to(torch.uint8)
    mask = torch.ones(9, 1, dtype=torch.bool)
    task = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    agent = Agent(encoder="snn", hidden=64, dfa=True, n_tasks=3, multihead=True).train()
    _sum_loss(agent, obs, mask, task).backward()
    g_mixed = agent.heads[1].actor[0].weight.grad.clone()
    agent.zero_grad()
    sel = task == 1
    _sum_loss(agent, obs[sel], mask[sel], 1).backward()
    assert torch.allclose(g_mixed, agent.heads[1].actor[0].weight.grad, atol=1e-5)
    assert agent.task_emb.weight.grad is not None and agent.task_emb.weight.grad.abs().sum() > 0


def test_dfa_encoder_scope_heads_exact_encoder_random():
    """dfa_scope='encoder': head gradients == backprop, SNN gradients come from random feedback."""
    torch.manual_seed(5)
    obs = torch.randint(0, 6, (8, 1, 7, 7, 3)).to(torch.uint8)
    mask = torch.ones(8, 1, dtype=torch.bool)
    dfa = Agent(encoder="snn", hidden=64, dfa=True, dfa_scope="encoder")
    bp = Agent(encoder="snn", hidden=64)
    bp.load_state_dict({k: v for k, v in dfa.state_dict().items() if ".dfa." not in "." + k})
    _sum_loss(dfa, obs, mask).backward()
    _sum_loss(bp, obs, mask).backward()
    for name in ("actor", "critic"):
        for i in (0, 2):
            assert torch.allclose(getattr(dfa.heads[0], name)[i].weight.grad,
                                  getattr(bp.heads[0], name)[i].weight.grad, atol=1e-6)
    assert dfa.encoder.convs[0].weight.grad.abs().sum() > 0
    assert not torch.allclose(dfa.encoder.fc.weight.grad, bp.encoder.fc.weight.grad)
