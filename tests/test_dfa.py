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
