"""Tests for the handwritten multi-head self-attention (src/model/attention.py).

Interface these tests expect (the last nn.Module defined in attention.py):

    attention = Cls(d_model, n_heads)
    y = attention(x, mask)     # x: (B, T, d_model) -> y: (B, T, d_model); mask is optional

`mask` is a boolean tensor broadcastable to (B, 1, T, T) where True means "this query may
attend to this key" (the convention of F.scaled_dot_product_attention). With no mask every
position attends to every position. Heads are contiguous slices of the channels, and the
Q/K/V projections and the output projection are nn.Linear layers.

test_matches_torch_reference is the equivalence test required by the brief: it records what
your own nn.Linear layers compute, then checks the rest of your module against
F.scaled_dot_product_attention on the same Q, K, V and the same mask (float32, about 1e-5).
"""

import itertools

import pytest
import torch
import torch.nn.functional as F
from torch import nn

B, T, D_MODEL, N_HEADS = 3, 10, 32, 4
N_VISUAL = 6  # size of the fully-visible prefix in the prefix-LM mask


def make(attention_cls, seed=0):
    torch.manual_seed(seed)
    return attention_cls(D_MODEL, N_HEADS).eval()


def randn(*shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def run(attention, x, mask=None):
    return attention(x) if mask is None else attention(x, mask)


def causal(t):
    return torch.tril(torch.ones(t, t, dtype=torch.bool))


def mask_cases():
    random_mask = torch.rand(B, 1, T, T, generator=torch.Generator().manual_seed(2)) < 0.6
    random_mask |= torch.eye(T, dtype=torch.bool)  # every query keeps at least itself
    prefix_lm = causal(T).clone()
    prefix_lm[:N_VISUAL, :N_VISUAL] = True  # "visual" tokens see each other, letters are causal
    return {
        "none": None,
        "causal_2d": causal(T),
        "causal_4d": causal(T)[None, None],
        "random_per_sample": random_mask,
        "prefix_lm": prefix_lm,
    }


def split_heads(t, n_heads):  # (B, T, D) -> (B, H, T, D / H)
    b, length, d = t.shape
    return t.view(b, length, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(t):  # (B, H, T, D / H) -> (B, T, D)
    b, h, length, dh = t.shape
    return t.transpose(1, 2).reshape(b, length, h * dh)


def trace_linears(attention, x, mask):
    """Run the module once, recording (linear, input, output) for every nn.Linear call."""
    records = []

    def hook(module, inputs, output):
        records.append((module, inputs[0].detach(), output.detach()))

    handles = [m.register_forward_hook(hook) for m in attention.modules() if isinstance(m, nn.Linear)]
    try:
        with torch.no_grad():
            y = run(attention, x, mask)
    finally:
        for handle in handles:
            handle.remove()
    return y, records


# --- equivalence with F.scaled_dot_product_attention ------------------------


@pytest.mark.parametrize("scale", [1.0, 20.0], ids=["unit-scale", "large-activations"])
@pytest.mark.parametrize("mask_name", list(mask_cases()))
def test_matches_torch_reference(attention_cls, mask_name, scale):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL) * scale
    mask = mask_cases()[mask_name]
    y, records = trace_linears(attention, x, mask)

    fed_with_x = [r for r in records if r[1].shape == x.shape and torch.equal(r[1], x)]
    others = [r for r in records if not (r[1].shape == x.shape and torch.equal(r[1], x))]
    assert fed_with_x, "no nn.Linear is applied directly to the input x (expected the Q/K/V projections)"
    assert len(others) == 1, "expected exactly one more nn.Linear: the output projection, applied to the merged heads"
    _, merged_heads, projected = others[0]

    outputs = [out for _, _, out in fed_with_x]
    if len(outputs) == 1 and outputs[0].shape[-1] == 3 * D_MODEL:
        qkv = list(outputs[0].chunk(3, dim=-1))  # one fused projection
    elif len(outputs) == 3 and all(out.shape[-1] == D_MODEL for out in outputs):
        qkv = outputs  # three separate projections
    else:
        pytest.fail("expected either three Linear(d_model, d_model) or one Linear(d_model, 3 * d_model) on x")

    # Swapping the roles of Q and K is a relabelling of the weights, so try every assignment.
    tolerance = dict(atol=1e-5, rtol=1e-5) if scale == 1.0 else dict(atol=1e-3, rtol=1e-3)
    for q, k, v in itertools.permutations(qkv):
        expected = merge_heads(
            F.scaled_dot_product_attention(
                split_heads(q, N_HEADS), split_heads(k, N_HEADS), split_heads(v, N_HEADS), attn_mask=mask
            )
        )
        if torch.isfinite(merged_heads).all() and torch.allclose(expected, merged_heads, **tolerance):
            break
    else:
        pytest.fail(
            f"the attention core does not match F.scaled_dot_product_attention (mask={mask_name}, scale={scale}). "
            "Check the sqrt(d_k) scaling, the mask, the softmax dimension, and the head split / merge."
        )
    torch.testing.assert_close(y, projected, **tolerance)


# --- shape and configuration ------------------------------------------------


@pytest.mark.parametrize("length", [1, 7, 45])
def test_output_shape_and_finite(attention_cls, length):
    y = run(make(attention_cls), randn(B, length, D_MODEL))
    assert y.shape == (B, length, D_MODEL)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_rejects_d_model_not_divisible_by_the_number_of_heads(attention_cls):
    with pytest.raises((ValueError, AssertionError)):
        attention_cls(30, 4)


def test_eval_mode_is_deterministic(attention_cls):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL)
    assert torch.equal(run(attention, x, causal(T)), run(attention, x, causal(T)))


# --- what the mask does -----------------------------------------------------


def test_causal_mask_blocks_the_future(attention_cls):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = randn(B, T - 6, D_MODEL, seed=3)
    before = run(attention, x, causal(T))
    after = run(attention, changed, causal(T))
    torch.testing.assert_close(before[:, :6], after[:, :6], atol=1e-6, rtol=0)
    assert not torch.allclose(before[:, 6:], after[:, 6:])


def test_without_a_mask_every_position_sees_the_future(attention_cls):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL)
    changed = x.clone()
    changed[:, 6:] = randn(B, T - 6, D_MODEL, seed=3)
    assert not torch.allclose(run(attention, x)[:, :6], run(attention, changed)[:, :6], atol=1e-4)


def test_a_masked_out_key_does_not_influence_other_positions(attention_cls):
    attention = make(attention_cls)
    j = 4
    mask = torch.ones(T, T, dtype=torch.bool)
    mask[:, j] = False  # nobody may attend to position j ...
    mask[j, j] = True  # ... except j itself, so its row is not empty
    x = randn(B, T, D_MODEL)
    changed = x.clone()
    changed[:, j] = randn(B, D_MODEL, seed=3)
    before, after = run(attention, x, mask), run(attention, changed, mask)
    others = [i for i in range(T) if i != j]
    torch.testing.assert_close(before[:, others], after[:, others], atol=1e-6, rtol=0)


def test_causal_outputs_do_not_depend_on_the_sequence_length(attention_cls):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL)
    k = 6
    full = run(attention, x, causal(T))
    prefix_only = run(attention, x[:, :k], causal(k))
    torch.testing.assert_close(full[:, :k], prefix_only, atol=1e-6, rtol=0)


@pytest.mark.parametrize("mask_name", ["none", "causal_2d"])
def test_samples_in_a_batch_are_independent(attention_cls, mask_name):
    attention = make(attention_cls)
    mask = mask_cases()[mask_name]
    x = randn(B, T, D_MODEL)
    changed = x.clone()
    changed[1] = randn(T, D_MODEL, seed=3)
    before, after = run(attention, x, mask), run(attention, changed, mask)
    torch.testing.assert_close(before[0], after[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(before[2], after[2], atol=1e-6, rtol=0)


# --- gradients --------------------------------------------------------------


def test_every_parameter_receives_a_gradient(attention_cls):
    attention = make(attention_cls)
    x = randn(B, T, D_MODEL).requires_grad_()
    run(attention, x, causal(T)).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), name
        if param.ndim >= 2:  # a key bias has an exactly-zero gradient (softmax ignores it), so only check weights
            assert param.grad.abs().sum() > 0, f"{name} has an all-zero gradient"
