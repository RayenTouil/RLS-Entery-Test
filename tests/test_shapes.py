"""Shape and sanity tests across the pipeline: image -> CNN feature map, and image + letters -> logits.

Interface these tests expect:

    encoder = Cls()                      # last nn.Module in src/model/encoder.py, default arguments
    feature_map = encoder(images)        # (B, 3, 64, 64) float32 -> (B, C, H', W')

    model = Cls()                        # last nn.Module in src/model/vlm.py, default arguments
    logits = model(images, input_ids)    # images (B, 3, 64, 64) float32, input_ids (B, T) int64 in [0, 27)
                                         # -> logits (B, T_out, 27) with T_out >= T

The last T logits must line up with input_ids: logits[:, -T + i] may depend on the image and on
input_ids[:, :i + 1] only (the decoder is causal). Which target each logit is trained on is your choice.
"""

import pytest
import torch

VOCAB_SIZE = 27  # 26 letters + <eos>
MAX_LETTERS = 45  # longest word


def images(batch, seed=0):
    return torch.randn(batch, 3, 64, 64, generator=torch.Generator().manual_seed(seed))


def letters(batch, length, seed=0):
    return torch.randint(0, VOCAB_SIZE, (batch, length), generator=torch.Generator().manual_seed(seed))


def build(cls, seed=0):
    torch.manual_seed(seed)
    return cls().eval()


def check_gradients(model, loss):
    model.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient (is part of the model disconnected?)"
        assert torch.isfinite(param.grad).all(), name
        if param.ndim >= 2:  # some biases legitimately get zero gradient (e.g. attention keys)
            assert param.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


# --- CNN encoder -------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_encoder_returns_a_square_feature_map(encoder_cls, batch):
    features = build(encoder_cls)(images(batch))
    assert features.ndim == 4
    b, channels, height, width = features.shape
    assert b == batch and channels >= 1
    assert height == width and 1 <= height <= 64
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_encoder_output_depends_on_the_image(encoder_cls):
    encoder = build(encoder_cls)
    assert not torch.allclose(encoder(images(2, seed=0)), encoder(images(2, seed=1)))


def test_encoder_samples_in_a_batch_are_independent(encoder_cls):
    encoder = build(encoder_cls)
    batch = images(3)
    changed = batch.clone()
    changed[1] = images(1, seed=5)[0]
    with torch.no_grad():
        before, after = encoder(batch), encoder(changed)
    torch.testing.assert_close(before[0], after[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(before[2], after[2], atol=1e-5, rtol=0)


def test_encoder_every_parameter_receives_a_gradient(encoder_cls):
    encoder = encoder_cls().train()
    check_gradients(encoder, encoder(images(4)).square().mean())


# --- full model --------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 14, MAX_LETTERS])
@pytest.mark.parametrize("batch", [1, 4])
def test_logits_shape(vlm_cls, batch, length):
    with torch.no_grad():
        logits = build(vlm_cls)(images(batch), letters(batch, length))
    assert logits.ndim == 3
    assert logits.shape[0] == batch
    assert logits.shape[1] >= length
    assert logits.shape[2] == VOCAB_SIZE
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_eval_mode_is_deterministic(vlm_cls):
    model = build(vlm_cls)
    with torch.no_grad():
        assert torch.equal(model(images(2), letters(2, 10)), model(images(2), letters(2, 10)))


def test_logits_depend_on_the_image(vlm_cls):
    model = build(vlm_cls)
    ids = letters(2, 10)
    with torch.no_grad():
        first, second = model(images(2, seed=0), ids), model(images(2, seed=1), ids)
    assert not torch.allclose(first, second, atol=1e-6), "the model ignores the image"


def test_logits_depend_on_the_previous_letters(vlm_cls):
    model = build(vlm_cls)
    with torch.no_grad():
        first, second = model(images(2), letters(2, 10, seed=0)), model(images(2), letters(2, 10, seed=1))
    assert not torch.allclose(first[:, -1], second[:, -1], atol=1e-6), "the model ignores the letters"


def test_letter_logits_do_not_see_future_letters(vlm_cls):
    model = build(vlm_cls)
    length, cut = 12, 6
    ids = letters(2, length)
    changed = ids.clone()
    changed[:, cut:] = letters(2, length - cut, seed=7)
    with torch.no_grad():
        before, after = model(images(2), ids), model(images(2), changed)
    torch.testing.assert_close(before[:, -length:][:, :cut], after[:, -length:][:, :cut], atol=1e-5, rtol=0)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-6)


def test_samples_in_a_batch_are_independent(vlm_cls):
    model = build(vlm_cls)
    batch, ids = images(3), letters(3, 10)
    changed_images, changed_ids = batch.clone(), ids.clone()
    changed_images[1], changed_ids[1] = images(1, seed=5)[0], letters(1, 10, seed=5)[0]
    with torch.no_grad():
        before, after = model(batch, ids), model(changed_images, changed_ids)
    torch.testing.assert_close(before[0], after[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(before[2], after[2], atol=1e-5, rtol=0)


def test_every_parameter_receives_a_gradient(vlm_cls):
    torch.manual_seed(0)
    model = vlm_cls().train()
    check_gradients(model, model(images(4), letters(4, 10)).square().mean())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_and_gpu_agree(vlm_cls):
    model = build(vlm_cls)
    batch, ids = images(2), letters(2, 10)
    with torch.no_grad():
        on_cpu = model(batch, ids)
        on_gpu = model.to("cuda")(batch.to("cuda"), ids.to("cuda")).cpu()
    torch.testing.assert_close(on_cpu, on_gpu, atol=1e-3, rtol=1e-3)
