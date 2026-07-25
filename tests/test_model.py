import torch

from llm_workout.model import GPT


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(
        vocab_size=37,
        d_model=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        max_seq_len=32,
    )


def test_forward_shapes():
    model = _tiny_model()
    idx = torch.randint(0, 37, (2, 8))
    logits, loss, caches = model(idx)
    assert logits.shape == (2, 8, 37)
    assert loss is None
    assert len(caches) == 2  # one cache per layer


def test_loss_is_computed_with_targets():
    model = _tiny_model()
    idx = torch.randint(0, 37, (2, 8))
    targets = torch.randint(0, 37, (2, 8))
    _, loss, _ = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0


def test_weight_tying():
    """Token embedding and the LM head must share the same weight tensor."""
    model = _tiny_model()
    assert model.token_embedding.weight is model.lm_head.weight


def test_kv_cache_matches_full_forward():
    """
    The most important invariant: incremental generation with the KV cache must
    produce the *same* logits as a single full-sequence forward pass.
    """
    model = _tiny_model()
    model.eval()
    seq = torch.randint(0, 37, (1, 10))

    # Reference: one full forward pass over the whole sequence.
    with torch.no_grad():
        full_logits, _, _ = model(seq)

    # Incremental: feed one token at a time, carrying the cache forward.
    caches = None
    start_pos = 0
    incremental = []
    with torch.no_grad():
        for t in range(seq.size(1)):
            tok = seq[:, t : t + 1]
            logits, _, caches = model(tok, start_pos=start_pos, kv_caches=caches)
            incremental.append(logits[:, -1, :])
            start_pos += 1

    incremental_logits = torch.stack(incremental, dim=1)
    assert incremental_logits.shape == full_logits.shape
    assert torch.allclose(full_logits, incremental_logits, atol=1e-4), (
        "KV-cache path diverged from the full forward pass"
    )


def test_generate_runs_and_respects_length():
    model = _tiny_model()
    context = torch.zeros((1, 1), dtype=torch.long)
    tokens = list(model.generate(context, max_new_tokens=5))
    assert len(tokens) == 5
    assert all(0 <= t < 37 for t in tokens)


def test_exceeding_max_seq_len_raises_clear_error():
    """A sequence longer than max_seq_len must fail with a helpful message,
    not an opaque RoPE broadcast assertion."""
    import pytest

    model = _tiny_model()  # max_seq_len=32
    too_long = torch.zeros((1, 33), dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)


def test_generation_beyond_max_seq_len_raises():
    """Prompt + generated tokens must fit within max_seq_len; overrunning during
    cached generation raises the same clear error."""
    import pytest

    model = _tiny_model()  # max_seq_len=32
    context = torch.zeros((1, 30), dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        # 30-token prompt + 5 new tokens reaches position 35 > 32
        list(model.generate(context, max_new_tokens=5))
