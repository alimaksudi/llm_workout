import torch

from llm_workout.layers import (
    RMSNorm,
    SwiGLU,
    MultiHeadCausalAttention,
    precompute_freqs_cis,
    apply_rotary_emb,
)


def test_rmsnorm_shape_and_unit_scale():
    """With weight=1, RMSNorm output should have ~unit root-mean-square."""
    norm = RMSNorm(dim=16)
    x = torch.randn(4, 8, 16) * 5.0
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_shape_preserved():
    ffn = SwiGLU(d_model=32, hidden_dim=64)
    x = torch.randn(2, 5, 32)
    assert ffn(x).shape == x.shape


def test_rope_preserves_shape_and_norm():
    """Rotation must not change vector magnitude (it only rotates)."""
    head_dim, seq_len = 8, 6
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    q = torch.randn(1, seq_len, 2, head_dim)
    k = torch.randn(1, seq_len, 2, head_dim)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_attention_output_shape_and_cache():
    attn = MultiHeadCausalAttention(d_model=32, num_heads=4)
    x = torch.randn(2, 7, 32)
    out, (k_cache, v_cache) = attn(x)
    assert out.shape == x.shape
    # Cache is stored as (batch, seq, heads, head_dim)
    assert k_cache.shape == (2, 7, 4, 8)
    assert v_cache.shape == (2, 7, 4, 8)


def test_decoder_block_equals_manual_prenorm_composition():
    """Guard the capstone's traceability claim (notebook 14, Section 3½).

    The library's ``DecoderBlock`` must be *exactly* the pre-norm residual
    composition of the pieces taught module by module — causal attention (3.1),
    RMSNorm (4.1), and SwiGLU (4.1) — wired as::

        h = x + attn(attn_norm(x))
        out = h + ffn(ffn_norm(h))

    If a future refactor changes that wiring (post-norm, a dropped residual, a
    reordered sublayer), this test fails — so "the model you import is the model
    you assembled by hand" can never silently become false.
    """
    from llm_workout.model import DecoderBlock

    torch.manual_seed(0)
    d_model, num_heads, hidden_dim, seq_len = 32, 4, 64, 9
    block = DecoderBlock(d_model, num_heads, hidden_dim)

    # Rebuild the block from the individual library layers, sharing its weights.
    attn = MultiHeadCausalAttention(d_model, num_heads)
    attn_norm = RMSNorm(d_model)
    ffn = SwiGLU(d_model, hidden_dim)
    ffn_norm = RMSNorm(d_model)
    attn.load_state_dict(block.attn.state_dict())
    attn_norm.load_state_dict(block.attn_norm.state_dict())
    ffn.load_state_dict(block.ffn.state_dict())
    ffn_norm.load_state_dict(block.ffn_norm.state_dict())

    x = torch.randn(2, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    freqs_cis = precompute_freqs_cis(d_model // num_heads, seq_len)

    lib_out, _ = block(x, mask=mask, freqs_cis=freqs_cis)

    attn_out, _ = attn(attn_norm(x), mask=mask, freqs_cis=freqs_cis)
    h = x + attn_out
    manual_out = h + ffn(ffn_norm(h))

    assert torch.allclose(lib_out, manual_out, atol=1e-6)
