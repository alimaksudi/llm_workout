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
