import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Tuple, Optional

# --- RoPE (Rotary Positional Embeddings) Math ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency tensor for complex exponentials (cis = cos + i*sin)
    dim: The dimension of the head (d_model // num_heads)
    end: The maximum sequence length
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # complex64 is basically (cos(x), sin(x))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes freqs_cis to match the shape of the Q and K tensors for broadcasting.
    x is shape: (batch_size, seq_len, num_heads, head_dim)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Target shape: (1, seq_len, 1, head_dim)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotates the Query and Key vectors on the complex plane.
    """
    # Reshape (float) -> (complex)
    # [batch, seq, heads, dim] -> [batch, seq, heads, dim//2] where each item is complex(real, imag)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Rotate by multiplying complex numbers!
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- Core Layers ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadCausalAttention(nn.Module):
    """
    Standard Multi-Head Attention, modernized with RoPE support and KV Caching.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        batch_size, seq_len, d_model = x.size()
        
        # 1. Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Reshape for heads: (batch, seq, heads, dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 3. Apply Rotary Positional Embeddings
        if freqs_cis is not None:
            Q, K = apply_rotary_emb(Q, K, freqs_cis)
        
        # 4. KV Caching! If we are generating token-by-token, append to history
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)
            
        new_kv_cache = (K, V)
        
        # Ensure transpose for math: (batch, heads, seq, dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 5. Core Attention Math
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        
        if mask is not None:
            # Mask must be [1, 1, seq_len, seq_len] for broadcasting
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 6. Re-assemble heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(output), new_kv_cache
