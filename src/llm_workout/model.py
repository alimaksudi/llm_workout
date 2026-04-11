import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .layers import RMSNorm, MultiHeadCausalAttention, SwiGLU, precompute_freqs_cis

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attn = MultiHeadCausalAttention(d_model, num_heads)
        self.attn_norm = RMSNorm(d_model)
        
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.ffn_norm = RMSNorm(d_model)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Apply pre-norm attention
        attn_out, new_kv_cache = self.attn(self.attn_norm(x), mask=mask, freqs_cis=freqs_cis, kv_cache=kv_cache)
        h = x + attn_out
        
        # Apply FFN
        out = h + self.ffn(self.ffn_norm(h))
        return out, new_kv_cache

class GPT(nn.Module):
    """
    A Decoder-only Language Model (Llama-style).
    Uses RMSNorm, SwiGLU, RoPE, and supports KV Caching.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, hidden_dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Precompute RoPE frequencies for the entire max sequence length
        freqs_cis = precompute_freqs_cis(d_model // num_heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ):
        B, T = idx.size()
        
        # 1. Lookup Embeddings (No absolute positional embeddings added!)
        x = self.token_embedding(idx)
        
        # 2. Slice out only the RoPE frequencies we need for this forward pass length
        freqs_cis = self.freqs_cis[start_pos : start_pos + T] # type: ignore
        
        # 3. Create causal mask. 
        # Crucially, if we are using the Cache (T==1), we don't need a mask because 1 token can't look into its own future!
        mask = None
        if T > 1:
            mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
            # If start_pos > 0, we must pad the mask to account for history we are attending to.
            if start_pos > 0:
                mask = torch.cat([
                    torch.ones(1, 1, T, start_pos, device=idx.device),
                    mask
                ], dim=-1)
                
        # 4. Forward through layers, collecting updated cache states
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, mask=mask, freqs_cis=freqs_cis, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Extremely fast generation utilizing the KV Cache.
        """
        self.eval()
        B, T = idx.size()
        
        kv_caches = None
        start_pos = 0
        
        for _ in range(max_new_tokens):
            # Pre-fill phase or normal token forward
            logits, _, kv_caches = self(idx, start_pos=start_pos, kv_caches=kv_caches)
            
            # Predict
            logits = logits[:, -1, :] # The very last token predicts the next
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Update start_pos so the mathematical ROPE sequence advances
            start_pos += idx.size(1)
            
            # The ONLY thing we feed back into the network is the single generated token!
            idx = idx_next 
            
            # To return the full string, we should yield or append. 
            # For simplicity in this demo, since `idx` is just 1 token now, we'll store it.
            # But the caller usually wants the whole string. Let's just yield tokens one by one
            # and let the inference wrapper join them. Or we can just build a list.
            yield idx_next.item()
            
        self.train()
