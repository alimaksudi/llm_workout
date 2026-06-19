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

        # Guard: RoPE frequencies are precomputed up to max_seq_len. The position
        # we reach is start_pos + T (start_pos > 0 during cached generation), and it
        # must fit inside that buffer. Without this check an over-long sequence slices
        # a too-short freqs_cis and fails with an opaque broadcast assertion deep in
        # apply_rotary_emb. max_seq_len must cover prompt length + tokens generated.
        max_pos = self.freqs_cis.size(0)  # type: ignore
        if start_pos + T > max_pos:
            raise ValueError(
                f"Sequence reaches position {start_pos + T}, which exceeds the model's "
                f"max_seq_len ({max_pos}). Construct GPT with a larger max_seq_len: it must "
                f"cover the prompt length plus the number of tokens you intend to generate."
            )

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
        Autoregressively generate tokens, yielding one token id at a time.

        Uses the KV cache: the prompt is processed once (the prefill), then only
        the single most recently generated token is fed back on each step.
        """
        was_training = self.training
        self.eval()

        kv_caches = None
        start_pos = 0

        try:
            for _ in range(max_new_tokens):
                # Prefill on the first pass (full prompt), then one token per step.
                logits, _, kv_caches = self(idx, start_pos=start_pos, kv_caches=kv_caches)

                # Sample the next token from the final position's distribution.
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Advance the RoPE position by the number of tokens we just processed.
                start_pos += idx.size(1)

                # Feed only the newly generated token back into the network.
                idx = idx_next

                yield idx_next.item()
        finally:
            # Restore the caller's original train/eval mode.
            self.train(was_training)
