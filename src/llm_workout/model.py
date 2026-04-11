import torch
import torch.nn as nn
from .layers import RMSNorm, MultiHeadCausalAttention, SwiGLU

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attn = MultiHeadCausalAttention(d_model, num_heads)
        self.attn_norm = RMSNorm(d_model)
        
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Llama-style pre-norm configuration
        h = x + self.attn(self.attn_norm(x), mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class GPT(nn.Module):
    """
    A Decoder-only Language Model (GPT-style).
    Uses modernized features: RMSNorm and SwiGLU.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, hidden_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (optional but standard for standard GPTs, though Llama doesn't always tie)
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Token + Positional = Initial x
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Create causal mask ensuring i only attends to <= i
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift logits and targets so we predict the *next* token
            # CrossEntropyLoss expects (Batch*Seq, Vocab) and (Batch*Seq)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max seq len to avoid index out of bounds in pos embedding
            idx_cond = idx if idx.size(1) <= self.position_embedding.num_embeddings else idx[:, -self.position_embedding.num_embeddings:]
            logits, _ = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
