import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes the scaled dot-product attention.
    
    Equation: Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) * V
    
    Args:
        query (torch.Tensor): Tensor of shape (..., Seq_Q, D_k)
        key (torch.Tensor): Tensor of shape (..., Seq_K, D_k)
        value (torch.Tensor): Tensor of shape (..., Seq_K, D_v)
        mask (torch.Tensor, optional): Mask to apply to the attention scores.
        
    Returns:
        output (torch.Tensor): The attention-weighted sum of values.
        attention_weights (torch.Tensor): The probability distribution over tokens.
    """
    # 1. Get dimension of key (dk)
    dk = query.size(-1)
    
    # 2. Compute Attention Scores (Similiarity)
    # Q * K^T -> (..., Seq_Q, Seq_K)
    # transposed key changes (..., Seq_K, D_k) -> (..., D_k, Seq_K)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(dk)
    
    # 3. Apply optional mask (e.g., for causal decoder)
    if mask is not None:
        # Fill masked positions with a very small number so softmax makes them 0
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 4. Softmax to turn scores into probabilities (rows sum to 1)
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. Weighted Sum of Values
    # (..., Seq_Q, Seq_K) * (..., Seq_K, D_v) -> (..., Seq_Q, D_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

def dot_product_similarity(a, b):
    """
    Simple dot product between two vectors to show similarity.
    Math: sum(a_i * b_i)
    """
    return torch.dot(a, b)
