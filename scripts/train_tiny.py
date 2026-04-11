import os
import torch
import urllib.request
from llm_workout.model import GPT

# --- Hyperparameters ---
batch_size = 32
block_size = 64  # Context length
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
eval_iters = 50
n_embd = 128
n_head = 4
n_layer = 4
hidden_dim = n_embd * 4

# Detect ideal device (MPS for Apple Silicon, CUDA for NVIDIA, fallback CPU)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Training on: {device}")

# --- Data Loading ---
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "tinyshakespeare.txt"

if not os.path.exists(data_path):
    print("Downloading TinyShakespeare dataset...")
    urllib.request.urlretrieve(data_url, data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
# --- Character Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Dataset length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

# --- Train/Val Split ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Initialization ---
model = GPT(
    vocab_size=vocab_size,
    d_model=n_embd,
    num_layers=n_layer,
    num_heads=n_head,
    hidden_dim=hidden_dim,
    max_seq_len=block_size
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")

# --- Training Loop ---
print("Starting training loop...")
for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Output a sample generation!
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = decode(model.generate(context, max_new_tokens=100)[0].tolist())
        print(f"\\n--- Generation at Step {iter} ---\\n{generated}\\n--- End ---\\n")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass and calculate loss
    logits, loss = model(xb, yb)
    
    # Backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished.")
