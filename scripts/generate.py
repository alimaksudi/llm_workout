"""
Load a checkpoint trained by `train_tiny.py` and stream generated text.

Usage:
    python scripts/generate.py                      # default prompt + 500 tokens
    python scripts/generate.py --prompt "ROMEO:" --max-new-tokens 300
"""
import argparse

import torch

from llm_workout.model import GPT

CHECKPOINT_PATH = "checkpoints/tiny_shakespeare.pt"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to the .pt checkpoint")
    parser.add_argument("--prompt", default="\n", help="Seed text to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Number of tokens to generate")
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {args.checkpoint} onto {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)

    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Encode the prompt (fall back to a newline if a char isn't in the vocab).
    seed = [stoi.get(c, stoi.get("\n", 0)) for c in args.prompt] or [stoi.get("\n", 0)]
    context = torch.tensor([seed], dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)
    for token_id in model.generate(context, max_new_tokens=args.max_new_tokens):
        print(decode([token_id]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
