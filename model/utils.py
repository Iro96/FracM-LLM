import torch

def generate_causal_mask(seq_len: int) -> torch.Tensor:
    # Create a lower-triangular mask
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
