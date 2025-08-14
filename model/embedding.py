import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=4096, rotary=False):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.rotary = rotary
        if not rotary:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        tok_embed = self.token_embed(x)
        if self.rotary:
            return tok_embed  # Position encoding will be applied in attention layer
        else:
            seq_len = x.size(1)
            return tok_embed + self.pos_embed[:, :seq_len, :]
