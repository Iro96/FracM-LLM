import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadSelfAttention
from model.utils import default

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1, rotary=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, rotary=rotary)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Attention with residual
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out

        # Feedforward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x
