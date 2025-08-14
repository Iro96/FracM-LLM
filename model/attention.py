import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, rotary=True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.rotary = rotary

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def apply_rotary(self, x):
        # Very simple rotary encoding stub (can improve later)
        seq_len = x.size(-2)
        freqs = torch.arange(0, self.head_dim, 2.0, device=x.device)
        freqs = 10000 ** (freqs / self.head_dim)
        t = torch.arange(seq_len, device=x.device)
        angles = torch.einsum("i,j->ij", t, freqs)

        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rot

    def forward(self, x, mask=None):
        B, T, D = x.size()
        qkv = self.qkv_proj(x)  # [B, T, 3 * D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary:
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, Hd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)
