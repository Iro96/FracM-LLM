import torch
import torch.nn as nn
from model.embedding import Embedding
from model.transformer_block import TransformerBlock
from model.config import FRMConfig
from model.utils import generate_causal_mask

class FractalModel(nn.Module):
    def __init__(self, config: FRMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Embedding layer (token + rotary/positional)
        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            rotary=config.rotary
        )

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                rotary=config.rotary
            )
            for _ in range(config.n_layers)
        ])

        # Optional: Hooks for future modules
        self.logic_composer = None  # Placeholder
        self.memory_module = None   # Placeholder

        if config.use_memory:
            from model.memory import FRMMemory
            self.memory_module = FRMMemory()

        # Final output head
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)


    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch, seq_len) - token ids
            mask: optional attention mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape

        if mask is None:
            mask = generate_causal_mask(T).to(input_ids.device)
        
        if self.memory_module:
            memory_tokens = self.memory_module.get_short()
            # prepend memory_tokens to input_ids (if space allows)

        x = self.embedding(input_ids)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.output(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, pad_token_id=50256):
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)  # (B, T, vocab_size)
            next_token_logits = logits[:, -1, :]  # last token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # If model keeps generating pad/end tokens, stop early
            if next_token.item() == pad_token_id:
                break

            generated = torch.cat((generated, next_token), dim=1)

        return generated
