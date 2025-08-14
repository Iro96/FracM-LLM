if __name__ == "__main__":
    import torch
    from model.config import FRMConfig
    from model.model import FractalModel

    config = FRMConfig(
        vocab_size=32000,
        max_seq_len=512,
        d_model=512,
        n_layers=6,
        n_heads=8,
    )

    model = FractalModel(config)
    dummy_input = torch.randint(0, config.vocab_size, (1, 128))  # (batch, seq_len)
    logits = model(dummy_input)  # (1, 128, vocab_size)
    print("Logits shape:", logits.shape)
