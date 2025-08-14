class FRMConfig:
    def __init__(
        self,
        vocab_size=32000,
        max_seq_len=4096,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dropout=0.1,
        ff_mult=4,
        rotary=True,
        use_memory=True,
        use_logic_composer=True,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.rotary = rotary
        self.use_memory = use_memory
        self.use_logic_composer = use_logic_composer
