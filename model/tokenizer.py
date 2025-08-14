from __future__ import annotations
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
