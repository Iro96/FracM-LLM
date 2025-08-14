import torch
from model.config import FRMConfig
from model.model import FractalModel
from model.memory import FRMMemory
from model.logic_composer import LogicComposer
from transformers import GPT2Tokenizer

def chat():
    # === Load tokenizer ===
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Model setup ===
    config = FRMConfig(vocab_size=tokenizer.vocab_size)
    model = FractalModel(config)
    model.load_state_dict(torch.load("frm.pth", map_location="cpu"))
    model.eval()

    memory = FRMMemory()
    composer = LogicComposer(model, tokenizer, memory)

    print("🤖 Fractal Reasoning Model (FRM) Ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # Generate response
        output = composer.compose(user_input)

        # Update short-term memory
        tokens = tokenizer.encode(user_input)
        memory.update_short(tokens)

        print("FRM:", output)

if __name__ == "__main__":
    chat()
