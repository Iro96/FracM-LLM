import torch
from model.model import FractalModel
from model.config import FRMConfig
from model.memory import FRMMemory
from model.logic_composer import LogicComposer
from transformers import GPT2Tokenizer

# === Load tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# === Load model ===
config = FRMConfig()
config.vocab_size = tokenizer.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FractalModel(config).to(device)
model.load_state_dict(torch.load("frm.pth", map_location=device))
model.eval()

# === Memory & Composer ===
memory = FRMMemory()
composer = LogicComposer(model=model, tokenizer=tokenizer, memory=memory)

# === Chat loop ===
print("🤖 Fractal Reasoning Model Ready. Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # Get model response via LogicComposer
    reply = composer.compose(user_input)

    # Store user tokens in short-term memory
    tokens = tokenizer.encode(user_input)
    memory.update_short(tokens)

    print("Bot:", reply)
