import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model.model import FractalModel
from model.config import FRMConfig
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from transformers import GPT2Tokenizer

# load the tokenizer that was saved by GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")

# set pad token (GPT-2 has no pad by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token     # common choice
    tokenizer.save_pretrained("./tokenizer")     # persist pad_token change

print("type:", type(tokenizer))
print("vocab_size:", tokenizer.vocab_size)
print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)


# === Dataset ===
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=block_size)
            self.examples.append(torch.tensor(tokens))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
config = FRMConfig(vocab_size=vocab_size)
model = FractalModel(config).to(device)

# Dummy text data
texts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum physics in simple terms.",
    "Why do we dream?"
]
dataset = TextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

# === Training Loop ===
epochs = 50
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "frm.pth")
print("✅ Model saved as frm.pth")
exit()