from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Save full tokenizer (including merges and vocab) into tokenizer.json
tokenizer.save_pretrained("./tokenizer")
print("Tokenizer saved to ./tokenizer")
