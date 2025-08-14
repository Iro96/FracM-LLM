import torch

class LogicComposer:
    def __init__(self, model, tokenizer, memory=None, max_steps=3):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.max_steps = max_steps

    def analyze_prompt(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()

        if any(q in prompt_lower for q in ["how", "why", "explain"]):
            return {"type": "reasoning", "steps": 2}

        elif any(q in prompt_lower for q in ["who is", "what is", "remember", "recall"]):
            return {"type": "memory_lookup"}

        elif any(q in prompt_lower for q in ["plan", "steps", "strategy"]):
            return {"type": "planning", "steps": 3}

        return {"type": "direct"}

    def build_prompt(self, context: str, user_input: str) -> str:
        return f"{context}\nUser: {user_input}\nAssistant:"

    def generate_step(self, prompt: str, max_new_tokens=64):
        """Manual greedy decoding loop (no .generate() call)."""
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :]  # last token prediction
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Stop if EOS token is reached
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[len(prompt):].strip()

    def compose(self, user_input: str):
        strategy = self.analyze_prompt(user_input)
        context = ""

        # Step 1: Retrieve memory context if needed
        if self.memory and strategy["type"] in {"reasoning", "planning", "memory_lookup"}:
            context += self.memory.to_context_string()

        # Step 2: Build base prompt
        base_prompt = self.build_prompt(context, user_input)

        # Step 3: Multi-step reasoning if needed
        final_response = ""
        for step in range(strategy.get("steps", 1)):
            full_prompt = base_prompt + final_response
            step_response = self.generate_step(full_prompt)
            final_response += " " + step_response

        return final_response.strip()
