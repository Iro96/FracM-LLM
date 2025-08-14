from __future__ import annotations

class FRMMemory:
    def __init__(self, max_short_memory=512, enable_long_memory=True):
        self.short_memory = []  # in-context memory: past tokens
        self.long_memory = {}   # fact memory: key-value store
        self.max_short_memory = max_short_memory
        self.enable_long_memory = enable_long_memory

    def clear(self):
        self.short_memory = []
        if self.enable_long_memory:
            self.long_memory.clear()

    def update_short(self, tokens: list[int]):
        self.short_memory.extend(tokens)
        if len(self.short_memory) > self.max_short_memory:
            self.short_memory = self.short_memory[-self.max_short_memory:]

    def get_short(self):
        return self.short_memory

    def add_fact(self, key: str, value: str):
        if not self.enable_long_memory:
            return
        self.long_memory[key.lower()] = value

    def query_fact(self, query: str, fuzzy=False):
        query = query.lower()
        if query in self.long_memory:
            return self.long_memory[query]
        if fuzzy:
            # Return best match
            for k in self.long_memory:
                if query in k:
                    return self.long_memory[k]
        return None

    def to_context_string(self):
        context = ""
        if self.enable_long_memory and self.long_memory:
            for k, v in self.long_memory.items():
                context += f"[{k}]: {v}\n"
        return context
