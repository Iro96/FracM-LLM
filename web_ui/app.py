import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from flask import Flask, request, jsonify, render_template
from model.model import FractalModel
from model.memory import FRMMemory
from model.logic_composer import LogicComposer
from model.config import FRMConfig
from transformers import GPT2Tokenizer

app = Flask(__name__)

# ✅ Load tokenizer from your saved folder
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# ✅ Match vocab size to tokenizer
config = FRMConfig()
config.vocab_size = tokenizer.vocab_size

# ✅ Load model
model = FractalModel(config)
model.load_state_dict(torch.load("frm.pth", map_location="cpu"))  # or "cuda" if available
model.eval()

memory = FRMMemory()
composer = LogicComposer(model=model, tokenizer=tokenizer, memory=memory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')
    response = composer.compose(user_msg)
    tokens = tokenizer.encode(user_msg)
    memory.update_short(tokens)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from flask import Flask, request, jsonify, render_template
from model.model import FractalModel
from model.memory import FRMMemory
from model.logic_composer import LogicComposer
from model.config import FRMConfig
from transformers import GPT2Tokenizer

app = Flask(__name__)

# ✅ Load tokenizer from your saved folder
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# ✅ Match vocab size to tokenizer
config = FRMConfig()
config.vocab_size = tokenizer.vocab_size

# ✅ Load model
model = FractalModel(config)
model.load_state_dict(torch.load("frm.pth", map_location="cpu"))  # or "cuda" if available
model.eval()

memory = FRMMemory()
composer = LogicComposer(model=model, tokenizer=tokenizer, memory=memory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')
    response = composer.compose(user_msg)
    tokens = tokenizer.encode(user_msg)
    memory.update_short(tokens)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
