from flask import Flask, render_template, request, jsonify
import torch
from model import GPT, GPTConfig
from prepare import encode, decode, init_tokenizer
import torch.nn.functional as F

app = Flask(__name__)

# Load model globally
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
raw_vocab_size = init_tokenizer()
vocab_size = 50304 # Padded size
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0.0)
config = GPTConfig(**model_args)
model = GPT(config).to(device)

try:
    state_dict = torch.load('model.pth', map_location=device)
    # Handle migration if needed (same as sample.py)
    if 'token_embeddings.weight' in state_dict:
         print("Detected old model format. Attempting migration...")
         new_sd = {}
         for k, v in state_dict.items():
             if k == 'token_embeddings.weight': new_sd['transformer.wte.weight'] = v
             elif k == 'position_embeddings.weight': new_sd['transformer.wpe.weight'] = v
             elif k == 'ln_f.weight': new_sd['transformer.ln_f.weight'] = v
             elif k == 'ln_f.bias': new_sd['transformer.ln_f.bias'] = v
             elif k.startswith('blocks.'):
                 k_new = k.replace('blocks.', 'transformer.h.')
                 k_new = k_new.replace('.sa.', '.attn.')
                 k_new = k_new.replace('.ffwd.net.0.', '.mlp.c_fc.')
                 k_new = k_new.replace('.ffwd.net.2.', '.mlp.c_proj.')
                 k_new = k_new.replace('.ln1.', '.ln_1.')
                 k_new = k_new.replace('.ln2.', '.ln_2.')
                 new_sd[k_new] = v
             elif k == 'lm_head.weight': new_sd['lm_head.weight'] = v
             else:
                 new_sd[k] = v
         state_dict = new_sd

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running with initialized weights (random output)")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    start_text = data.get('prompt', 'ROMEO:')
    max_tokens = int(data.get('max_tokens', 100))
    temperature = float(data.get('temperature', 0.8))
    
    # Generate logic (adapted from sample.py)
    idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    
    generated_text = decode(idx[0].tolist())
    # Return only the new part? For now, return full text
    return jsonify({'result': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
