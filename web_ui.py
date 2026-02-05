from flask import Flask, render_template, request, jsonify
import torch
import os
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

def load_model(path, model_name="Model"):
    try:
        if not os.path.exists(path):
            print(f"{model_name} not found at {path}")
            return None
            
        print(f"Loading {model_name} from {path}...")
        
        # Load checkpoint first to check for config
        checkpoint = torch.load(path, map_location=device)
        
        # Determine model configuration
        current_args = model_args.copy() # Default to Nano config
        state_dict = checkpoint
        
        # Handle checkpoints from finetune.py which contain 'model_args' and 'model'
        if isinstance(checkpoint, dict) and 'model_args' in checkpoint:
            print(f"Found model_args in checkpoint for {model_name}: {checkpoint['model_args']}")
            # Update defaults with checkpoint args, ensuring required keys exist
            for k, v in checkpoint['model_args'].items():
                if k in current_args:
                    current_args[k] = v
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
             # Case where 'model' key exists but 'model_args' might be missing (unlikely for finetune.py but possible)
             state_dict = checkpoint['model']

        config = GPTConfig(**current_args)
        model = GPT(config).to(device)
            
        # Handle migration if needed
        if 'token_embeddings.weight' in state_dict:
             print(f"Detected old format for {model_name}. Migrating...")
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
             
        # Fix keys from torch.compile or DDP (start with _orig_mod.)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval()
        print(f"{model_name} loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

# Load models
models = {}
models['base'] = load_model('model.pth', "Base Model")
models['sft'] = load_model('out-sft/ckpt.pt', "SFT Model")

# Fallback if no models
if not models['base'] and not models['sft']:
    print("No models found! Initializing random model for testing.")
    config = GPTConfig(**model_args)
    models['base'] = GPT(config).to(device)

@app.route('/')
def home():
    return render_template('index.html', has_sft=(models['sft'] is not None))

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    start_text = data.get('prompt', 'ROMEO:')
    max_tokens = int(data.get('max_tokens', 100))
    temperature = float(data.get('temperature', 0.8))
    model_type = data.get('model', 'base')
    
    idx_start = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    
    results = {}
    
    target_models = []
    if model_type == 'compare':
        if models['base']: target_models.append(('base', models['base']))
        if models['sft']: target_models.append(('sft', models['sft']))
    else:
        m = models.get(model_type) or models.get('base')
        if m: target_models.append((model_type, m))
    
    with torch.no_grad():
        for name, model in target_models:
            idx = idx_start.clone()
            for _ in range(max_tokens):
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            results[name] = decode(idx[0].tolist())

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
