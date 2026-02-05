import os
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from prepare import encode, decode, init_tokenizer

def generate(model, start_text, max_new_tokens=100, temperature=1.0, block_size=256, device='cpu'):
    """Generate text starting from start_text."""
    model.eval()
    
    # Encode the start text
    idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    
    # Generate new tokens
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # Forward pass through the model
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # Take last time step and apply temperature
        probs = F.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return decode(idx[0].tolist())

def main():
    # Model parameters (should match training)
    n_embd = 384
    n_head = 6
    n_layer = 6
    block_size = 256
    
    # Initialize tokenizer and get vocabulary size
    raw_vocab_size = init_tokenizer()
    # Padded vocab size used in training
    vocab_size = 50304
    print(f'Actual vocabulary size from tokenizer: {raw_vocab_size}')
    print(f'Model vocabulary size (padded): {vocab_size}')
    
    # Set up device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load the model
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0.0)
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    
    # Load the model and check its parameters
    try:
        state_dict = torch.load('model.pth', map_location=device)
        # Check vocabulary size in the model's embedding layer
        # Note: state_dict keys might have changed due to model refactor?
        # Let's check keys first. Old model keys were like 'token_embeddings.weight'
        # New model keys are 'transformer.wte.weight'
        # We need to handle this migration if loading old model.pth
        
        # But wait, if user runs train.py now, it will save new format.
        # If user has old model.pth, it will fail.
        # Let's assume we are forward looking.
        
        if 'token_embeddings.weight' in state_dict:
            print("Detected old model format. Please re-train or use a migration script.")
            # Simple migration for demo purposes if possible, but structure changed significantly?
            # Old: self.token_embeddings = nn.Embedding
            # New: self.transformer.wte = nn.Embedding
            # Let's try to remap keys on the fly
            new_sd = {}
            for k, v in state_dict.items():
                if k == 'token_embeddings.weight': new_sd['transformer.wte.weight'] = v
                elif k == 'position_embeddings.weight': new_sd['transformer.wpe.weight'] = v
                elif k == 'ln_f.weight': new_sd['transformer.ln_f.weight'] = v
                elif k == 'ln_f.bias': new_sd['transformer.ln_f.bias'] = v
                elif k.startswith('blocks.'):
                    # blocks.0.sa.c_attn.weight -> transformer.h.0.attn.c_attn.weight
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
            print("Attempted to migrate keys to new format.")

        embedding_size = state_dict['transformer.wte.weight'].shape[0]
        print(f'Checkpoint vocabulary size: {embedding_size}')
        
        if embedding_size != vocab_size:
            print(f'WARNING: Checkpoint vocab size ({embedding_size}) != Model config vocab size ({vocab_size})')
            
        model.load_state_dict(state_dict)
        print('Model loaded successfully!')
    except FileNotFoundError:
        print("Model file not found. Using initialized weights (random output).")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Generate text
    while True:
        try:
            start_text = input('\nEnter some starting text (or press Ctrl+C to exit): ')
            max_tokens = int(input('How many tokens to generate? '))
            temperature = float(input('Enter temperature (0.0-1.0): '))
            
            generated_text = generate(
                model,
                start_text,
                max_new_tokens=max_tokens,
                temperature=temperature,
                block_size=block_size,
                device=device
            )
            print('\nGenerated text:')
            print('='*80)
            print(generated_text)
            print('='*80)
            
        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except Exception as e:
            print(f'Error: {e}')

if __name__ == '__main__':
    main()