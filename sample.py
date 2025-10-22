import os
import torch
import torch.nn.functional as F
from model import GPT
from prepare import encode, decode, init_tokenizer, vocab_size as get_vocab_size

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
        logits = model(idx_cond)
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
    vocab_size = init_tokenizer()
    print(f'Actual vocabulary size from text: {vocab_size}')
    
    # Set up device
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load the model
    model = GPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
    ).to(device)
    
    # Load the model and check its parameters
    state_dict = torch.load('model.pth', map_location=device)
    
    # Check vocabulary size in the model's embedding layer
    embedding_size = state_dict['token_embeddings.weight'].shape[0]
    print(f'Model vocabulary size: {embedding_size}')
    
    if embedding_size != vocab_size:
        print(f'WARNING: Model vocabulary size ({embedding_size}) differs from text vocabulary size ({vocab_size})')
        print('Adjusting model initialization to match trained model size...')
        vocab_size = embedding_size
        
    model.load_state_dict(state_dict)
    print('Model loaded successfully!')
    
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