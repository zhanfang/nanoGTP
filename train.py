import os
import numpy as np
import torch

from model import GPT, GPTConfig
from prepare import init_tokenizer


class DataLoader:
    def __init__(self, dataset, batch_size, block_size):
        # dataset should be a torch tensor already loaded in memory
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
    
    def get_batch(self):
        # 生成一个随机批次
        ix = torch.randint(len(self.dataset) - self.block_size, (self.batch_size,))
        x = torch.stack([self.dataset[i : i + self.block_size] for i in ix])
        y = torch.stack([self.dataset[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y



# -----------------------------------------------------------------------------
# default config values
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
learning_rate = 3e-4
max_iters = 5000
eval_interval = 100
log_interval = 10
always_save_checkpoint = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

def train():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = os.path.join('data', 'shakespeare')
    # Use memmap to read file, but convert to tensor once efficiently
    train_data_memmap = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    train_data = torch.from_numpy(np.array(train_data_memmap)).long()

    # 初始化词表并获取实际的词表大小
    # GPT-2 vocab_size is 50257, pad to 50304 for efficiency
    raw_vocab_size = init_tokenizer()
    vocab_size = 50304
    print(f"Vocabulary size: {vocab_size} (padded from {raw_vocab_size})")
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=vocab_size, dropout=0.0)
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)

    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
    
    # Create DataLoader instance once
    train_loader = DataLoader(train_data, batch_size, block_size)

    for iter in range(max_iters):
        xb, yb = train_loader.get_batch()
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if iter % log_interval == 0:
            print(f"Iteration {iter}, loss: {loss.item()}")

    print("Saving model to model.pth")
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()