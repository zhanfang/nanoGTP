import os
import numpy as np
import torch

from model import GPT
from prepare import init_tokenizer


class DataLoader:
    def __init__(self, dataset, batch_size, block_size):
        # Convert numpy memmap to PyTorch tensor
        self.dataset = torch.from_numpy(np.array(dataset)).long()
        self.batch_size = batch_size
        self.block_size = block_size
    
    def get_batch(self):
        # 生成一个随机批次
        ix = torch.randint(len(self.dataset) - self.block_size, (self.batch_size,))
        x = torch.stack([self.dataset[i : i + self.block_size] for i in ix])
        y = torch.stack([self.dataset[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y
    

def train():
    # 训练循环的占位符
    batch_size = 64
    block_size = 256

    n_embd = 384
    n_head = 6
    n_layer = 6
    learning_rate = 3e-4
    max_iters = 5000

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(device)
    
    data_dir = os.path.join('data', 'shakespeare')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    # 初始化词表并获取实际的词表大小
    vocab_size = init_tokenizer()
    print(f"Vocabulary size: {vocab_size}")
    
    model = GPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        xb, yb = DataLoader(train_data, batch_size, block_size).get_batch()
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        print(f"Iteration {iter}, loss: {loss.item()}")

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()