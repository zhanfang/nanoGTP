"""
GPT model implementation based on the transformer architecture.
"""

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)  # (B,T,T)
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super(Block, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size):
        super(GPT, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embeddings(idx)  # (B,T,n_embd)
        pos_emb = self.position_embeddings(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)
