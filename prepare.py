import os
import numpy as np
import torch

# 词表和编码/解码函数
chars = None
stoi = None
itos = None
vocab_size = None

def init_tokenizer():
    """初始化分词器，加载词表"""
    global chars, stoi, itos, vocab_size
    
    # 读取莎士比亚文本
    input_file = "data/shakespeare/input.txt"
    with open(input_file, 'r') as f:
        text = f.read()

    # 创建字符级词表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # 65 (ASCII字符)

    # 创建编码/解码器
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    return vocab_size

def encode(s):
    """将文本编码为token ids"""
    if stoi is None:
        init_tokenizer()
    return [stoi[c] for c in s]

def decode(l):
    """将token ids解码为文本"""
    if itos is None:
        init_tokenizer()
    return ''.join([itos[i] for i in l])

def prepare_data():
    """准备训练数据"""
    # 初始化分词器
    init_tokenizer()
    
    # 读取文本
    input_file = "data/shakespeare/input.txt"
    with open(input_file, 'r') as f:
        text = f.read()
    
    # 编码并保存为二进制文件
    data = np.array(encode(text), dtype=np.uint16)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    train_data.tofile('data/shakespeare/train.bin')
    val_data.tofile('data/shakespeare/val.bin')
    
    return vocab_size

if __name__ == "__main__":
    prepare_data()