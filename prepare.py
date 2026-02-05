import os
import numpy as np
import torch
import tiktoken

# Global tokenizer instance
enc = None

def init_tokenizer():
    """初始化分词器 (GPT-2 BPE)"""
    global enc
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    # GPT-2 vocab size is 50257
    return enc.n_vocab

def encode(s):
    """将文本编码为token ids"""
    if enc is None:
        init_tokenizer()
    return enc.encode(s)

def decode(l):
    """将token ids解码为文本"""
    if enc is None:
        init_tokenizer()
    return enc.decode(l)

def prepare_data():
    """准备训练数据"""
    # 初始化分词器
    init_tokenizer()
    
    # 读取文本
    input_file = "data/shakespeare/input.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please download it first.")
        return

    with open(input_file, 'r') as f:
        text = f.read()
    
    print("Encoding data with GPT-2 BPE...")
    ids = encode(text)
    print(f"Total tokens: {len(ids)}")

    # 编码并保存为二进制文件
    # GPT-2 vocab size is 50257, which fits in uint16 (0-65535)
    data = np.array(ids, dtype=np.uint16)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    train_data.tofile('data/shakespeare/train.bin')
    val_data.tofile('data/shakespeare/val.bin')
    
    return enc.n_vocab

if __name__ == "__main__":
    prepare_data()