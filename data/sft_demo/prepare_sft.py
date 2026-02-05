import os
import json
import tiktoken
import numpy as np
import pickle

# 1. 配置
DATA_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(DATA_DIR, 'data.jsonl')
TRAIN_BIN = os.path.join(DATA_DIR, 'train.bin')
VAL_BIN = os.path.join(DATA_DIR, 'val.bin')
META_FILE = os.path.join(DATA_DIR, 'meta.pkl')

# Prompt Template
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def format_prompt(entry):
    instruction = entry.get('instruction', '')
    input_text = entry.get('input', '')
    
    # 构造 Prompt 部分
    if input_text:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)
    else:
        # 如果没有 input，我们可以调整模板，或者保留空 Input 占位
        # 这里为了简单，我们使用相同的模板，但 input 为空
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, input="")
    
    return prompt

def process_data():
    enc = tiktoken.get_encoding("gpt2")
    
    all_input_ids = []
    all_targets = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            
            prompt = format_prompt(entry)
            output = entry.get('output', '')
            
            # 编码
            prompt_ids = enc.encode(prompt)
            output_ids = enc.encode(output)
            eos_id = enc.eot_token # GPT-2 的 EOT token 通常作为 EOS
            
            # 构造 input_ids: [Prompt] + [Output] + [EOS]
            input_ids = prompt_ids + output_ids + [eos_id]
            
            # 构造 targets: [-1] * len(Prompt) + [Output] + [EOS]
            # 注意: 我们只对 Output 部分计算 Loss
            # prompt_ids 的对应的 target 是 -1
            targets = [-1] * len(prompt_ids) + output_ids + [eos_id]
            
            all_input_ids.extend(input_ids)
            all_targets.extend(targets)
    
    # 转换为 numpy 数组
    # 我们通常使用 uint16 (如果 vocab < 65535)，但 -1 需要 signed int
    # 所以这里 input_ids 用 uint16, targets 用 int16 (假设 -1 是合法的 ignore_index)
    # 但是 PyTorch CrossEntropyLoss 的 ignore_index 默认为 -100
    # 我们这里约定: 在 dataset get_item 时，如果读到特定值再转为 -100 或者直接存 int16
    # 为了简单，我们存 int32 以防万一，或者存 int16 (-1 是 0xFFFF 吗? 不，是 -1)
    
    # 修正: targets 中包含 -1，uint16 存不下。
    # 我们可以把 targets 中的 -1 存为 65535 (如果 vocab_size < 65535)，然后在读取时转回 -1。
    # 或者直接存 int32。考虑到数据集很小，直接存 int32 没问题。
    # 对于大规模训练，通常做法是只存 input_ids，然后在 DataCollator 中动态计算 Mask。
    # 但这里我们想在 bin 文件里就把 mask 做好。
    
    # 让我们使用 uint16 存储 input_ids (GPT-2 vocab 50257 < 65535)
    # 对于 targets，我们创建一个单独的 bin 文件，或者交错存储？
    # 为了复用 train.py，train.py 期望的是单一的 train.bin。
    # 原有的 train.py 逻辑是: x = data[i:i+block_size], y = data[i+1:i+1+block_size]
    # 这对于 SFT 是有问题的，因为 SFT 需要精确的 Mask。
    
    # **策略调整**:
    # 为了让 SFT 最简单地集成，我们不修改 train.py 的数据读取逻辑 (它只读一个 bin)。
    # 但是我们需要 mask。
    # 方案: 创建 `train.bin` (input_ids) 和 `train_mask.bin` (targets/mask)。
    # 然后修改 `finetune.py` (或 `train.py`) 去读取这两个文件。
    
    # 这里我们生成两个文件:
    # 1. input_ids (uint16)
    # 2. targets (int16) -> -1 会被正确存储
    
    print(f"Total tokens: {len(all_input_ids)}")
    
    # 划分训练集和验证集 (简单 90% / 10%)
    n = len(all_input_ids)
    split_idx = int(n * 0.9)
    
    train_ids = np.array(all_input_ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(all_input_ids[split_idx:], dtype=np.uint16)
    
    # 修正: GPT-2 vocab 50257，而 int16 范围是 -32768 到 32767
    # 所以 targets 不能用 int16 存储 (因为 50257 > 32767)
    # 我们改用 int32 存储 targets
    train_targets = np.array(all_targets[:split_idx], dtype=np.int32)
    val_targets = np.array(all_targets[split_idx:], dtype=np.int32)
    
    # 保存
    train_ids.tofile(os.path.join(DATA_DIR, 'train_ids.bin'))
    val_ids.tofile(os.path.join(DATA_DIR, 'val_ids.bin'))
    
    train_targets.tofile(os.path.join(DATA_DIR, 'train_targets.bin'))
    val_targets.tofile(os.path.join(DATA_DIR, 'val_targets.bin'))
    
    # 保存 meta
    meta = {
        'vocab_size': 50304, # 保持与 GPT-2 BPE 一致
        'eos_id': enc.eot_token
    }
    with open(META_FILE, 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    process_data()
