# SFT (Supervised Fine-Tuning) 设计方案

Supervised Fine-Tuning (SFT) 是将预训练模型转化为指令跟随 (Instruction Following) 模型的关键步骤。

## 1. 核心差异 (Pre-training vs SFT)

| 特性 | Pre-training (预训练) | SFT (监督微调) |
| :--- | :--- | :--- |
| **目标** | Next Token Prediction (文本接龙) | Instruction Following (听懂指令) |
| **数据** | 海量非结构化文本 (Books, Web) | 结构化指令对 (Instruction, Input, Output) |
| **关注点** | 学习语法、知识、逻辑 | 学习对话格式、对齐人类意图 |
| **Loss** | 计算所有 Token 的 Loss | **只计算 Output 部分的 Loss** (Loss Masking) |

## 2. 数据格式设计

我们需要从纯文本转变为结构化的 JSONL 格式。

### 源数据格式 (JSONL)
```json
{"instruction": "解释什么是人工智能。", "input": "", "output": "人工智能(AI)是指由计算机系统模拟的人类智能..."}
{"instruction": "将以下句子翻译成英文。", "input": "你好，世界。", "output": "Hello, world."}
```

### Prompt 模板
为了让模型理解输入的结构，我们需要将 JSON 展平成字符串：
```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```
*(如果没有 input，则省略 Input 部分)*

## 3. Loss Masking (关键技术)

在 SFT 中，我们不希望模型去学习如何生成“Instruction”本身，只希望它学习生成“Response”。
因此，在构建训练数据时，我们需要构造 `targets` 序列，将 Instruction 部分设为 `-1` (Ignore Index)。

**示例**:
*   **Input IDs**: `[Instruction Tokens] [Response Tokens] [EOS]`
*   **Targets**: `[-1, -1, ..., -1] [Response Tokens] [EOS]`

## 4. 实现计划

### 步骤 1: 数据准备 (`data/sft_demo/prepare_sft.py`)
1.  读取 `data.jsonl`。
2.  应用 Prompt 模板。
3.  使用 `tiktoken` 进行分词。
4.  生成 `input_ids` 和 `targets` (应用 Masking)。
5.  保存为 `train.bin` 和 `val.bin`。注意：SFT 数据通常不再是连续的长流，而是独立的样本。为了复用现有的 `train.py` (它假设是连续流)，我们可以将样本拼接起来，或者修改 `train.py` 的数据加载逻辑。
    *   *方案 A (简单)*: 拼接样本，中间用 EOS 分隔。
    *   *方案 B (严谨)*: 修改 `train.py`，每次取一个完整的样本（需要 Padding）。

为了保持 nanoGTP 的简洁性，我们采用 **方案 A (拼接 + EOS)**，这样可以直接复用 `train.py` 或只需微调。

### 步骤 2: 微调脚本 (`finetune.py`)
1.  基于 `train.py` 修改。
2.  **初始化**: 必须从预训练模型加载权重 (`init_from='resume'`)。
3.  **超参数**:
    *   `learning_rate`: 远小于预训练 (e.g., 1e-5 vs 6e-4)。
    *   `max_iters`: 较少 (e.g., 几百步)。
    *   `warmup_iters`: 较少。

## 5. 目录结构
```
nanoGTP/
├── data/
│   └── sft_demo/
│       ├── data.jsonl      # 原始数据
│       └── prepare_sft.py  # SFT 预处理脚本
├── finetune.py             # SFT 训练脚本
└── SFT_DESIGN.md           # 本文档
```
