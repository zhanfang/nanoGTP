# nanoGTP 演进之路 (Project Evolution)

本文档旨在记录 nanoGTP 项目的演进历程。通过回顾每一个关键的技术决策和优化步骤，我们可以更深入地理解构建现代 LLM (Large Language Model) 的核心要素。

---

## 📅 阶段一：原型诞生 (v1.0)
> **Commit**: `e6f3e6d` - `feat: finish nanaGPT v1.0`

### 🎯 目标
构建一个最小可行的 GPT 模型 (Decoder-only Transformer)，用于验证最基本的文本生成能力。

### 🛠️ 核心实现
1.  **Tokenizer (分词器)**:
    - 采用最简单的 **Character-level (字符级)** 分词。
    - 词表大小 (Vocab Size): ~65 (仅包含 ASCII 字符)。
    - *特点*: 实现简单，但语义稀疏。模型需要逐个字符地“拼写”单词（如 `h-e-l-l-o`）。

2.  **Model (模型)**:
    - 标准的 Transformer 结构：Embedding -> Positional Encoding -> Blocks (Multi-Head Attention + FeedForward) -> LayerNorm -> Head。
    - 简单的自注意力机制实现。

---

## 🚀 阶段二：性能与效率优化
> **Commit**: `157dc18` - `perf: optimize training data loading and use flash attention`

随着训练的深入，我们发现了两个主要的性能瓶颈：训练速度慢和显存占用高。

### ⚡ 优化 1: Flash Attention
*   **问题**: 传统的 Attention 计算需要构建 $N \times N$ 的注意力矩阵，显存占用与序列长度的平方成正比 $O(N^2)$，且涉及大量的内存读写操作。
*   **解决方案**: 引入 PyTorch 2.0+ 的 `F.scaled_dot_product_attention`。
*   **原理**: Flash Attention 通过平铺 (Tiling) 和重计算 (Recomputation) 技术，减少了 GPU HBM (高带宽内存) 的读写次数。
*   **代码变化 (`model.py`)**:
    ```python
    # Before
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    y = F.softmax(att, dim=-1) @ v

    # After (Flash Attention)
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
    )
    ```

### 🚅 优化 2: DataLoader 改进
*   **问题**: 原始代码在每次训练迭代 (iteration) 中都重新实例化 `DataLoader` 和 `Dataset`。
*   **后果**: 导致大量重复的文件 I/O 和对象创建开销，CPU 成为瓶颈。
*   **解决方案**: 将 `DataLoader` 的实例化移至训练循环外部，并让 Dataset 一次性加载数据到内存（针对小数据集）。

---

## 💻 阶段三：交互体验升级 (Web UI)
> **Commit**: `7080a5f` - `feat: add web ui and update docs`

为了让模型不仅仅停留在命令行，我们决定为它赋予一个现代化的界面。

### ✨ 实现细节
*   **技术栈**: Flask (后端) + HTML/JS (前端)。
*   **功能**:
    *   构建了一个类似 ChatGPT 的对话界面。
    *   支持用户输入 Prompt，模型实时返回生成结果。
    *   将模型加载逻辑封装，避免每次请求重新加载模型。

---

## 🧠 阶段四：核心能力质变 (BPE Upgrade)
> **Commit**: `4ee0e8a` - `feat: upgrade tokenizer to GPT-2 BPE using tiktoken`

这是项目迄今为止最重要的一次升级。我们将分词器从字符级升级为 GPT-2 标准的 BPE (Byte Pair Encoding)。

### 🧐 为什么要升级？
字符级模型虽然简单，但存在致命弱点：
1.  **信息密度低**: 一个 Token 只是一个字母。Context Window (比如 1024) 只能看到 1024 个字母（约 200 个单词）。
2.  **学习难度大**: 模型必须花费大量参数去学习“单词拼写”，而不是专注于语义和逻辑。

### 🔄 升级内容 (BPE)
*   **库**: 引入 OpenAI 的 `tiktoken` 库。
*   **词表**: 扩展到 **50,304** (GPT-2 原生 50257 + Padding)。
*   **效果**:
    *   **信息密度提升 ~3-4 倍**: 现在一个 Token 通常代表一个完整的单词或词根（如 ` learning`）。
    *   **生成质量飞跃**: 模型能生成更连贯、语法更正确的文本，因为它是在“词”的层面上进行思考。

### 📊 对比
| 特性 | Character-level (旧) | BPE (新) |
| :--- | :--- | :--- |
| **词表大小** | ~65 | 50,304 |
| **Token 代表** | 单个字符 (a, b, c) | 子词/单词 (the, apple, ing) |
| **序列长度效率** | 低 (一句话需 50 tokens) | 高 (一句话需 15 tokens) |
| **语言理解能力** | 弱 | 强 |

---

## 🔮 未来展望
我们的 nanoGTP 已经初具雏形，接下来的演进方向可能包括：
1.  **模型规模扩展**: 增加层数 (n_layer) 和 嵌入维度 (n_embd)。
2.  **预训练 (Pre-training)**: 在更大的数据集 (如 OpenWebText) 上进行训练。
3.  **微调 (Fine-tuning)**: 尝试指令微调 (Instruction Tuning)，让它更能听懂指令。

---
*Created by Trae AI Assistant, 2026*
