# NanoGTP Project Analysis

## Project Overview
NanoGTP is a lightweight implementation of a Generative Pre-trained Transformer (GPT) model, specifically designed to be trained on Shakespeare's works for character-level text generation. It is built using PyTorch and aims to be simple and educational.

## Core Architecture

### Model (`model.py`)
- **Type**: Decoder-only Transformer (GPT).
- **Components**:
  - `GPT`: Main class combining embeddings (token & position) and transformer blocks.
  - `Block`: Transformer block with Pre-LN (LayerNorm before attention/FFN).
  - `MultiHeadAttention`: Manual implementation of multi-head self-attention with causal masking.
  - `FeedForward`: Standard position-wise feed-forward network (Linear -> GELU -> Linear).
- **Key Hyperparameters** (Default in `train.py`):
  - Embedding Dimension (`n_embd`): 384
  - Layers (`n_layer`): 6
  - Heads (`n_head`): 6
  - Block Size (Context Window): 256
  - Vocab Size: 65 (Character-level)

### Training (`train.py`)
- **Optimizer**: AdamW.
- **Device Support**: Auto-detection for MPS (Apple Silicon) and CPU.
- **Data Loading**: Uses `numpy.memmap` for efficient handling of binary dataset files (`train.bin`, `val.bin`).
- **Loop**: Simple iteration loop with cross-entropy loss calculation.

### Data Processing (`prepare.py`)
- **Tokenizer**: Character-level tokenizer.
  - Builds vocabulary from `input.txt`.
  - Maps characters to integers (`stoi`) and vice versa (`itos`).
- **Storage**: Saves processed data as raw `uint16` binary files using `numpy`.

## Workflow
1.  **Preparation**: Run `prepare.py` to download/read `input.txt`, build the vocab, and save `train.bin`/`val.bin`.
2.  **Training**: Run `train.py` to train the model. It loads data from binary files and saves the checkpoint to `model.pth`.
3.  **Inference**: Run `sample.py` (assumed, based on standard practice) to generate text using the trained `model.pth`.

## File Analysis

### Core Files
- `model.py`: The model definition.
- `train.py`: The training script.
- `prepare.py`: Data preprocessing and tokenizer logic.
- `sample.py`: Script for generating text from the trained model.
- `config/`: Configuration files (e.g., `train_shakespeare_char.py`).

### Peripheral/Unused Files
The following files appear to be unrelated to the main Shakespeare GPT task, possibly from other experiments (NER, Code Search):
- `tokenizer.py`: Script to train a BPE tokenizer on `code_search_net`.
- `token_utils.py`: Utilities for Token Classification (NER) using HuggingFace Transformers.
- `analyze_vocab.py`: Likely for analyzing the vocabulary of the BPE tokenizer.
- `code-search-net-tokenizer/`: Directory containing BPE tokenizer artifacts.

## Quick Start Guide

1.  **Environment Setup**:
    ```bash
    python3 -m venv .env
    source .env/bin/activate
    pip install -r requirements.txt
    ```

2.  **Data Preparation**:
    Download the Shakespeare dataset and process it into binary format:
    ```bash
    # Download dataset
    mkdir -p data/shakespeare
    curl -o data/shakespeare/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    
    # Process data
    python prepare.py
    ```

3.  **Training**:
    Train the model using the optimized training script:
    ```bash
    python train.py
    ```
    The model will be saved to `model.pth`.

4.  **Generation**:
    Generate text using the trained model:
    ```bash
    python sample.py
    ```

5.  **Web UI**:
    Talk to the model in a web interface:
    ```bash
    python web_ui.py
    ```
    Then visit `http://127.0.0.1:5000` in your browser.

6.  **BPE Tokenizer**:
    - **Library**: `tiktoken` (GPT-2 encoding)
    - **Vocab Size**: 50304 (padded from 50257)
    - **Upgrade**: Run `python prepare.py` to re-encode data with BPE.

7.  **SFT (Supervised Fine-Tuning)**:
    -   **Prepare Data**:
        ```bash
        # Process JSONL data (instruction-response pairs)
        python data/sft_demo/prepare_sft.py
        ```
    -   **Run Finetuning**:
        ```bash
        # Finetune the base model
        python finetune.py
        ```
    -   **Output**:
        -   The finetuned model checkpoint will be saved to `out-sft/ckpt.pt`.
    -   **Design**: See [SFT Design](sft_design.md) for details.

## Identified Issues & Improvements
1.  **[FIXED] Performance Bottleneck in `train.py`**:
    - The `DataLoader` was instantiated inside the training loop, causing the entire dataset to be re-read from disk into memory at every iteration.
    - **Fix**: Moved `DataLoader` instantiation outside the loop and optimized memory mapping.

2.  **[FIXED] Attention Implementation**:
    - `Head` class used manual matrix multiplication for attention, which was memory-intensive and slow.
    - **Fix**: Replaced with `torch.nn.functional.scaled_dot_product_attention` (Flash Attention). This reduced memory usage and significantly sped up training by using optimized CUDA/MPS kernels.
