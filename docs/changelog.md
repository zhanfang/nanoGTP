# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-02-05

### Features
- **BPE Tokenizer**: Upgraded from character-level tokenizer to GPT-2 BPE tokenizer using `tiktoken`.
  - Improved language understanding and generation capabilities.
  - Vocabulary size increased from ~65 to 50304 (padded).
  - Updated `prepare.py`, `train.py`, `sample.py`, and `web_ui.py` to support BPE.
- **Web UI**: Added a Flask-based web interface (`web_ui.py` and `templates/index.html`) to interact with the model in a ChatGPT-like environment.

### Performance Improvements
- **Optimized Data Loading**: Refactored `train.py` to fix a critical performance bottleneck. The `DataLoader` was previously re-instantiating and re-reading the entire dataset from disk at every iteration. It now loads data once into memory, significantly improving training speed.
- **Flash Attention**: Updated `model.py` to use `torch.nn.functional.scaled_dot_product_attention`. This replaces the manual multi-head attention implementation with optimized CUDA/MPS kernels, reducing memory usage and accelerating computation.

### Documentation
- **Agent.md**: Created a comprehensive project analysis document, including architecture overview, file analysis, and a new "Quick Start Guide" for easier onboarding.
