"""
Default configuration for GPT model training.
"""

class Config:
    # Model parameters
    n_layer = 12
    n_head = 12
    n_embd = 768
    vocab_size = 50257  # GPT-2 vocabulary size
    block_size = 1024   # Maximum sequence length
    dropout = 0.1

    # Training parameters
    batch_size = 8
    learning_rate = 3e-4
    max_epochs = 10
    warmup_tokens = 375e6
    final_tokens = 260e9
    
    # Data parameters
    train_data_path = "data/train.txt"
    val_data_path = "data/val.txt"
    
    # Logging and checkpointing
    log_interval = 100
    eval_interval = 1000
    save_interval = 1000
    checkpoint_dir = "checkpoints"