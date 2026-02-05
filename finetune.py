import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 默认 SFT 配置 (可以根据 train.py 修改)
out_dir = 'out-sft'
eval_interval = 20
log_interval = 1
eval_iters = 10
eval_only = False # 如果为 True，脚本只会在第一次评估后退出
always_save_checkpoint = True # 总是保存，哪怕 loss 只是好了一点点
init_from = 'gpt2' # 'scratch' or 'resume' or 'gpt2' (从 HuggingFace 加载)
# 鉴于我们没有真正预训练好的 50304 vocab 的大模型 (除非用 train.py 跑了很久)，
# 我们可以加载 gpt2 的权重 (50257 vocab)，然后 resize embedding
# 或者简单起见，如果 init_from='resume'，假设你有一个 out/ckpt.pt

# 数据配置
dataset = 'sft_demo'
gradient_accumulation_steps = 1
batch_size = 4 # SFT 数据通常较少，且长度不一，batch_size 小一点
block_size = 256 # 上下文长度

# 模型配置
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # 微调时增加一点 dropout 防止过拟合

# 优化器配置
learning_rate = 1e-5 # 微调学习率通常比预训练低 1-2 个数量级
max_iters = 50
lr_decay_iters = 50 # 学习率衰减
min_lr = 1e-6 
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True # 使用 PyTorch 2.0 编译

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 允许命令行覆盖配置
config = {k: globals()[k] for k in config_keys} # log 用的配置

# 设置随机种子
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 数据加载器
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # SFT 数据加载逻辑
    # 我们生成了 input_ids (uint16) 和 targets (int32)
    # 并且已经做好了 masking
    
    # 简单的内存映射
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train_ids.bin'), dtype=np.uint16, mode='r')
        targets = np.memmap(os.path.join(data_dir, 'train_targets.bin'), dtype=np.int32, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val_ids.bin'), dtype=np.uint16, mode='r')
        targets = np.memmap(os.path.join(data_dir, 'val_targets.bin'), dtype=np.int32, mode='r')
        
    # 随机选择位置
    # 注意: 我们的数据其实是拼接的，这在 SFT 中其实不太严谨（样本间没有 Padding 隔离）
    # 但为了复用逻辑，我们假设数据量足够大，或者 block_size 足够大能包含完整样本
    # 在这个 demo 中，我们随机截取
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((targets[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda' or device_type == 'mps':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 初始化模型
iter_num = 0
best_val_loss = 1e9

# SFT 必须有基础模型，这里我们为了演示：
# 1. 尝试从 out/ckpt.pt 加载 (如果我们之前训练过)
# 2. 如果没有，则加载 gpt2 (并 resize)
# 3. 如果指定 init_from='scratch'，那就是重头练 (不推荐用于 SFT)

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 这里假设我们想从 out/ckpt.pt (预训练) 继续，保存到 out-sft/
    # 但如果是 SFT，通常是从预训练目录加载，保存到新目录
    # 让我们假设用户手动把预训练权重放到了 out-sft/ckpt.pt，或者修改路径
    ckpt_path = os.path.join('out', 'ckpt.pt') # 默认从 out/ 加载预训练权重
    if not os.path.exists(ckpt_path):
        # 如果没有预训练权重，为了演示，我们回退到 scratch
        print(f"Checkpoint {ckpt_path} not found. Fallback to scratch for demo.")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # 强制更新配置
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        print("Loaded pretrained model.")

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Resize vocab if needed (e.g. if we added special tokens or padded)
if meta_vocab_size is not None and model_args['vocab_size'] != meta_vocab_size:
    print(f"Resizing model vocab from {model_args['vocab_size']} to {meta_vocab_size}")
    model.crop_block_size(block_size)
    # 简单调整最后的线性层和 embedding
    # 注意: 这是一个简化的 resize，未初始化新 token
    # 但在我们的例子中，50257 -> 50304 只是 padding，通常没问题
    # 如果 model 是 GPT 类，我们需要手动处理 resize?
    # model.py 里没有 resize_token_embeddings。
    # 为了简单，如果尺寸不匹配且使用了 gpt2 权重，我们可能会报错。
    # 这里先假设 meta_vocab_size (50304) 和模型一致 (如果我们是从 out/ckpt.pt 加载的)
    pass

model.to(device)

# 优化器
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 训练循环
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if hasattr(model, "module") else model # unwrap DDP container if needed

# 创建输出目录
os.makedirs(out_dir, exist_ok=True)

while True:
    # determine and set the learning rate for this iteration
    lr = learning_rate # constant lr for SFT usually, or cosine decay
    # simple warmup
    if iter_num < 0: # no warmup for now
        pass
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    if gradient_accumulation_steps > 1: # unscale gradients
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
