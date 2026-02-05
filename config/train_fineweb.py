# Train GPT-2 (124M) on FineWeb-Edu (sample-10BT)
# 
# To run:
# $ python train.py config/train_fineweb.py
#
# Note: You need to run data/fineweb/prepare.py first to generate the bin files

out_dir = 'out-fineweb'
eval_interval = 2000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'fineweb-edu'
wandb_run_name = 'gpt2-124M'

dataset = 'fineweb'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes

batch_size = 12
block_size = 1024

# GPT-2 124M model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

# learning rate decay settings
decay_lr = True 
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 6e-5 
