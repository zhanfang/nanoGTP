# Train GPT-2 (124M) on OpenWebText
# 
# To run:
# $ python train.py config/train_gpt2.py
#
# Note: You need to run data/openwebtext/prepare.py first to generate the bin files

out_dir = 'out-gpt2'
eval_interval = 2000
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes

batch_size = 12
block_size = 1024

# GPT-2 124M model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10
