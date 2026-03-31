import torch

batch_size    = 32
block_size    = 64
max_iters     = 5000
eval_interval = 500
learning_rate = 1e-3
n_embd        = 164
n_head        = 2
n_layer       = 4
dropout       = 0.2

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")