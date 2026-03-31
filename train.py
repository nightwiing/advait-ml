import torch
import json
from model import GPTModel
from config import (batch_size, block_size, max_iters,
                    eval_interval, learning_rate, device)
import os

# Data
with open('data/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: ''.join([itos[i] for i in l])

data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            xb, yb   = get_batch(split)
            _, loss  = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training
model     = GPTModel(vocab_size).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=learning_rate,
                               weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_iters)

best_val_loss = float('inf')

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'saved/advait.pt')
            print(f"  → best model saved at step {step}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

# Save tokenizer
os.makedirs('saved', exist_ok=True)

with open('saved/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump({
        'stoi': stoi,
        'itos': {str(k): v for k, v in itos.items()},
        'vocab_size': vocab_size
    }, f, ensure_ascii=False)

print("Training complete!")
print(f"Best val loss: {best_val_loss:.4f}")