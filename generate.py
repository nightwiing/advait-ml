import torch
import json
import argparse
from model import GPTModel
from config import device, block_size

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--prompt',      default='',    type=str)
parser.add_argument('--temperature', default=0.9,   type=float)
parser.add_argument('--top_k',       default=40,    type=int)
parser.add_argument('--max_tokens',  default=300,   type=int)
args = parser.parse_args()

# Load tokenizer
with open('saved/tokenizer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stoi       = data['stoi']
itos       = {int(k): v for k, v in data['itos'].items()}
vocab_size = data['vocab_size']
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: ''.join([itos[i] for i in l])

# Load model
model = GPTModel(vocab_size).to(device)
model.load_state_dict(torch.load('saved/advait.pt', map_location=device))
model.eval()
print(f"Model loaded — generating...")

# Generate output
if args.prompt:
    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

output = decode(
    model.generate(
        context,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )[0].tolist()
)

print("\n--- Generated text ---")
print(output)