import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# Take JSONL file

raw_data = []
with open('training_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        gabungan = f"Instruksi: {data['instruksi']}\nRespons: {data['respons']}\n<END>\n"
        raw_data.append(gabungan)
full_text = "".join(raw_data)

chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data_tensor = torch.tensor(encode(full_text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

# Hyperparameter
batch_size = 4    
block_size = 16   
max_iters = 1500  
learning_rate = 1e-3
n_embd = 32       

# Function for picking random in dataset
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# Architecture model
class Head(nn.Module):
    """Satu unit Self-Attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Attention Math (Q dikali K Transpose)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # Closed text for preventing LLM to cheat
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        return wei @ v

class NanoLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Checking Loss from LLM
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = NanoLLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Training loop
print("\n[!] Mesin dinyalakan. Memulai proses belajar...\n")
for iter in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # Kalkulus Backpropagation
    optimizer.step() # Update kecerdasan model

    if iter % 150 == 0:
        print(f"Langkah {iter} | Tingkat Kebodohan (Loss): {loss.item():.4f}")

print("\n[v] Proses belajar selesai!")

# Testing
print("\n--- HASIL GENERATE SETELAH DILATIH ---")
# Memberikan huruf 'I' sebagai awalan tebakan (biasanya awal dari kata "Instruksi")
context = torch.tensor([[stoi['I']]], dtype=torch.long)
hasil = model.generate(context, max_new_tokens=150)[0].tolist()
print(decode(hasil))