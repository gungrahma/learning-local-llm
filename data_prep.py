import torch
import json

raw_data = []
with open ('training_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        gabungan_text = f"Instruksi: {data['instruksi']}n\Respons: {data['respons']}\n<END>\n"
        raw_data.append(gabungan_text)

full_text = "".join(raw_data)
print("Preview text mentah: ")
print(full_text[:150] + "...\n")

# Tokenisasi dan Check berapa token yang dibuat

chars = sorted(list(set(full_text)))
vocab_size = len(chars)

print(f"Jumlah karakter untuk (Vocab Size): {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

data_tensor = torch.tensor(encode(full_text), dtype = torch.long)

print("\nBentuk Tensor (Angka Matematika): ")
print(data_tensor[:20])

n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

print(f"\nData siap! Total token latih: {len(train_data)}")