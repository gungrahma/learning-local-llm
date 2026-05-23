# Local LLM - Belajar LLM dari Nol

Proyek ini adalah implementasi Neural Network Language Model (NanoLLM) dari awal menggunakan PyTorch. Dibuat untuk memahami cara kerja Large Language Model (LLM) secara fundamental.

## Struktur Proyek

```
local-llm/
├── data_prep.py          # Persiapan data training
├── train.py              # Training model
├── training_data.jsonl   # Dataset latihan (Indonesian)
├── local_llm/            # Virtual environment
└── README.md
```

## Fitur

- **Self-Attention** - Implementasi attention mechanism dari nol
- **Tokenisasi Karakter** - Bekerja dengan level karakter
- **Text Generation** - Generasi teks berdasarkan seed
- **Training Loop** - Backpropagation dan optimasi AdamW

## Cara Pakai

### 1. Setup Environment

```bash
cd local-llm
source local_llm/bin/activate
```

### 2. Jalankan Training

```bash
python train.py
```

### 3. Lihat Hasil

Model akan melatih selama ~1500 iterasi dan menampilkan loss setiap 150 langkah. Setelah selesai, model menguji generasi teks.

## Arsitektur Model

```
NanoLLM
├── Token Embedding Table
├── Self-Attention Head (1 unit)
└── Linear Language Model Head
```

**Hyperparameter:**
- Batch size: 4
- Block size: 16
- Embedding dimensions: 32
- Learning rate: 1e-3
- Max iterations: 1500

## Dataset

Dataset terdiri dari pasangan instruksi-respons dalam Bahasa Indonesia (format JSONL):

```
{"instruksi": "...", "respons": "..."}
```

## Dependencies

- Python 3.x
- PyTorch