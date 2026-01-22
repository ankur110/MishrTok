# MishrTok

![Language](https://img.shields.io/badge/Language-Python-blue)
![Efficiency](https://img.shields.io/badge/Efficiency-1.14x_vs_GPT4o-brightgreen)
![Vocab](https://img.shields.io/badge/Vocab_Size-32k-orange)

**MishrTok** (मिश्र = Mixed) – A custom Byte Pair Encoding (BPE) tokenizer optimized for **code-mixed Romanized Hinglish** and **Devanagari Hindi**.

Trained on a large Hinglish corpus with intelligent regex-based pre-tokenization, this 32k tokenizer achieves **~14.1% better token efficiency** than OpenAI's state-of-the-art `o200k_base` (GPT-4o) tokenizer on diverse Hinglish and Hindi text — all with a vocab size 6× smaller.

This project is heavily inspired by **Andrej Karpathy's** ["Let's build the GPT tokenizer"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and his clean [`minbpe`](https://github.com/karpathy/minbpe) implementation.

## Why This Tokenizer?

General-purpose tokenizers struggle with Romanized Hinglish and Devanagari Hindi because:
- Common Hindi words/phrases ("bhai", "yaar", "tension na le", "ज़िंदगी", etc.) get overly fragmented.
- Code-mixing and script-mixing patterns are underrepresented in English-dominated training data.
- Older tokenizers (cl100k, p50k) completely fall apart on pure Devanagari text.

This tokenizer uses:
- A powerful regex to pre-tokenize URLs, mentions, hashtags, emojis, numbers, **Devanagari script blocks**, and separate Latin/Romanized words.
- Frequency-weighted BPE merges on real Hinglish data.
- Result: Fewer tokens → faster inference, lower costs, better context utilization for Hindi/Hinglish LLMs.

## Features

- Regex pre-tokenization (handles mixed scripts, emojis, social media artifacts)
- Supports both Romanized Hinglish and pure Devanagari Hindi efficiently
- Frequency-filtered pre-tokens (`min_word_freq`, `max_unique_words`)
- Training checkpoints
- Clean encode/decode with perfect round-trip
- **Outperforms** `o200k_base` (GPT-4o), `cl100k_base`, and older OpenAI tokenizers on Hinglish/Hindi

## 📊 Benchmark Results

We compared `MishrTok` against OpenAI's latest `o200k_base` (GPT-4o) on a diverse test set.

| Task Category | MishrTok Tokens | GPT-4o Tokens | Efficiency |
| :--- | :---: | :---: | :--- |
| **Casual Chat** | 93 | 103 | ✅ **1.11x** |
| **Tech Discussion** | 90 | 91 | ✅ **1.01x** |
| **Emotional Rant** | 107 | 129 | ✅ **1.21x** |
| **Hardcore Hinglish** | 70 | 95 | ✅ **1.36x** |
| **Pure Hindi** | 60 | 70 | ✅ **1.17x** |
| **TOTAL** | **510** | **582** | ✅ **1.14x** |

> **Verdict:** MishrTok is **14.1% more efficient** than GPT-4o on mixed-script text.


## 📂 Project Structure

```text
Hinglish-Tokenizer/
├── data/
│   └── ds.ipynb                  # Dataset preparation / exploration notebook
├── models/
│   ├── hinglish_32k.model        # Trained tokenizer merges & metadata
│   └── hinglish_32k.vocab.json   # Vocab (hex-encoded bytes)
├── src/
│   ├── HinglishBPE.py            # Core tokenizer class (Train/Encode/Decode)
│   └── training.py               # Training script
├── inference.py                  # Comprehensive benchmark vs OpenAI
└── README.md

```

## 🛠️ Quick Start

### 1. Installation

This project does not require a heavy environment. Just install the dependencies:

```bash
pip install regex tqdm tiktoken datasets

```

### 2. Usage

You can load the pre-trained model and start encoding immediately.

```python
from src.HinglishBPE import HinglishBPE

# Load the tokenizer (ensure path is correct)
tok = HinglishBPE()
tok.load("models/hinglish_32k")

text = "विज्ञान और प्रौद्योगिकी के क्षेत्र में हमने बहुत प्रगति की है।"

# Encode
ids = tok.encode(text)
print(f"Tokens ({len(ids)}): {ids}")

# Decode (Round-trip check)
decoded = tok.decode(ids)
print(f"Decoded: {decoded}")
assert text == decoded

```

### 3. Running Benchmarks

To reproduce the efficiency results on your own machine:

```bash
python inference.py

```

### 4. Training from Scratch

If you have a custom corpus (e.g., `corpus.txt`), you can retrain the model:

```python
from src.HinglishBPE import HinglishBPE

tok = HinglishBPE()
tok.train(
    filename="data/corpus.txt",
    vocab_size=32768,
    min_word_freq=2,
    max_unique_words=3_000_000,
    verbose=True,
    checkpoint_prefix="hinglish_32k_chk",
)
tok.save("models/hinglish_32k")

```

## 🧠 Acknowledgments

* **Andrej Karpathy:** For the clean [`minbpe`](https://github.com/karpathy/minbpe) architecture and educational resources.
* **OpenAI:** For `tiktoken`, used here for benchmarking comparisons.
* **The Indic NLP Community:** For creating datasets like HinGE and IndicCorp that made this training possible.


