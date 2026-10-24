# GPT-2 from Scratch in TensorFlow

A GPT-2 style decoder-only transformer, written from first principles with Keras layers and trained on real text from Hugging Face. **No pretrained weights are used.** The model learns language from random initialisation.

The goal of this project is to understand and reproduce the core mechanics of a large language model: tokenisation, causal self-attention, transformer blocks, next-token training, and autoregressive sampling.

---

## Highlights

- **Multi-head causal self-attention** implemented by hand (Q/K/V projections, head splitting, scaled dot-product, causal mask)
- **Transformer blocks** with GELU feed-forward networks, residual connections, layer normalisation and dropout
- **Learned token and positional embeddings**, using the same 50,257-token BPE vocabulary as GPT-2
- **Data pipeline** that pulls any Hugging Face parquet dataset and converts it to GPT-2 token ids
- **Training** with Adam, gradient clipping and per-epoch checkpointing
- **Text generation** with temperature and top-k sampling
- Scales from a **29M-parameter laptop model** to the **124M GPT-2 Small** configuration by changing constants

---

## Architecture

```
token ids ─► Token Embedding + Positional Embedding
                     │
          ┌──────────▼──────────┐
          │  Transformer Block  │  × N
          │  ├─ Causal Multi-Head Self-Attention
          │  ├─ Add & LayerNorm
          │  ├─ Feed-Forward (Dense → GELU → Dense)
          │  └─ Add & LayerNorm
          └──────────┬──────────┘
                     ▼
              Final LayerNorm
                     ▼
         Linear → logits over 50,257 tokens
```

| Configuration | Layers | Heads | Embed dim | FFN dim | Context | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| Default (this repo) | 4 | 4 | 256 | 1024 | 128 | ~29M |
| GPT-2 Small (`GPT-2.py` demo) | 12 | 12 | 768 | 3072 | 1024 | ~124M+ |

---

## Quick start

```bash
pip install -r requirements.txt     # Python 3.11

python train_gpt2.py                # train (tokenised WikiText-2 is included)
python generate.py                  # write text from a prompt
```

To train on a different dataset, edit the settings at the top of `download_data.py`, then run:

```bash
python download_data.py
```

> On Windows TensorFlow runs on CPU only. For GPU training use Linux, WSL2 or Google Colab.

---

## Project structure

| File | Purpose |
| --- | --- |
| `GPT-2.py` | Model definition: attention, feed-forward, transformer block, GPT-2 |
| `download_data.py` | Downloads a Hugging Face dataset and tokenises it with the GPT-2 BPE tokenizer |
| `train_gpt2.py` | Builds (input, target) sequences, trains the model and saves checkpoints |
| `generate.py` | Autoregressive generation with temperature and top-k sampling |
| `model.ipynb` | Step-by-step notebook version of the model, for learning |
| `data/tokens/` | 2M pre-tokenised WikiText-2 tokens, so training works out of the box |

---

## Training setup

| Setting | Value |
| --- | --- |
| Dataset | WikiText-2 (raw), 2,000,000 tokens |
| Split | 95% train / 5% validation |
| Sequence length | 128 |
| Batch size | 8 |
| Optimiser | Adam, lr 3e-4, gradient clip 1.0 |
| Loss | Sparse categorical cross-entropy (next-token prediction) |
| Epochs | 3 |

All settings are constants at the top of each file. If you change the model's size, delete `checkpoints/` first, because old weights won't fit the new shape.

---

## Generation settings

| Setting | Default | Effect |
| --- | --- | --- |
| `PROMPT` | "The history of" | Text the model continues |
| `MAX_NEW_TOKENS` | 100 | Length of the output |
| `TEMPERATURE` | 0.8 | Lower is more conservative, higher is more creative |
| `TOP_K` | 40 | Sample only from the 40 most likely tokens |

---

## Roadmap

- [ ] Pre-LayerNorm blocks and weight tying (as in the original GPT-2)
- [ ] Rotary positional embeddings (RoPE), RMSNorm and SwiGLU (as in modern LLMs)
- [ ] Learning-rate warm-up and cosine decay
- [ ] Training on TinyStories and reporting perplexity
- [ ] Mixed-precision GPU training

---

## Author

**Christopher Sathiya Satchi**, Junior Researcher at CIIAE, Spain
[LinkedIn](https://www.linkedin.com/in/christophers828269107/) · [Google Scholar](https://scholar.google.com/citations?user=s9fiBJoAAAAJ)
