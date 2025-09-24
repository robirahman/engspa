# English-Spanish Translation Training Setup

This repository is geared toward preparing an English-Spanish machine translation (MT) system using a single NVIDIA RTX 4050 GPU. Below is an overview of the key Python packages to install and import, a recommended dataset for initial experimentation, and guidance on model sizing for a 6 GB memory budget.

## Core Python Packages

| Purpose | Package | Notes |
| --- | --- | --- |
| Deep learning framework with CUDA support | `torch` | Install via `pip install torch --index-url https://download.pytorch.org/whl/cu121` to match the RTX 4050 (CUDA 12.1). |
| Model architectures, tokenizers, and utilities | `transformers` | Provides encoder-decoder translation models (e.g., `Helsinki-NLP/opus-mt-en-es`) and trainer utilities. |
| Dataset loading and preprocessing | `datasets` | Handles streaming/download of translation corpora from the Hugging Face Hub. |
| Evaluation metrics | `evaluate`, `sacrebleu` | `evaluate` wraps metrics such as BLEU; `sacrebleu` offers consistent BLEU scoring for MT. |
| Training optimizations & mixed precision | `accelerate` | Simplifies mixed-precision and distributed training even on a single GPU. |
| Tokenizer construction | `sentencepiece` | Many translation checkpoints rely on SentencePiece tokenizers. |
| Experiment tracking (optional) | `tensorboard` or `wandb` | For visualizing training metrics. |

Typical imports inside your training script will therefore look similar to:

```python
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
import evaluate
import sacrebleu
```

## Recommended Dataset

For a well-balanced, high-quality English-Spanish parallel corpus that is easy to access programmatically, use the **WMT14 English-Spanish** dataset available through the Hugging Face `datasets` library. It contains news-domain sentence pairs that are widely used for benchmarking MT models.

```python
dataset = load_dataset("wmt14", "es-en")
```

Key properties:

- **Size**: ~1.9 million sentence pairs across train/validation/test splits.
- **Quality**: Curated for the WMT evaluation campaign with professional translations.
- **Licensing**: Released for research; review WMT licensing before commercial use.

### Alternatives

If you need a different domain or a lighter-weight corpus, consider:

- **OPUS100 (`opus100`, "en-es")**: 1M curated sentence pairs extracted from the OPUS collection.
- **Tatoeba (`tatoeba`, "eng-spa")**: Smaller, community-contributed sentences, useful for quick experiments.

All of these datasets can be loaded through `datasets.load_dataset(...)` and will work seamlessly with the Hugging Face `Seq2SeqTrainer` API.

## Model Size Guidance for a 6 GB RTX 4050

Although a 6 GB GPU might appear to hold 3 billion parameters when counting only half-precision weights (2 bytes each), practical training requires several additional memory components:

| Component | Precision | Bytes per parameter (approx.) | Notes |
| --- | --- | --- | --- |
| Model weights | fp16/bfloat16 | 2 | Stored on the GPU during forward/backward passes. |
| Gradients | fp16/bfloat16 | 2 | Created during backpropagation before optimizer updates. |
| Optimizer states (Adam/AdamW) | fp32 | 8 | Two fp32 buffers (`m` and `v`) are maintained per parameter. |
| Activation checkpoints | fp16/fp32 | Variable | Depends on batch size, sequence length, and whether gradient checkpointing is used. |
| KV cache (during generation) | fp16 | ~2 × hidden size × sequence length | Needed for autoregressive decoding at inference time. |

Putting this together, standard mixed-precision training with AdamW consumes roughly 12 bytes per parameter _before_ accounting for activations. With only 6 GB available, that limits you to ≈500 million parameters in theory, and closer to 300–400 million parameters in practice once you budget 1–2 GB for activations, data batches, and kernel workspaces. Sequence-to-sequence translation models such as `Helsinki-NLP/opus-mt-en-es` (~75M parameters) or `facebook/mbart-large-50` (~610M parameters) illustrate the trade-offs:

- **For full fine-tuning** on a single RTX 4050, stay with models in the 200–400M parameter range (e.g., Marian MT or BART-base variants) to leave room for activations and maintain reasonable batch sizes.
- **For larger checkpoints** (≥500M parameters such as mBART-large), use strategies like gradient accumulation, gradient checkpointing, reduced sequence lengths, or parameter-efficient tuning (LoRA, adapters) to stay within memory limits.

During inference, you no longer need gradients or optimizer states, so the memory footprint drops to roughly the model weights plus the KV cache. Even then, a 1B-parameter fp16 model already consumes ~2 GB for weights and can require several additional GB for the KV cache when translating long sequences. Consequently, a 1B-parameter model is close to the upper bound for comfortable inference on a 6 GB GPU, while 2–3B parameter models will exceed the available memory once the KV cache and workspace requirements are considered.

In summary, the 6 GB RTX 4050 is best suited for training medium-sized (≤400M parameter) MT models in full precision or larger models via parameter-efficient fine-tuning, and for running inference on models up to roughly 1B parameters when sequence lengths are moderate.
