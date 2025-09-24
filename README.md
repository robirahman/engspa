# English-Spanish Translation Training

This repository fine-tunes English→Spanish neural machine translation models with Hugging Face's Transformers Trainer. The recommended environment, dataset, and sizing guidance come from [docs/training_setup.md](docs/training_setup.md).

## 1. Prerequisites

1. **Python**: 3.9 or newer.
2. **GPU**: Single NVIDIA RTX 4050 (6 GB) or compatible CUDA GPU.
3. **Python packages**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets evaluate sacrebleu accelerate sentencepiece tensorboard
   ```

   > ℹ️ `tensorboard` is optional but enables visualization of logged training metrics.

## 2. Dataset

By default the training script downloads the **WMT14 English-Spanish** dataset via the Hugging Face `datasets` library:

```bash
python train_translation.py --dataset wmt14 --dataset-config es-en
```

The script expects a `translation` column containing dictionaries with language codes. Alternative corpora (e.g., `opus100`, `tatoeba`) can be used by passing different `--dataset` / `--dataset-config` values.

## 3. Launching Training

Run the trainer from the project root:

```bash
python train_translation.py \
  --model Helsinki-NLP/opus-mt-en-es \
  --output-dir checkpoints/en-es \
  --max-source-length 128 \
  --max-target-length 128 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 3 \
  --warmup-steps 500 \
  --logging-steps 50 \
  --save-total-limit 2 \
  --validation-split 0.02 \
  --max-memory-gb 6
```

All flags shown above already match the script defaults; override them as needed. Additional useful options include:

- `--no-fp16`: disable mixed precision if training on CPU or a GPU without fp16 support.
- `--memory-summary-steps N`: periodically emit `torch.cuda.memory_summary()` diagnostics.
- `--push-to-hub`: upload checkpoints to the Hugging Face Hub.

## 4. Monitoring and Outputs

- Logs stream to STDOUT and include BLEU scores, generation length, and CUDA memory usage.
- Checkpoints and the final tokenizer are saved in the directory specified by `--output-dir`.
- Launch TensorBoard to inspect metrics:

  ```bash
  tensorboard --logdir checkpoints/en-es
  ```

## 5. GPU Memory Guidance

Training on a 6 GB RTX 4050 comfortably fits models up to ~400M parameters in full fine-tuning mode. For larger checkpoints, consider gradient accumulation, gradient checkpointing, shorter sequence lengths, or parameter-efficient techniques (LoRA/adapters). During inference, expect a maximum of ~1B parameters before KV cache usage exhausts memory.

Refer to [docs/training_setup.md](docs/training_setup.md) for deeper explanations of package choices, dataset alternatives, and memory budgeting tips.
