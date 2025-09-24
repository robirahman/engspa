"""Training script for English-Spanish translation models.

This module wires together Hugging Face's 🤗 Transformers Trainer API
with the WMT14 English-Spanish dataset.  It intentionally keeps the
surface area small so that you can launch fine-tuning runs on a single
RTX 4050 by adjusting only a few CLI arguments.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import evaluate

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Runtime configuration for the translation experiment."""

    model_name_or_path: str
    dataset_name: str
    dataset_config: str
    source_lang: str
    target_lang: str
    output_dir: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    warmup_steps: int
    logging_steps: int
    save_total_limit: int
    seed: int
    fp16: bool
    predict_with_generate: bool
    push_to_hub: bool
    validation_split: float
    max_memory_gb: float
    memory_summary_steps: int


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for En→Es translation.")
    parser.add_argument("--model", dest="model_name_or_path", default="Helsinki-NLP/opus-mt-en-es")
    parser.add_argument("--dataset", dest="dataset_name", default="wmt14")
    parser.add_argument("--dataset-config", dest="dataset_config", default="es-en")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="es")
    parser.add_argument("--output-dir", default="checkpoints/en-es")
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--predict-with-generate", dest="predict_with_generate", action="store_true")
    parser.add_argument("--no-predict-with-generate", dest="predict_with_generate", action="store_false")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--validation-split", type=float, default=0.02, help="Fraction of train split to use for validation when missing.")
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=6.0,
        help="Maximum desired CUDA memory footprint in GB before emitting a warning.",
    )
    parser.add_argument(
        "--memory-summary-steps",
        type=int,
        default=0,
        help="Interval (in steps) at which to log the full cuda.memory_summary(); 0 disables periodic summaries.",
    )

    parser.set_defaults(fp16=True, predict_with_generate=True)

    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def prepare_datasets(cfg: TrainingConfig, tokenizer) -> DatasetDict:
    LOGGER.info("Loading dataset %s/%s", cfg.dataset_name, cfg.dataset_config)
    raw_datasets = load_dataset(cfg.dataset_name, cfg.dataset_config)

    if "train" not in raw_datasets:
        raise ValueError("Dataset must contain a train split")

    if "validation" not in raw_datasets:
        LOGGER.info("Validation split missing; sampling %.2f%% from train", cfg.validation_split * 100)
        split = raw_datasets["train"].train_test_split(test_size=cfg.validation_split, seed=cfg.seed)
        raw_datasets = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })
    else:
        raw_datasets = DatasetDict({
            "train": raw_datasets["train"],
            "validation": raw_datasets["validation"],
        })

    column_names = raw_datasets["train"].column_names
    if "translation" not in column_names:
        raise ValueError("Expected a 'translation' column with language dictionaries")

    def preprocess_function(examples: Dict[str, list]) -> Dict[str, list]:
        translations = examples["translation"]
        inputs = [t[cfg.source_lang] for t in translations]
        targets = [t[cfg.target_lang] for t in translations]
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )
    return tokenized_datasets


def build_compute_metrics(tokenizer) -> Callable:
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu = metric.compute(predictions=decoded_preds, references=decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        return {
            "bleu": bleu["score"],
            "gen_len": np.mean(prediction_lens),
        }

    return compute_metrics


class CudaMemoryCallback(TrainerCallback):
    """Monitor CUDA memory usage during training."""

    def __init__(self, limit_gb: float, summary_interval: int) -> None:
        self._limit_bytes = limit_gb * (1024 ** 3)
        self._summary_interval = max(summary_interval, 0)
        self._cuda_available = torch.cuda.is_available()

    def _log_memory_summary(self, label: str) -> None:
        if not self._cuda_available:
            return
        summary = torch.cuda.memory_summary(abbreviated=True)
        LOGGER.info("CUDA memory summary (%s):\n%s", label, summary)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not self._cuda_available:
            return
        torch.cuda.reset_peak_memory_stats()
        self._log_memory_summary("train begin")

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not self._cuda_available:
            return

        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        LOGGER.info(
            "CUDA memory usage — allocated: %.2f GB, peak since last log: %.2f GB",
            allocated / (1024 ** 3),
            max_allocated / (1024 ** 3),
        )

        if max_allocated > self._limit_bytes:
            LOGGER.warning(
                "Peak CUDA memory %.2f GB exceeded configured limit of %.2f GB.",
                max_allocated / (1024 ** 3),
                self._limit_bytes / (1024 ** 3),
            )

        if self._summary_interval and state.global_step % self._summary_interval == 0 and state.global_step > 0:
            self._log_memory_summary(f"step {state.global_step}")

        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    cfg = parse_args()
    configure_logging()

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. Training will run on CPU, which is very slow.")

    LOGGER.info("Loading tokenizer and model from %s", cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)

    tokenized_datasets = prepare_datasets(cfg, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=cfg.predict_with_generate,
        fp16=cfg.fp16,
        seed=cfg.seed,
        push_to_hub=cfg.push_to_hub,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[CudaMemoryCallback(cfg.max_memory_gb, cfg.memory_summary_steps)],
    )

    LOGGER.info("Starting training")
    trainer.train()

    LOGGER.info("Saving final model to %s", cfg.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "tokenizer"))


if __name__ == "__main__":
    main()
