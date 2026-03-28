"""
Training loop for IndoBERT sentiment classification using HuggingFace Trainer.

Usage:
    python src/train.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import evaluate as hf_evaluate

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.dataset import build_datasets
from src.model import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_accuracy_metric = hf_evaluate.load("accuracy")
_f1_metric = hf_evaluate.load("f1")


def compute_metrics(eval_pred) -> dict:
    """
    Compute accuracy and macro / per-class F1 from Trainer predictions.

    Parameters
    ----------
    eval_pred : EvalPrediction
        Namedtuple with ``predictions`` and ``label_ids``.

    Returns
    -------
    dict
        Metrics dictionary consumed by Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = _accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    macro_f1 = _f1_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["f1"]

    # Per-class F1
    per_class = _f1_metric.compute(
        predictions=preds, references=labels, average=None
    )["f1"]

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
    for i, name in enumerate(config.LABEL_NAMES):
        metrics[f"f1_{name}"] = float(per_class[i])

    return metrics


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train() -> None:
    """Run full fine-tuning of IndoBERT on IndoNLU SmSA."""
    set_seed()
    print(f"Device: {config.DEVICE}")
    print(f"Loading tokenizer: {config.MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = build_model(config.MODEL_NAME, config.NUM_LABELS)
    model.to(config.DEVICE)

    print("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(tokenizer)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    best_dir = str(config.CHECKPOINTS_DIR / "best")
    run_dir = str(config.CHECKPOINTS_DIR / "run")

    # Compute warmup steps
    steps_per_epoch = len(train_ds) // config.BATCH_SIZE
    total_steps = steps_per_epoch * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=str(config.OUTPUTS_DIR / "logs"),
        seed=config.RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save best model + tokenizer
    print(f"Saving best model to {best_dir}")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\n=== Test Results ===")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    import json
    results_path = config.RESULTS_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    train()
