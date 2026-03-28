"""
Training script for baseline models: BiLSTM, mBERT, and XLM-R.

Reuses the same SmSA dataset, metrics, and evaluation protocol as the
main IndoBERT training script (src/train.py).

Usage:
    python src/train_baselines.py --model mbert
    python src/train_baselines.py --model xlmr
    python src/train_baselines.py --model bilstm
    python src/train_baselines.py --model all      # train all baselines
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.dataset import build_datasets, load_smsa_dataset, normalize_indonesian
from src.train import compute_metrics, set_seed

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mbert": {
        "hf_name": config.MBERT_MODEL_NAME,
        "lr": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "weight_decay": config.WEIGHT_DECAY,
    },
    "xlmr": {
        "hf_name": config.XLMR_MODEL_NAME,
        "lr": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "weight_decay": config.WEIGHT_DECAY,
    },
    "bilstm": {
        "hf_name": None,
        "lr": config.BILSTM_LR,
        "batch_size": config.BILSTM_BATCH_SIZE,
        "epochs": config.BILSTM_EPOCHS,
        "weight_decay": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Transformer baseline training (mBERT / XLM-R)
# ---------------------------------------------------------------------------


def train_transformer(model_key: str) -> dict:
    """
    Fine-tune a transformer baseline (mBERT or XLM-R) on SmSA.

    Parameters
    ----------
    model_key : str
        One of ``"mbert"`` or ``"xlmr"``.

    Returns
    -------
    dict
        Test set evaluation results.
    """
    info = MODEL_REGISTRY[model_key]
    hf_name = info["hf_name"]

    print(f"\n{'=' * 60}")
    print(f"Training {model_key.upper()} ({hf_name})")
    print(f"{'=' * 60}\n")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=config.NUM_LABELS
    )
    model.to(config.DEVICE)

    print("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(tokenizer)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    best_dir = str(config.CHECKPOINTS_DIR / f"best_{model_key}")
    run_dir = str(config.CHECKPOINTS_DIR / f"run_{model_key}")

    steps_per_epoch = len(train_ds) // info["batch_size"]
    total_steps = steps_per_epoch * info["epochs"]
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=info["epochs"],
        per_device_train_batch_size=info["batch_size"],
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=info["lr"],
        weight_decay=info["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=config.RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save best model
    print(f"Saving best model to {best_dir}")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    _print_and_save_results(model_key, test_results)
    return test_results


# ---------------------------------------------------------------------------
# BiLSTM training
# ---------------------------------------------------------------------------


def train_bilstm() -> dict:
    """
    Train a BiLSTM classifier with FastText embeddings on SmSA.

    Returns
    -------
    dict
        Test set evaluation results.
    """
    from src.models.bilstm import (
        BiLSTMClassifier,
        BiLSTMDataCollator,
        BiLSTMDataset,
        build_vocab,
        load_fasttext_embeddings,
    )

    info = MODEL_REGISTRY["bilstm"]

    print(f"\n{'=' * 60}")
    print("Training BiLSTM with FastText embeddings")
    print(f"{'=' * 60}\n")

    # Load raw data
    raw = load_smsa_dataset()

    def _extract(split):
        texts = [str(ex["text"]) for ex in raw[split]]
        labels = [int(ex["label"]) for ex in raw[split]]
        return texts, labels

    train_texts, train_labels = _extract("train")
    val_texts, val_labels = _extract("validation")
    test_texts, test_labels = _extract("test")

    # Build vocabulary from training set
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=2)
    print(f"  Vocabulary size: {len(vocab)}")

    # Load FastText embeddings
    embedding_matrix = load_fasttext_embeddings(vocab)

    # Build datasets
    train_ds = BiLSTMDataset(train_texts, train_labels, vocab)
    val_ds = BiLSTMDataset(val_texts, val_labels, vocab)
    test_ds = BiLSTMDataset(test_texts, test_labels, vocab)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Build model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        pretrained_embeddings=embedding_matrix,
    )
    model.to(config.DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  BiLSTM parameters: {num_params:,}")

    best_dir = str(config.CHECKPOINTS_DIR / "best_bilstm")
    run_dir = str(config.CHECKPOINTS_DIR / "run_bilstm")

    steps_per_epoch = len(train_ds) // info["batch_size"]
    total_steps = steps_per_epoch * info["epochs"]
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=info["epochs"],
        per_device_train_batch_size=info["batch_size"],
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=info["lr"],
        weight_decay=info["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=config.RANDOM_SEED,
        fp16=False,  # BiLSTM doesn't benefit much from fp16
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=BiLSTMDataCollator(),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save best model
    print(f"Saving best model to {best_dir}")
    os.makedirs(best_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "vocab_size": len(vocab),
                "embedding_dim": config.BILSTM_EMBEDDING_DIM,
                "hidden_dim": config.BILSTM_HIDDEN_DIM,
                "num_layers": config.BILSTM_NUM_LAYERS,
                "num_labels": config.NUM_LABELS,
                "dropout": config.BILSTM_DROPOUT,
            },
        },
        os.path.join(best_dir, "bilstm_model.pt"),
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    _print_and_save_results("bilstm", test_results)
    return test_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_and_save_results(model_key: str, results: dict) -> None:
    """Print and save test results to JSON."""
    print(f"\n=== {model_key.upper()} Test Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_path = config.RESULTS_DIR / f"test_results_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for SmSA sentiment classification."
    )
    parser.add_argument(
        "--model",
        choices=["mbert", "xlmr", "bilstm", "all"],
        required=True,
        help="Which baseline model to train.",
    )
    args = parser.parse_args()

    set_seed()

    if args.model == "all":
        models_to_train = ["bilstm", "mbert", "xlmr"]
    else:
        models_to_train = [args.model]

    all_results = {}
    for model_key in models_to_train:
        if model_key == "bilstm":
            results = train_bilstm()
        else:
            results = train_transformer(model_key)
        all_results[model_key] = results

    # Save combined results summary
    if len(all_results) > 1:
        summary_path = config.RESULTS_DIR / "baseline_results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
