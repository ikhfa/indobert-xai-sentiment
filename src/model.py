"""
IndoBERT fine-tuning wrapper for 3-class sentiment classification.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_tokenizer(model_name: str = config.MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Load IndoBERT tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    PreTrainedTokenizerBase
        Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def build_model(
    model_name: str = config.MODEL_NAME,
    num_labels: int = config.NUM_LABELS,
    from_checkpoint: Optional[str] = None,
) -> PreTrainedModel:
    """
    Build IndoBERT classification model.

    Loads ``indobenchmark/indobert-base-p1`` with a linear classification head
    on top of the [CLS] token representation. All layers are fine-tuned.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    num_labels : int
        Number of output classes (3 for negative/neutral/positive).
    from_checkpoint : str, optional
        Path to a saved checkpoint directory. If provided, loads weights from there.

    Returns
    -------
    PreTrainedModel
        Model ready for training or inference.
    """
    if from_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(from_checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
    return model


def load_best_model(
    checkpoint_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load the best saved checkpoint and its tokenizer.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Path to checkpoint directory. Defaults to ``config.CHECKPOINTS_DIR / 'best'``.

    Returns
    -------
    Tuple[PreTrainedModel, PreTrainedTokenizerBase]
        (model, tokenizer) placed on ``config.DEVICE``.
    """
    if checkpoint_dir is None:
        checkpoint_dir = str(config.CHECKPOINTS_DIR / "best")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.to(config.DEVICE)
    model.eval()
    return model, tokenizer


def predict_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list,
    batch_size: int = 32,
) -> "torch.Tensor":
    """
    Run inference and return class probabilities.

    Parameters
    ----------
    model : PreTrainedModel
        Fine-tuned IndoBERT model.
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    texts : list of str
        Input texts.
    batch_size : int
        Inference batch size.

    Returns
    -------
    torch.Tensor
        Shape (N, num_labels) probability tensor on CPU.
    """
    from src.dataset import normalize_indonesian

    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = [normalize_indonesian(t) for t in texts[i: i + batch_size]]
        enc = tokenizer(
            batch_texts,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(config.DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)

    return torch.cat(all_probs, dim=0)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = build_model()
    model.to(config.DEVICE)

    sample = "Aplikasi ini sangat bagus dan mudah digunakan!"
    probs = predict_proba(model, tokenizer, [sample])
    pred = probs.argmax(dim=-1).item()
    print(f"Text: {sample}")
    print(f"Probs: {probs.numpy()}")
    print(f"Prediction: {config.LABEL_MAP[pred]}")
