"""
Dataset loading and preprocessing for IndoNLU SmSA sentiment analysis.
"""

import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

# Common Indonesian emoticons to preserve as single tokens
_EMOTICON_PATTERN = re.compile(
    r"(?::\)|:\(|:D|:P|;\)|:o|:O|<3|>\.<|:\||:\/|B\)|:'\(|>\w+<|"
    r"\^_?\^|\*_?\*|T_?T|>_?<|~_?~|=_?=|o_?o|O_?O)"
)

# Repeated character normalization: 3+ identical chars → 2
_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")


def normalize_indonesian(text: str) -> str:
    """
    Normalize informal Indonesian text for IndoBERT input.

    Steps:
    1. Lowercase
    2. Preserve emoticons as placeholder tokens
    3. Normalize repeated characters (e.g. 'bagussss' → 'baguss', capped at 2)
    4. Collapse extra whitespace

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Normalized text.
    """
    text = text.lower()

    # Extract and preserve emoticons
    emoticons: List[str] = []

    def _replace_emoticon(m: re.Match) -> str:
        idx = len(emoticons)
        emoticons.append(m.group())
        return f" __EMO{idx}__ "

    text = _EMOTICON_PATTERN.sub(_replace_emoticon, text)

    # Normalize repeated characters
    text = _REPEAT_PATTERN.sub(r"\1\1", text)

    # Restore emoticons
    for i, emo in enumerate(emoticons):
        text = text.replace(f"__emo{i}__", emo)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# HuggingFace Dataset Loading
# ---------------------------------------------------------------------------

def load_smsa_dataset() -> DatasetDict:
    """
    Load IndoNLU SmSA dataset from HuggingFace.

    Returns
    -------
    DatasetDict
        Keys: 'train', 'validation', 'test'.
    """
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    return dataset


# ---------------------------------------------------------------------------
# PyTorch Dataset Wrapper
# ---------------------------------------------------------------------------

class SmSADataset(Dataset):
    """
    PyTorch Dataset wrapping IndoNLU SmSA splits with tokenization and preprocessing.

    Parameters
    ----------
    texts : List[str]
        Raw text samples.
    labels : List[int]
        Integer labels (0=negative, 1=neutral, 2=positive).
    tokenizer : PreTrainedTokenizerBase
        IndoBERT tokenizer.
    max_len : int
        Maximum sequence length (default from config).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = config.MAX_SEQ_LEN,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._encodings = self._tokenize_all()

    def _tokenize_all(self) -> Dict:
        normalized = [normalize_indonesian(t) for t in self.texts]
        return self.tokenizer(
            normalized,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
            "token_type_ids": self._encodings.get(
                "token_type_ids",
                torch.zeros_like(self._encodings["input_ids"][idx]),
            )[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def build_datasets(
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple["SmSADataset", "SmSADataset", "SmSADataset"]:
    """
    Load SmSA from HuggingFace and return train/val/test SmSADataset instances.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer to use for encoding.

    Returns
    -------
    Tuple[SmSADataset, SmSADataset, SmSADataset]
        (train_dataset, val_dataset, test_dataset)
    """
    raw = load_smsa_dataset()

    def _extract(split: str) -> Tuple[List[str], List[int]]:
        texts = [str(ex["text"]) for ex in raw[split]]
        labels = [int(ex["label"]) for ex in raw[split]]
        return texts, labels

    train_texts, train_labels = _extract("train")
    val_texts, val_labels = _extract("validation")
    test_texts, test_labels = _extract("test")

    train_ds = SmSADataset(train_texts, train_labels, tokenizer)
    val_ds = SmSADataset(val_texts, val_labels, tokenizer)
    test_ds = SmSADataset(test_texts, test_labels, tokenizer)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds, val_ds, test_ds = build_datasets(tokenizer)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    sample = train_ds[0]
    print("Sample keys:", list(sample.keys()))
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["labels"].item(), "→", config.LABEL_MAP[sample["labels"].item()])
